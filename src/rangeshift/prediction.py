"""Prediction utilities for trained RangeShift AI models."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def load_model_bundle(path: str | Path) -> dict:
    """Load a RangeShift model bundle saved during training."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle does not exist: {path}")

    bundle = joblib.load(path)
    required = {"model", "feature_columns", "target_column", "metrics"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Invalid RangeShift model bundle; missing: {sorted(missing)}")
    return bundle


def predict_suitability(frame: pd.DataFrame, bundle: dict) -> pd.Series:
    """Predict habitat-suitability probabilities for rows of environmental data."""
    feature_columns = list(bundle["feature_columns"])
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction data is missing predictors: {', '.join(missing)}")

    predictors = frame[feature_columns]
    if predictors.isna().any().any():
        raise ValueError("Prediction predictors cannot contain missing values in v0.1.")

    probabilities = bundle["model"].predict_proba(predictors)[:, 1]
    return pd.Series(probabilities, index=frame.index, name="suitability")
