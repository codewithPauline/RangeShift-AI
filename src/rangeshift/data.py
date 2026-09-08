"""Data loading and validation utilities for RangeShift AI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def load_occurrence_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV table containing a binary target and environmental predictors."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input table does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("RangeShift v0.1 currently accepts CSV input only.")
    return pd.read_csv(path)


def validate_training_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> None:
    """Validate the minimum requirements for binary habitat-suitability training."""
    if frame.empty:
        raise ValueError("Training data is empty.")

    if not feature_columns:
        raise ValueError("At least one environmental predictor is required.")

    required = [target_column, *feature_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    duplicated = [column for column in feature_columns if feature_columns.count(column) > 1]
    if duplicated:
        raise ValueError("Feature column names must be unique.")

    if frame[required].isna().any().any():
        raise ValueError("Target and predictor columns cannot contain missing values in v0.1.")

    target_values = set(frame[target_column].unique().tolist())
    if target_values != {0, 1}:
        raise ValueError(
            f"Target column '{target_column}' must contain both binary classes 0 and 1; "
            f"found {sorted(target_values)}."
        )

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(
            "Environmental predictors must be numeric. Non-numeric columns: "
            + ", ".join(non_numeric)
        )

    class_counts = frame[target_column].value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each target class must contain at least two observations.")
