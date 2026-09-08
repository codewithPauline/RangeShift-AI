"""Baseline habitat-suitability model for RangeShift AI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import validate_training_frame


@dataclass
class TrainingResult:
    """Artifacts returned after training a baseline habitat-suitability model."""

    model: RandomForestClassifier
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    feature_columns: list[str]
    target_column: str


def train_habitat_model(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str = "presence",
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 300,
) -> TrainingResult:
    """Train and evaluate a Random Forest habitat-suitability classifier.

    Parameters
    ----------
    frame:
        Input table containing the binary target and numeric predictors.
    feature_columns:
        Environmental predictor column names.
    target_column:
        Binary response column, where 1 = presence and 0 = absence/background.
    test_size:
        Fraction of observations reserved for evaluation.
    random_state:
        Seed controlling the data split and Random Forest reproducibility.
    n_estimators:
        Number of trees in the Random Forest.
    """
    feature_columns = list(feature_columns)
    validate_training_frame(frame, feature_columns, target_column)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")

    X = frame[feature_columns]
    y = frame[target_column].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }

    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False, ignore_index=True)
    )

    return TrainingResult(
        model=model,
        metrics=metrics,
        feature_importance=feature_importance,
        feature_columns=feature_columns,
        target_column=target_column,
    )


def save_model_bundle(result: TrainingResult, path: str | Path) -> Path:
    """Persist a trained model with the metadata needed for later prediction."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": result.model,
        "feature_columns": result.feature_columns,
        "target_column": result.target_column,
        "metrics": result.metrics,
    }
    joblib.dump(bundle, path)
    return path
