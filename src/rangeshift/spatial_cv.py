"""Repeated spatial cross-validation for RangeShift AI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .data import validate_training_frame
from .spatial import assign_spatial_blocks


@dataclass
class SpatialCVResult:
    """Summary and fold-level metrics from repeated spatial holdouts."""

    fold_metrics: pd.DataFrame
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    requested_splits: int
    valid_splits: int
    occupied_blocks: int


def _calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def cross_validate_spatial_blocks(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    blocks: pd.Series,
    target_column: str = "presence",
    *,
    n_splits: int = 5,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 300,
) -> SpatialCVResult:
    """Run repeated grouped holdouts using precomputed spatial block labels."""
    feature_columns = list(feature_columns)
    validate_training_frame(frame, feature_columns, target_column)

    if not blocks.index.equals(frame.index):
        raise ValueError("Spatial block labels must have the same index as the input frame.")
    if blocks.isna().any():
        raise ValueError("Spatial block labels cannot contain missing values.")
    if blocks.nunique() < 2:
        raise ValueError("Spatial cross-validation requires at least two occupied blocks.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")

    X = frame[feature_columns]
    y = frame[target_column].astype(int)
    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )

    rows: list[dict[str, float | int]] = []
    for split_number, (train_indices, test_indices) in enumerate(
        splitter.split(X, y, groups=blocks),
        start=1,
    ):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
        if set(y_train.unique()) != {0, 1} or set(y_test.unique()) != {0, 1}:
            continue

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state + split_number,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X.iloc[train_indices], y_train)
        probabilities = model.predict_proba(X.iloc[test_indices])[:, 1]
        predictions = model.predict(X.iloc[test_indices])
        metrics = _calculate_metrics(y_test, probabilities, predictions)

        rows.append(
            {
                "split": split_number,
                "train_n": len(train_indices),
                "test_n": len(test_indices),
                "train_blocks": blocks.iloc[train_indices].nunique(),
                "test_blocks": blocks.iloc[test_indices].nunique(),
                **metrics,
            }
        )

    if not rows:
        raise ValueError(
            "No valid spatial split contained both binary classes in training and test data."
        )

    fold_metrics = pd.DataFrame(rows)
    metric_names = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    mean_metrics = {
        metric: float(fold_metrics[metric].mean())
        for metric in metric_names
    }
    std_metrics = {
        metric: float(fold_metrics[metric].std(ddof=0))
        for metric in metric_names
    }

    return SpatialCVResult(
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        requested_splits=n_splits,
        valid_splits=len(fold_metrics),
        occupied_blocks=int(blocks.nunique()),
    )


def spatial_cross_validate(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str = "presence",
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    block_size_degrees: float = 1.0,
    n_splits: int = 5,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 300,
) -> SpatialCVResult:
    """Run repeated spatial holdouts using geographic degree-based blocks."""
    blocks = assign_spatial_blocks(
        frame,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        block_size_degrees=block_size_degrees,
    )
    return cross_validate_spatial_blocks(
        frame,
        feature_columns=feature_columns,
        blocks=blocks,
        target_column=target_column,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
    )
