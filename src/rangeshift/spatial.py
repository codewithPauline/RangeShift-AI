"""Spatial validation utilities for RangeShift AI.

The v0.2 spatial split deliberately keeps entire coordinate grid cells together so
training and evaluation samples do not come from the same spatial block. This is a
baseline guard against overly optimistic random train/test splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .data import validate_training_frame
from .model import TrainingResult, train_habitat_model


@dataclass
class SpatialEvaluationResult:
    """Artifacts from one spatially separated train/test evaluation."""

    model: RandomForestClassifier
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_blocks: list[str]
    test_blocks: list[str]
    block_size_degrees: float


@dataclass
class SpatialComparisonResult:
    """Random-split and spatial-split performance for the same dataset."""

    random: TrainingResult
    spatial: SpatialEvaluationResult
    spatial_minus_random: dict[str, float]


def validate_coordinates(
    frame: pd.DataFrame,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
) -> None:
    """Validate WGS84-style latitude and longitude coordinate columns."""
    missing = [
        column
        for column in (latitude_column, longitude_column)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing coordinate columns: {', '.join(missing)}")

    for column in (latitude_column, longitude_column):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Coordinate column '{column}' must be numeric.")
        if frame[column].isna().any():
            raise ValueError(f"Coordinate column '{column}' cannot contain missing values.")

    if not frame[latitude_column].between(-90, 90).all():
        raise ValueError("Latitude values must fall between -90 and 90 degrees.")
    if not frame[longitude_column].between(-180, 180).all():
        raise ValueError("Longitude values must fall between -180 and 180 degrees.")


def assign_spatial_blocks(
    frame: pd.DataFrame,
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    block_size_degrees: float = 1.0,
) -> pd.Series:
    """Assign observations to non-overlapping latitude/longitude grid cells.

    Notes
    -----
    Degree-based cells are transparent and dependency-light, but they are not
    equal-area spatial units. They are suitable as a first diagnostic rather than
    a final global spatial-validation strategy.
    """
    validate_coordinates(frame, latitude_column, longitude_column)
    if block_size_degrees <= 0 or block_size_degrees > 180:
        raise ValueError("block_size_degrees must be greater than 0 and at most 180.")

    lat_bins = np.floor((frame[latitude_column].to_numpy() + 90.0) / block_size_degrees)
    lon_bins = np.floor((frame[longitude_column].to_numpy() + 180.0) / block_size_degrees)

    blocks = pd.Series(
        [f"lat{int(lat)}_lon{int(lon)}" for lat, lon in zip(lat_bins, lon_bins, strict=True)],
        index=frame.index,
        name="spatial_block",
    )
    return blocks


def _metrics(y_true: pd.Series, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Calculate the binary classification metrics used across RangeShift."""
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def evaluate_spatial_holdout(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str = "presence",
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    block_size_degrees: float = 1.0,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 300,
    max_split_attempts: int = 100,
) -> SpatialEvaluationResult:
    """Evaluate a Random Forest while holding complete spatial blocks out.

    The splitter tries multiple grouped partitions and selects the first split in
    which both the training and test partitions contain both binary classes.
    """
    feature_columns = list(feature_columns)
    validate_training_frame(frame, feature_columns, target_column)
    validate_coordinates(frame, latitude_column, longitude_column)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")
    if max_split_attempts < 1:
        raise ValueError("max_split_attempts must be at least 1.")

    blocks = assign_spatial_blocks(
        frame,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        block_size_degrees=block_size_degrees,
    )
    if blocks.nunique() < 2:
        raise ValueError(
            "Spatial evaluation requires at least two occupied spatial blocks. "
            "Use a smaller block size or a dataset with broader geographic coverage."
        )

    X = frame[feature_columns]
    y = frame[target_column].astype(int)
    splitter = GroupShuffleSplit(
        n_splits=max_split_attempts,
        test_size=test_size,
        random_state=random_state,
    )

    selected: tuple[np.ndarray, np.ndarray] | None = None
    for train_indices, test_indices in splitter.split(X, y, groups=blocks):
        train_classes = set(y.iloc[train_indices].unique().tolist())
        test_classes = set(y.iloc[test_indices].unique().tolist())
        if train_classes == {0, 1} and test_classes == {0, 1}:
            selected = (train_indices, test_indices)
            break

    if selected is None:
        raise ValueError(
            "Could not create a spatial holdout containing both target classes in training "
            "and test data. Try a smaller block size, a different test size, or more data."
        )

    train_indices, test_indices = selected
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    metrics = _metrics(y_test, probabilities, predictions)

    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False, ignore_index=True)
    )

    train_blocks = sorted(set(blocks.iloc[train_indices].tolist()))
    test_blocks = sorted(set(blocks.iloc[test_indices].tolist()))

    return SpatialEvaluationResult(
        model=model,
        metrics=metrics,
        feature_importance=feature_importance,
        feature_columns=feature_columns,
        target_column=target_column,
        train_indices=train_indices,
        test_indices=test_indices,
        train_blocks=train_blocks,
        test_blocks=test_blocks,
        block_size_degrees=block_size_degrees,
    )


def compare_random_and_spatial(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str = "presence",
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    block_size_degrees: float = 1.0,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 300,
) -> SpatialComparisonResult:
    """Compare conventional random holdout with block-separated spatial holdout."""
    random_result = train_habitat_model(
        frame,
        feature_columns=feature_columns,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    spatial_result = evaluate_spatial_holdout(
        frame,
        feature_columns=feature_columns,
        target_column=target_column,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        block_size_degrees=block_size_degrees,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
    )

    delta = {
        metric: spatial_result.metrics[metric] - random_result.metrics[metric]
        for metric in random_result.metrics
    }
    return SpatialComparisonResult(
        random=random_result,
        spatial=spatial_result,
        spatial_minus_random=delta,
    )


def to_geodataframe(
    frame: pd.DataFrame,
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
):
    """Convert a coordinate table to a GeoPandas GeoDataFrame in EPSG:4326.

    GeoPandas is optional so the lightweight ML core remains installable without
    compiled geospatial dependencies.
    """
    validate_coordinates(frame, latitude_column, longitude_column)
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "GeoPandas is required for GeoDataFrame conversion. "
            "Install RangeShift with the 'geo' optional dependency."
        ) from exc

    return gpd.GeoDataFrame(
        frame.copy(),
        geometry=gpd.points_from_xy(frame[longitude_column], frame[latitude_column]),
        crs="EPSG:4326",
    )
