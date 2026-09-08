"""Tests for RangeShift spatial validation utilities."""

import numpy as np
import pandas as pd
import pytest

from rangeshift.spatial import (
    assign_spatial_blocks,
    compare_random_and_spatial,
    evaluate_spatial_holdout,
    validate_coordinates,
)


def make_spatial_frame() -> pd.DataFrame:
    """Create a small dataset with both target classes represented in every block."""
    rows: list[dict[str, float | int]] = []
    rng = np.random.default_rng(42)

    block_centers = [
        (30.2, -100.2),
        (30.2, -96.2),
        (34.2, -100.2),
        (34.2, -96.2),
        (38.2, -100.2),
        (38.2, -96.2),
        (42.2, -100.2),
        (42.2, -96.2),
    ]

    for latitude, longitude in block_centers:
        for presence in (0, 0, 1, 1):
            climate_signal = 2.0 * presence + rng.normal(0, 0.2)
            rows.append(
                {
                    "presence": presence,
                    "bio1": 10.0 + climate_signal,
                    "bio12": 700.0 + 80.0 * presence + rng.normal(0, 10),
                    "elevation": 150.0 + 30.0 * presence + rng.normal(0, 5),
                    "latitude": latitude + rng.normal(0, 0.03),
                    "longitude": longitude + rng.normal(0, 0.03),
                }
            )

    return pd.DataFrame(rows)


def test_assign_spatial_blocks_is_deterministic() -> None:
    frame = make_spatial_frame()
    first = assign_spatial_blocks(frame, block_size_degrees=2.0)
    second = assign_spatial_blocks(frame, block_size_degrees=2.0)

    assert first.equals(second)
    assert first.nunique() >= 4


def test_spatial_holdout_keeps_blocks_separate() -> None:
    frame = make_spatial_frame()
    result = evaluate_spatial_holdout(
        frame,
        feature_columns=["bio1", "bio12", "elevation"],
        block_size_degrees=2.0,
        test_size=0.25,
        random_state=7,
        n_estimators=50,
    )

    assert set(result.train_blocks).isdisjoint(result.test_blocks)
    assert set(result.metrics) == {"roc_auc", "accuracy", "precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in result.metrics.values())
    assert len(result.train_indices) + len(result.test_indices) == len(frame)


def test_compare_random_and_spatial_returns_metric_deltas() -> None:
    frame = make_spatial_frame()
    result = compare_random_and_spatial(
        frame,
        feature_columns=["bio1", "bio12", "elevation"],
        block_size_degrees=2.0,
        test_size=0.25,
        random_state=12,
        n_estimators=50,
    )

    assert set(result.spatial_minus_random) == {
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    }


def test_coordinate_validation_rejects_invalid_latitude() -> None:
    frame = make_spatial_frame()
    frame.loc[0, "latitude"] = 91.0

    with pytest.raises(ValueError, match="Latitude"):
        validate_coordinates(frame)


def test_spatial_holdout_requires_more_than_one_block() -> None:
    frame = make_spatial_frame().iloc[:4].copy()
    frame["latitude"] = 35.0
    frame["longitude"] = -95.0

    with pytest.raises(ValueError, match="at least two occupied spatial blocks"):
        evaluate_spatial_holdout(
            frame,
            feature_columns=["bio1", "bio12", "elevation"],
            block_size_degrees=5.0,
            n_estimators=10,
        )
