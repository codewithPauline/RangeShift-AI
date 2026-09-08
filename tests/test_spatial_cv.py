"""Tests for repeated spatial cross-validation."""

import numpy as np
import pandas as pd

from rangeshift.spatial_cv import spatial_cross_validate


def make_frame() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows: list[dict[str, float | int]] = []
    centers = [
        (30.0, -100.0),
        (30.0, -96.0),
        (34.0, -100.0),
        (34.0, -96.0),
        (38.0, -100.0),
        (38.0, -96.0),
        (42.0, -100.0),
        (42.0, -96.0),
    ]

    for latitude, longitude in centers:
        for presence in (0, 0, 1, 1, 0, 1):
            rows.append(
                {
                    "presence": presence,
                    "bio1": 12.0 + 2.5 * presence + rng.normal(0, 0.4),
                    "bio12": 650.0 + 120.0 * presence + rng.normal(0, 20),
                    "elevation": 180.0 + 40.0 * presence + rng.normal(0, 8),
                    "latitude": latitude + rng.normal(0, 0.05),
                    "longitude": longitude + rng.normal(0, 0.05),
                }
            )

    return pd.DataFrame(rows)


def test_repeated_spatial_cross_validation_returns_fold_summary() -> None:
    frame = make_frame()
    result = spatial_cross_validate(
        frame,
        feature_columns=["bio1", "bio12", "elevation"],
        block_size_degrees=2.0,
        n_splits=4,
        test_size=0.25,
        random_state=5,
        n_estimators=30,
    )

    assert result.requested_splits == 4
    assert result.valid_splits == 4
    assert result.occupied_blocks >= 4
    assert len(result.fold_metrics) == 4
    assert set(result.mean_metrics) == {"roc_auc", "accuracy", "precision", "recall", "f1"}
    assert set(result.std_metrics) == {"roc_auc", "accuracy", "precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in result.mean_metrics.values())
    assert (result.fold_metrics["train_blocks"] > 0).all()
    assert (result.fold_metrics["test_blocks"] > 0).all()
