"""Tests for calibrated RangeShift habitat-suitability models."""

from pathlib import Path

import numpy as np
import pandas as pd

from rangeshift.calibration import (
    save_calibrated_model_bundle,
    train_calibrated_habitat_model,
)
from rangeshift.prediction import load_model_bundle, predict_suitability


def _synthetic_frame(n: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(77)
    bio1 = rng.normal(14.0, 3.0, n)
    bio12 = rng.normal(900.0, 180.0, n)
    elevation = rng.normal(300.0, 120.0, n)
    signal = (
        -((bio1 - 13.5) ** 2) / 8.0
        + (bio12 - 850.0) / 260.0
        + (elevation - 250.0) / 500.0
        + rng.normal(0.0, 0.75, n)
    )
    presence = (signal > np.median(signal)).astype(int)
    return pd.DataFrame(
        {
            "presence": presence,
            "bio1": bio1,
            "bio12": bio12,
            "elevation": elevation,
        }
    )


def test_calibrated_training_uses_three_disjoint_splits() -> None:
    frame = _synthetic_frame()
    result = train_calibrated_habitat_model(
        frame,
        feature_columns=["bio1", "bio12", "elevation"],
        n_estimators=40,
        calibration_cv=3,
        calibration_bins=6,
        random_state=19,
    )

    assert sum(result.split_sizes.values()) == len(frame)
    assert all(size > 0 for size in result.split_sizes.values())
    assert 0.0 < result.selected_threshold < 1.0
    assert result.threshold_method == "tss"
    assert result.calibration_method == "sigmoid"
    assert set(result.test_probability_metrics) == {
        "roc_auc",
        "brier_score",
        "log_loss",
    }
    assert result.calibration_table.shape[0] <= 6
    assert not result.threshold_table.empty


def test_calibrated_bundle_reloads_and_predicts(tmp_path: Path) -> None:
    frame = _synthetic_frame()
    result = train_calibrated_habitat_model(
        frame,
        feature_columns=["bio1", "bio12", "elevation"],
        n_estimators=30,
        calibration_cv=3,
        random_state=23,
    )

    model_path = tmp_path / "calibrated.joblib"
    saved = save_calibrated_model_bundle(result, model_path)
    bundle = load_model_bundle(saved)

    predictions = predict_suitability(frame.head(12), bundle)
    assert len(predictions) == 12
    assert predictions.between(0.0, 1.0).all()
    assert bundle["selected_threshold"] == result.selected_threshold
    assert bundle["threshold_method"] == result.threshold_method
    assert bundle["calibration_method"] == result.calibration_method


def test_calibration_reports_fixed_threshold_test_metrics() -> None:
    result = train_calibrated_habitat_model(
        _synthetic_frame(),
        feature_columns=["bio1", "bio12", "elevation"],
        threshold_method="balanced_accuracy",
        calibration_method="sigmoid",
        calibration_cv=3,
        n_estimators=35,
        random_state=31,
    )

    metrics = result.test_classification_metrics
    assert metrics["threshold"] == result.selected_threshold
    for name in (
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "accuracy",
        "balanced_accuracy",
    ):
        assert 0.0 <= metrics[name] <= 1.0
