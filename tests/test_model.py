"""Tests for the core RangeShift AI training and prediction workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rangeshift.data import validate_training_frame
from rangeshift.model import save_model_bundle, train_habitat_model
from rangeshift.prediction import load_model_bundle, predict_suitability


def make_training_frame(n: int = 240, seed: int = 7) -> pd.DataFrame:
    """Create a deterministic synthetic ecological classification dataset."""
    rng = np.random.default_rng(seed)
    temperature = rng.normal(14.0, 4.0, n)
    precipitation = rng.normal(950.0, 180.0, n)
    elevation = rng.uniform(50.0, 850.0, n)
    noise = rng.normal(0.0, 0.7, n)

    suitability_signal = (
        -0.45 * np.abs(temperature - 13.0)
        + 0.004 * (precipitation - 850.0)
        + 0.001 * elevation
        + noise
    )
    presence = (suitability_signal > np.median(suitability_signal)).astype(int)

    return pd.DataFrame(
        {
            "presence": presence,
            "bio1": temperature,
            "bio12": precipitation,
            "elevation": elevation,
        }
    )


def test_train_habitat_model_returns_valid_outputs() -> None:
    frame = make_training_frame()
    result = train_habitat_model(
        frame,
        feature_columns=["bio1", "bio12", "elevation"],
        random_state=42,
        n_estimators=100,
    )

    assert set(result.metrics) == {"roc_auc", "accuracy", "precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in result.metrics.values())
    assert list(result.feature_importance.columns) == ["feature", "importance"]
    assert set(result.feature_importance["feature"]) == {"bio1", "bio12", "elevation"}
    assert result.feature_importance["importance"].sum() == pytest.approx(1.0)


def test_model_bundle_can_predict_suitability(tmp_path) -> None:
    frame = make_training_frame()
    result = train_habitat_model(
        frame,
        feature_columns=["bio1", "bio12", "elevation"],
        n_estimators=50,
    )

    model_path = tmp_path / "model.joblib"
    save_model_bundle(result, model_path)
    bundle = load_model_bundle(model_path)
    predictions = predict_suitability(frame.head(12), bundle)

    assert len(predictions) == 12
    assert predictions.name == "suitability"
    assert predictions.between(0.0, 1.0).all()


def test_validation_rejects_missing_predictor() -> None:
    frame = make_training_frame()

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_training_frame(frame, ["bio1", "not_a_column"], "presence")


def test_validation_requires_binary_target() -> None:
    frame = make_training_frame()
    frame["presence"] = 1

    with pytest.raises(ValueError, match="both binary classes"):
        validate_training_frame(frame, ["bio1", "bio12"], "presence")
