from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rangeshift.model_selection import save_selected_model_bundle, tune_and_compare_models
from rangeshift.prediction import load_model_bundle, predict_suitability


def _frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for group in range(8):
        for replicate in range(6):
            bio1 = rng.normal(10 + group * 0.5, 1.0)
            bio12 = rng.normal(900 - group * 20, 40)
            elevation = rng.normal(250 + group * 10, 25)
            signal = -0.6 * bio1 + 0.006 * bio12 + 0.004 * elevation
            presence = int(signal + rng.normal(0, 0.6) > 0.0)
            rows.append(
                {
                    "bio1": bio1,
                    "bio12": bio12,
                    "elevation": elevation,
                    "presence": presence,
                    "group": f"g{group}",
                    "replicate": replicate,
                }
            )
    frame = pd.DataFrame(rows)
    if set(frame["presence"].unique()) != {0, 1}:
        raise AssertionError("Synthetic model-selection data lost one class.")
    return frame


def _small_grids():
    return {
        "random_forest": {
            "n_estimators": [20],
            "max_depth": [None, 4],
            "min_samples_leaf": [1],
            "max_features": ["sqrt"],
        },
        "gradient_boosting": {
            "n_estimators": [20],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2],
            "min_samples_leaf": [1],
        },
    }


def test_tune_and_compare_models_returns_two_model_results(tmp_path):
    frame = _frame()
    result = tune_and_compare_models(
        frame,
        feature_columns=["bio1", "bio12", "elevation"],
        groups=frame["group"],
        n_splits=4,
        param_grids=_small_grids(),
        n_jobs=1,
    )

    assert result.best_model_name in {"random_forest", "gradient_boosting"}
    assert 0.0 <= result.best_score <= 1.0
    assert set(result.cv_results["model"]) == {"random_forest", "gradient_boosting"}
    assert result.spatial_groups_used is True

    model_path = save_selected_model_bundle(result, tmp_path / "selected.joblib")
    bundle = load_model_bundle(model_path)
    predictions = predict_suitability(frame.head(5), bundle)
    assert predictions.between(0.0, 1.0).all()


def test_model_selection_rejects_incomplete_param_grid():
    frame = _frame()

    with pytest.raises(ValueError, match="exactly"):
        tune_and_compare_models(
            frame,
            feature_columns=["bio1", "bio12", "elevation"],
            n_splits=3,
            param_grids={"random_forest": {"n_estimators": [10]}},
            n_jobs=1,
        )
