from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from rangeshift.explainability import (
    partial_dependence_table,
    plot_response_curves,
    shap_importance_table,
)


def _fitted_model_and_frame():
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(
        {
            "bio1": rng.normal(13.0, 2.5, 80),
            "bio12": rng.normal(900.0, 140.0, 80),
        }
    )
    signal = -(frame["bio1"] - 12.5) ** 2 + (frame["bio12"] - 850.0) / 30.0
    target = (signal + rng.normal(0.0, 2.0, 80) > 0.0).astype(int)
    model = RandomForestClassifier(n_estimators=30, random_state=4)
    model.fit(frame, target)
    return model, frame


def test_partial_dependence_table_and_plot(tmp_path):
    model, frame = _fitted_model_and_frame()
    table = partial_dependence_table(
        model,
        frame,
        ["bio1", "bio12"],
        grid_resolution=10,
    )

    assert set(table["feature"]) == {"bio1", "bio12"}
    assert table["partial_dependence"].notna().all()

    pytest.importorskip("matplotlib")
    output = plot_response_curves(table, tmp_path / "response_curves.png", dpi=100)
    assert output.exists()
    assert output.stat().st_size > 0


def test_shap_importance_for_random_forest():
    pytest.importorskip("shap")
    model, frame = _fitted_model_and_frame()

    table = shap_importance_table(
        model,
        frame,
        ["bio1", "bio12"],
        max_samples=30,
    )

    assert list(table.columns) == ["feature", "mean_abs_shap"]
    assert set(table["feature"]) == {"bio1", "bio12"}
    assert (table["mean_abs_shap"] >= 0.0).all()
