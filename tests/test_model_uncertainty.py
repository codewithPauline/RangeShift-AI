import numpy as np
import pandas as pd
import pytest

from rangeshift.model_uncertainty import evaluate_model_resampling_uncertainty


def _frames():
    rng = np.random.default_rng(7)
    x = rng.normal(size=120)
    y = np.array([0, 1] * 60)
    frame = pd.DataFrame({"bio1": x + y, "presence": y})
    return frame.iloc[:80].copy(), frame.iloc[80:].copy()


def test_row_bootstrap_is_reproducible_and_matches_evaluation_index():
    training, evaluation = _frames()
    first = evaluate_model_resampling_uncertainty(
        training, evaluation, ["bio1"], n_resamples=3, n_estimators=12, random_state=3
    )
    second = evaluate_model_resampling_uncertainty(
        training, evaluation, ["bio1"], n_resamples=3, n_estimators=12, random_state=3
    )
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert first.predictions.index.equals(evaluation.index)
    assert len(first.replicate_metrics) == 3
    assert first.resampling_unit == "stratified_row"
    assert ((first.predictions >= 0) & (first.predictions <= 1)).all().all()


def test_group_bootstrap_and_input_guardrails():
    training, evaluation = _frames()
    groups = np.array([str(i // 10) for i in range(len(training))])
    result = evaluate_model_resampling_uncertainty(
        training, evaluation, ["bio1"], training_groups=groups,
        n_resamples=3, n_estimators=10
    )
    assert result.resampling_unit == "spatial_group"
    with pytest.raises(ValueError, match="n_resamples"):
        evaluate_model_resampling_uncertainty(
            training, evaluation, ["bio1"], n_resamples=1
        )
    with pytest.raises(ValueError, match="row count"):
        evaluate_model_resampling_uncertainty(
            training, evaluation, ["bio1"], training_groups=["a"]
        )
