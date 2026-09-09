"""Tests for RangeShift threshold diagnostics and selection."""

import numpy as np
import pytest

from rangeshift.threshold import evaluate_thresholds, select_threshold


def test_threshold_selection_finds_perfect_validation_cutoff() -> None:
    y_true = [0, 0, 1, 1]
    probabilities = [0.10, 0.40, 0.60, 0.90]

    result = select_threshold(
        y_true,
        probabilities,
        method="tss",
        thresholds=[0.30, 0.50, 0.70],
    )

    assert result.threshold == 0.50
    assert result.score == pytest.approx(1.0)
    assert result.metrics["sensitivity"] == pytest.approx(1.0)
    assert result.metrics["specificity"] == pytest.approx(1.0)
    assert result.metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_threshold_table_reports_expected_metrics() -> None:
    table = evaluate_thresholds(
        [0, 0, 1, 1, 1],
        [0.05, 0.45, 0.55, 0.75, 0.95],
        thresholds=[0.50],
    )

    assert list(table["threshold"]) == [0.50]
    assert table.loc[0, "sensitivity"] == pytest.approx(1.0)
    assert table.loc[0, "specificity"] == pytest.approx(1.0)
    assert table.loc[0, "tss"] == pytest.approx(1.0)
    assert table.loc[0, "youden_j"] == pytest.approx(table.loc[0, "tss"])


def test_threshold_selection_supports_multiple_criteria() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.25, 0.55, 0.45, 0.70, 0.95])

    for method in ("tss", "youden_j", "f1", "balanced_accuracy"):
        result = select_threshold(y_true, probabilities, method=method)
        assert 0.0 < result.threshold < 1.0
        assert np.isfinite(result.score)


def test_threshold_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="both binary classes"):
        evaluate_thresholds([1, 1], [0.5, 0.8])

    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_thresholds([0, 1], [0.2, 1.2])

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        evaluate_thresholds([0, 1], [0.2, 0.8], thresholds=[0.0, 0.5])

    with pytest.raises(ValueError, match="Unsupported"):
        select_threshold([0, 1], [0.2, 0.8], method="accuracy")
