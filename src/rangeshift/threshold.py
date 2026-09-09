"""Threshold diagnostics and selection for RangeShift AI.

Thresholds are selected from validation predictions, not from final test data.
This keeps the decision rule separate from the data used for final reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

SUPPORTED_THRESHOLD_METHODS = {
    "tss",
    "youden_j",
    "f1",
    "balanced_accuracy",
}


@dataclass
class ThresholdSelectionResult:
    """Selected threshold plus the full diagnostic table."""

    threshold: float
    method: str
    score: float
    metrics: dict[str, float]
    table: pd.DataFrame


def _validate_binary_inputs(
    y_true: Iterable[int],
    probabilities: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)

    if y.ndim != 1 or p.ndim != 1:
        raise ValueError("y_true and probabilities must be one-dimensional.")
    if y.size == 0:
        raise ValueError("Threshold evaluation requires at least one observation.")
    if y.size != p.size:
        raise ValueError("y_true and probabilities must contain the same number of values.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("y_true must contain both binary classes 0 and 1.")
    if not np.isfinite(p).all():
        raise ValueError("probabilities must contain only finite values.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must fall between 0 and 1.")

    return y, p


def _default_thresholds() -> np.ndarray:
    return np.round(np.arange(0.01, 1.0, 0.01), 2)


def evaluate_thresholds(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Evaluate binary classification performance across candidate thresholds."""
    y, p = _validate_binary_inputs(y_true, probabilities)
    candidates = (
        _default_thresholds()
        if thresholds is None
        else np.asarray(list(thresholds), dtype=float)
    )

    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("thresholds must contain at least one value.")
    if not np.isfinite(candidates).all():
        raise ValueError("thresholds must contain only finite values.")
    if np.any((candidates <= 0.0) | (candidates >= 1.0)):
        raise ValueError("Every threshold must be strictly between 0 and 1.")

    rows: list[dict[str, float]] = []
    for threshold in np.unique(candidates):
        predicted = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        accuracy = (tp + tn) / y.size
        f1 = (
            2.0 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity)
            else 0.0
        )
        tss = sensitivity + specificity - 1.0
        balanced_accuracy = (sensitivity + specificity) / 2.0

        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "recall": float(sensitivity),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_accuracy),
                "tss": float(tss),
                "youden_j": float(tss),
                "predicted_positive_rate": float(predicted.mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("threshold", ignore_index=True)


def select_threshold(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    method: str = "tss",
    thresholds: Iterable[float] | None = None,
) -> ThresholdSelectionResult:
    """Select a threshold using validation predictions and an explicit criterion.

    Supported methods are TSS, Youden's J, F1, and balanced accuracy. Ties are
    broken by choosing the candidate closest to 0.5, then the smaller threshold.
    """
    if method not in SUPPORTED_THRESHOLD_METHODS:
        supported = ", ".join(sorted(SUPPORTED_THRESHOLD_METHODS))
        raise ValueError(f"Unsupported threshold method '{method}'. Choose from: {supported}.")

    table = evaluate_thresholds(y_true, probabilities, thresholds=thresholds)
    ranked = table.assign(distance_from_half=(table["threshold"] - 0.5).abs()).sort_values(
        [method, "distance_from_half", "threshold"],
        ascending=[False, True, True],
        ignore_index=True,
    )
    selected = ranked.iloc[0]
    metrics = {
        key: float(selected[key])
        for key in (
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "balanced_accuracy",
            "tss",
            "youden_j",
            "predicted_positive_rate",
        )
    }

    return ThresholdSelectionResult(
        threshold=float(selected["threshold"]),
        method=method,
        score=float(selected[method]),
        metrics=metrics,
        table=table,
    )
