"""Model-fit variability on a fixed, independent evaluation set.

Bootstrap refits quantify sensitivity to training-sample composition, not
uncertainty from climate scenarios or calibrated occupancy probability.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from .data import validate_training_frame


@dataclass
class ResamplingUncertaintyResult:
    """Per-row predictions and diagnostics across independently refitted models."""

    predictions: pd.DataFrame
    replicate_metrics: pd.DataFrame
    n_resamples: int
    resampling_unit: str


def evaluate_model_resampling_uncertainty(
    training_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    target_column: str = "presence",
    n_resamples: int = 20,
    n_estimators: int = 300,
    random_state: int = 42,
    training_groups: Sequence[str] | None = None,
) -> ResamplingUncertaintyResult:
    """Refit balanced forests on bootstrap training samples and predict fixed rows.

    Training and evaluation rows must be disjoint by the user's study design.
    Optional `training_groups` resample entire spatial blocks with replacement;
    otherwise rows are resampled independently within each target class. The
    prediction quantiles describe model-fit variability conditional on the
    chosen data/model, not frequentist coverage or occupancy probabilities.
    """
    features = list(feature_columns)
    validate_training_frame(training_frame, features, target_column)
    validate_training_frame(evaluation_frame, features, target_column)
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")
    if features != list(dict.fromkeys(features)):
        raise ValueError("feature_columns must be unique.")
    if training_groups is not None and len(training_groups) != len(training_frame):
        raise ValueError("training_groups must match the training row count.")

    y = training_frame[target_column].to_numpy(dtype=int)
    eval_y = evaluation_frame[target_column].to_numpy(dtype=int)
    if set(np.unique(eval_y)) != {0, 1}:
        raise ValueError("evaluation_frame must contain both target classes for ROC-AUC.")

    rng = np.random.default_rng(random_state)
    group_labels = None if training_groups is None else np.asarray(training_groups)
    if group_labels is not None:
        if pd.isna(group_labels).any():
            raise ValueError("training_groups cannot contain missing labels.")
        unique_groups = np.unique(group_labels)
        if len(unique_groups) < 2:
            raise ValueError("Block bootstrap needs at least two distinct groups.")
        group_indices = {group: np.flatnonzero(group_labels == group) for group in unique_groups}
    else:
        class_indices = {label: np.flatnonzero(y == label) for label in (0, 1)}

    probabilities: list[np.ndarray] = []
    metrics: list[dict[str, float | int]] = []
    for replicate in range(n_resamples):
        for _attempt in range(100):
            if group_labels is None:
                indices = np.concatenate(
                    [
                        rng.choice(rows, size=len(rows), replace=True)
                        for rows in class_indices.values()
                    ]
                )
            else:
                chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                indices = np.concatenate([group_indices[group] for group in chosen])
            if set(np.unique(y[indices])) == {0, 1}:
                break
        else:
            raise ValueError(
                "Unable to draw a two-class bootstrap sample after 100 attempts. "
                "Use a different spatial grouping or add more mixed/independent blocks."
            )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state + replicate,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(training_frame.iloc[indices][features], y[indices])
        scores = model.predict_proba(evaluation_frame[features])[:, 1]
        probabilities.append(scores)
        metrics.append(
            {
                "replicate": replicate,
                "training_rows_with_replacement": int(len(indices)),
                "roc_auc": float(roc_auc_score(eval_y, scores)),
            }
        )

    stack = np.stack(probabilities, axis=0)
    predictions = pd.DataFrame(
        {
            "mean_suitability": stack.mean(axis=0),
            "sd_suitability": stack.std(axis=0, ddof=1),
            "q025_suitability": np.quantile(stack, 0.025, axis=0),
            "q975_suitability": np.quantile(stack, 0.975, axis=0),
        },
        index=evaluation_frame.index,
    )
    return ResamplingUncertaintyResult(
        predictions=predictions,
        replicate_metrics=pd.DataFrame(metrics),
        n_resamples=n_resamples,
        resampling_unit="spatial_group" if group_labels is not None else "stratified_row",
    )
