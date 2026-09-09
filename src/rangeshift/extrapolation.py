"""Environmental novelty and extrapolation diagnostics for RangeShift AI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExtrapolationResult:
    """Row-level and feature-level environmental novelty diagnostics."""

    row_diagnostics: pd.DataFrame
    feature_summary: pd.DataFrame
    training_envelope: pd.DataFrame


def diagnose_extrapolation(
    training_frame: pd.DataFrame,
    projection_frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> ExtrapolationResult:
    """Flag projection values outside the univariate training envelope.

    This is intentionally transparent rather than a full multivariate MESS
    implementation. Each predictor is compared with its observed training range,
    and each projection row records how many predictors require extrapolation.
    """
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("At least one feature is required for extrapolation diagnostics.")

    for label, frame in (("training", training_frame), ("projection", projection_frame)):
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{label.title()} data are missing: {', '.join(missing)}")
        if frame[feature_columns].isna().any().any():
            raise ValueError(f"{label.title()} predictors cannot contain missing values.")
        non_numeric = [
            column
            for column in feature_columns
            if not pd.api.types.is_numeric_dtype(frame[column])
        ]
        if non_numeric:
            raise ValueError(f"Predictors must be numeric: {', '.join(non_numeric)}")

    minima = training_frame[feature_columns].min()
    maxima = training_frame[feature_columns].max()
    spans = (maxima - minima).replace(0.0, np.nan)

    projection = projection_frame[feature_columns]
    below = projection.lt(minima, axis="columns")
    above = projection.gt(maxima, axis="columns")
    novel = below | above

    lower_excess = minima - projection
    upper_excess = projection - maxima
    excess = lower_excess.where(below, 0.0).clip(lower=0.0)
    excess = excess + upper_excess.where(above, 0.0).clip(lower=0.0)
    normalized_excess = excess.div(spans, axis="columns").fillna(
        excess.gt(0.0).astype(float)
    )

    row_diagnostics = pd.DataFrame(index=projection_frame.index)
    row_diagnostics["novel_feature_count"] = novel.sum(axis=1).astype(int)
    row_diagnostics["novel_feature_fraction"] = novel.mean(axis=1).astype(float)
    row_diagnostics["max_normalized_excess"] = normalized_excess.max(axis=1).astype(float)
    row_diagnostics["requires_extrapolation"] = row_diagnostics["novel_feature_count"] > 0

    feature_summary = pd.DataFrame(
        {
            "feature": feature_columns,
            "training_min": minima.to_numpy(dtype=float),
            "training_max": maxima.to_numpy(dtype=float),
            "projection_min": projection.min().to_numpy(dtype=float),
            "projection_max": projection.max().to_numpy(dtype=float),
            "fraction_below_training": below.mean().to_numpy(dtype=float),
            "fraction_above_training": above.mean().to_numpy(dtype=float),
            "fraction_outside_training": novel.mean().to_numpy(dtype=float),
        }
    )
    envelope = pd.DataFrame(
        {
            "feature": feature_columns,
            "training_min": minima.to_numpy(dtype=float),
            "training_max": maxima.to_numpy(dtype=float),
        }
    )
    return ExtrapolationResult(
        row_diagnostics=row_diagnostics,
        feature_summary=feature_summary,
        training_envelope=envelope,
    )
