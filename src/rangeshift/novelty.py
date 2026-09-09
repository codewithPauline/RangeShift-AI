"""Novel-climate warnings for environmental transfer in RangeShift AI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .extrapolation import ExtrapolationResult, diagnose_extrapolation


@dataclass
class NovelClimateResult:
    """Combined univariate extrapolation and multivariate novelty diagnostics."""

    row_diagnostics: pd.DataFrame
    feature_summary: pd.DataFrame
    training_envelope: pd.DataFrame
    multivariate_distance_threshold: float
    distance_quantile: float


def diagnose_novel_climate(
    training_frame: pd.DataFrame,
    projection_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    distance_quantile: float = 0.99,
) -> NovelClimateResult:
    """Warn when projection environments are outside the training support.

    Two transparent checks are combined:

    1. univariate envelope extrapolation from :func:`diagnose_extrapolation`;
    2. standardized nearest-neighbor distance in multivariate environmental space.

    The multivariate warning threshold is learned from leave-one-out nearest-neighbor
    distances among training observations, using ``distance_quantile``.
    """
    if not 0.5 < distance_quantile < 1.0:
        raise ValueError("distance_quantile must be between 0.5 and 1.0.")

    features = list(feature_columns)
    if len(training_frame) < 3:
        raise ValueError("Novel-climate diagnostics require at least three training rows.")

    extrapolation: ExtrapolationResult = diagnose_extrapolation(
        training_frame,
        projection_frame,
        features,
    )

    training = training_frame[features].astype(float)
    projection = projection_frame[features].astype(float)
    means = training.mean(axis=0)
    standard_deviations = training.std(axis=0, ddof=0).replace(0.0, 1.0)

    training_scaled = (training - means) / standard_deviations
    projection_scaled = (projection - means) / standard_deviations

    training_neighbors = NearestNeighbors(n_neighbors=2, metric="euclidean")
    training_neighbors.fit(training_scaled)
    training_distances, _ = training_neighbors.kneighbors(training_scaled)
    leave_one_out_distance = training_distances[:, 1]
    threshold = float(np.quantile(leave_one_out_distance, distance_quantile))

    projection_neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean")
    projection_neighbors.fit(training_scaled)
    projection_distance, _ = projection_neighbors.kneighbors(projection_scaled)
    projection_distance = projection_distance[:, 0]

    diagnostics = extrapolation.row_diagnostics.copy()
    diagnostics["multivariate_nearest_distance"] = projection_distance.astype(float)
    diagnostics["multivariate_distance_threshold"] = threshold
    diagnostics["multivariate_novel"] = diagnostics["multivariate_nearest_distance"] > threshold
    diagnostics["novel_climate_warning"] = (
        diagnostics["requires_extrapolation"] | diagnostics["multivariate_novel"]
    )

    return NovelClimateResult(
        row_diagnostics=diagnostics,
        feature_summary=extrapolation.feature_summary,
        training_envelope=extrapolation.training_envelope,
        multivariate_distance_threshold=threshold,
        distance_quantile=float(distance_quantile),
    )
