"""Environmental predictor collinearity diagnostics for RangeShift AI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CollinearityResult:
    """Correlation and variance-inflation diagnostics."""

    correlation_matrix: pd.DataFrame
    high_correlation_pairs: pd.DataFrame
    vif_table: pd.DataFrame
    flagged_features: list[str]
    correlation_threshold: float
    vif_threshold: float


def _validate_predictors(frame: pd.DataFrame, feature_columns: Sequence[str]) -> list[str]:
    features = list(feature_columns)
    if len(features) < 2:
        raise ValueError("Collinearity diagnostics require at least two predictors.")
    if len(set(features)) != len(features):
        raise ValueError("feature_columns cannot contain duplicates.")
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(f"Missing predictor columns: {', '.join(missing)}")
    if frame[features].isna().any().any():
        raise ValueError("Predictors cannot contain missing values.")
    non_numeric = [
        feature for feature in features if not pd.api.types.is_numeric_dtype(frame[feature])
    ]
    if non_numeric:
        raise ValueError(f"Predictors must be numeric: {', '.join(non_numeric)}")
    return features


def _variance_inflation_factors(values: np.ndarray, features: list[str]) -> pd.DataFrame:
    rows = []
    for position, feature in enumerate(features):
        y = values[:, position]
        others = np.delete(values, position, axis=1)
        if np.isclose(np.var(y), 0.0):
            vif = float("inf")
            r_squared = 1.0
        else:
            design = np.column_stack([np.ones(len(others)), others])
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            fitted = design @ coefficients
            residual_sum = float(np.sum((y - fitted) ** 2))
            total_sum = float(np.sum((y - np.mean(y)) ** 2))
            r_squared = 1.0 - residual_sum / total_sum
            r_squared = float(np.clip(r_squared, 0.0, 1.0))
            vif = float("inf") if r_squared >= 1.0 - 1e-12 else float(1.0 / (1.0 - r_squared))
        rows.append({"feature": feature, "r_squared": r_squared, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False, ignore_index=True)


def diagnose_collinearity(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    correlation_threshold: float = 0.7,
    vif_threshold: float = 5.0,
) -> CollinearityResult:
    """Report strongly correlated predictor pairs and variance inflation factors."""
    features = _validate_predictors(frame, feature_columns)
    if not 0.0 < correlation_threshold < 1.0:
        raise ValueError("correlation_threshold must be between 0 and 1.")
    if vif_threshold <= 1.0:
        raise ValueError("vif_threshold must be greater than 1.")

    predictors = frame[features].astype(float)
    correlation = predictors.corr(method="pearson")
    pair_rows = []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            value = float(correlation.loc[left, right])
            if abs(value) >= correlation_threshold:
                pair_rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "correlation": value,
                        "absolute_correlation": abs(value),
                    }
                )
    pairs = pd.DataFrame(
        pair_rows,
        columns=["feature_a", "feature_b", "correlation", "absolute_correlation"],
    )
    if not pairs.empty:
        pairs = pairs.sort_values("absolute_correlation", ascending=False, ignore_index=True)

    values = predictors.to_numpy(dtype=float)
    vif_table = _variance_inflation_factors(values, features)
    flagged = set(vif_table.loc[vif_table["vif"] >= vif_threshold, "feature"].tolist())
    if not pairs.empty:
        flagged.update(pairs["feature_a"].tolist())
        flagged.update(pairs["feature_b"].tolist())

    return CollinearityResult(
        correlation_matrix=correlation,
        high_correlation_pairs=pairs,
        vif_table=vif_table,
        flagged_features=sorted(flagged),
        correlation_threshold=float(correlation_threshold),
        vif_threshold=float(vif_threshold),
    )
