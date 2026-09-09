"""Model interpretation utilities for RangeShift AI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence


def partial_dependence_table(
    model,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    grid_resolution: int = 25,
) -> pd.DataFrame:
    """Return one-dimensional partial-dependence response curves as a table."""
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("At least one feature is required for partial dependence.")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    if grid_resolution < 2:
        raise ValueError("grid_resolution must be at least 2.")

    X = frame[feature_columns]
    rows = []
    for feature_index, feature in enumerate(feature_columns):
        result = partial_dependence(
            model,
            X,
            features=[feature_index],
            kind="average",
            grid_resolution=grid_resolution,
            response_method="auto",
        )
        grid_values = result.get("grid_values", result.get("values"))
        if grid_values is None:
            raise RuntimeError("Unsupported scikit-learn partial-dependence result format.")
        values = np.asarray(grid_values[0], dtype=float)
        average = np.asarray(result["average"])
        response = average.reshape(-1)
        if len(response) != len(values):
            response = average[0].reshape(-1)
        rows.extend(
            {
                "feature": feature,
                "feature_value": float(value),
                "partial_dependence": float(effect),
            }
            for value, effect in zip(values, response, strict=True)
        )
    return pd.DataFrame(rows)


def shap_importance_table(
    model,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    max_samples: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate mean absolute Tree SHAP attribution for supported tree models."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP interpretation requires the optional 'explain' dependency. "
            "Install RangeShift with pip install -e '.[explain]'."
        ) from exc

    feature_columns = list(feature_columns)
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1.")

    X = frame[feature_columns]
    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=random_state)

    try:
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(X)
    except Exception as exc:
        raise ValueError(
            "Tree SHAP could not explain this estimator. Use a supported fitted tree model "
            "such as RangeShift's Random Forest or Gradient Boosting model."
        ) from exc

    if isinstance(raw_values, list):
        values = np.asarray(raw_values[-1])
    else:
        values = np.asarray(raw_values)
        if values.ndim == 3:
            values = values[:, :, -1]
    if values.ndim != 2 or values.shape[1] != len(feature_columns):
        raise RuntimeError("Unexpected SHAP output shape for the supplied estimator.")

    mean_absolute = np.mean(np.abs(values), axis=0)
    return (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "mean_abs_shap": mean_absolute,
            }
        )
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    )


def plot_response_curves(
    table: pd.DataFrame,
    output_path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Render partial-dependence response curves from ``partial_dependence_table``."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Response-curve plotting requires Matplotlib. Install the 'geo' optional extra."
        ) from exc

    required = {"feature", "feature_value", "partial_dependence"}
    if not required.issubset(table.columns):
        raise ValueError("Response-curve table is missing required columns.")
    if dpi < 72:
        raise ValueError("dpi must be at least 72.")

    features = table["feature"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(features),
        1,
        figsize=(7.5, max(3.0, 2.8 * len(features))),
        squeeze=False,
    )
    for axis, feature in zip(axes[:, 0], features, strict=True):
        subset = table.loc[table["feature"] == feature].sort_values("feature_value")
        axis.plot(subset["feature_value"], subset["partial_dependence"])
        axis.set_xlabel(feature)
        axis.set_ylabel("Partial dependence")
        axis.set_title(f"Response curve: {feature}")

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path
