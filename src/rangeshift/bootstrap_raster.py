"""Train bootstrap model ensembles and project model-fit variability to rasters."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .data import validate_training_frame
from .model_raster_uncertainty import ModelRasterUncertaintyResult, summarize_model_fit_rasters


@dataclass
class BootstrapRasterRun:
    """Raster summaries and a machine-readable record of the resampling design."""

    raster: ModelRasterUncertaintyResult
    manifest_path: Path


def run_bootstrap_raster_uncertainty(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    layers: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    target_column: str = "presence",
    training_groups: Sequence[str] | None = None,
    n_resamples: int = 20,
    n_estimators: int = 300,
    random_state: int = 42,
    threshold: float | None = None,
    window_size: int = 256,
) -> BootstrapRasterRun:
    """Fit bootstrap forests and produce descriptive model-fit spread rasters.

    Requires caller-provided training data, spatial blocks where appropriate,
    and a separately validated threshold if agreement is requested. Does not
    estimate climate uncertainty, occupancy probabilities, or interval coverage.
    """
    features = list(feature_columns)
    validate_training_frame(training_frame, features, target_column)
    if not features or len(set(features)) != len(features):
        raise ValueError("feature_columns must be nonempty and unique.")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")
    if training_groups is not None and len(training_groups) != len(training_frame):
        raise ValueError("training_groups must match the training row count.")

    y = training_frame[target_column].to_numpy(dtype=int)
    rng = np.random.default_rng(random_state)
    groups = None if training_groups is None else np.asarray(training_groups)
    if groups is not None:
        if pd.isna(groups).any():
            raise ValueError("training_groups cannot contain missing labels.")
        unique = pd.unique(groups)
        if len(unique) < 2:
            raise ValueError("Spatial-group bootstrap needs at least two groups.")
        blocks = [np.flatnonzero(groups == group) for group in unique]
    else:
        classes = [np.flatnonzero(y == target) for target in (0, 1)]

    bundles = []
    draw_sizes = []
    for replicate in range(n_resamples):
        for _attempt in range(100):
            if groups is None:
                selected = np.concatenate([
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in classes
                ])
            else:
                draws = rng.integers(0, len(blocks), size=len(blocks))
                selected = np.concatenate([blocks[int(j)] for j in draws])
            if np.unique(y[selected]).size == 2:
                break
        else:
            raise ValueError(
                "Unable to draw two-class block bootstrap within 100 attempts; "
                "check spatial grouping and class balance."
            )
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state + replicate,
            n_jobs=-1,
        )
        estimator.fit(training_frame.iloc[selected][features], y[selected])
        bundles.append({"model": estimator, "feature_columns": features})
        draw_sizes.append(int(len(selected)))

    directory = Path(output_dir)
    result = summarize_model_fit_rasters(
        bundles,
        layers,
        directory / "rasters",
        threshold=threshold,
        window_size=window_size,
    )
    manifest_path = directory / "bootstrap_manifest.json"
    manifest = {
        "n_resamples": n_resamples,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "features": features,
        "target_column": target_column,
        "training_row_count": len(training_frame),
        "resampling_unit": "spatial_group" if groups is not None else "stratified_row",
        "n_groups": None if groups is None else len(blocks),
        "bootstrap_draw_row_counts": draw_sizes,
        "threshold": threshold,
        "raster_outputs": {
            "mean": str(result.mean_path),
            "sample_sd": str(result.sd_path),
            "q025": str(result.q025_path),
            "q975": str(result.q975_path),
            "threshold_agreement": (
                str(result.agreement_path) if result.agreement_path is not None else None
            ),
        },
        "interpretation": (
            "Descriptive variation across bootstrapped Random Forest fits, "
            "not future climate uncertainty, calibrated occupancy, or "
            "coverage-guaranteed prediction intervals."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return BootstrapRasterRun(raster=result, manifest_path=manifest_path)
