"""Windowed raster summaries across independently fitted suitability models.

Model-fit variability differs from variation across future climate scenarios.
All supplied models must share predictor ordering and a compatible suitability scale.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .raster import (
    _output_profile,
    _require_rasterio,
    _validate_feature_mapping,
    _validate_open_datasets,
)


@dataclass
class ModelRasterUncertaintyResult:
    """Paths to descriptive, between-fit raster summaries."""

    mean_path: Path
    sd_path: Path
    q025_path: Path
    q975_path: Path
    agreement_path: Path | None
    model_count: int
    valid_cells: int
    threshold: float | None


def summarize_model_fit_rasters(
    models: Sequence[dict],
    layer_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    threshold: float | None = None,
    window_size: int = 256,
    nodata: float = -9999.0,
) -> ModelRasterUncertaintyResult:
    """Project fitted-model ensemble onto identical predictor grid in windows.

    Each model bundle needs `model` (sklearn-style predict_proba) and ordered
    `feature_columns`. Results are descriptive fit-to-fit statistics, not
    calibrated probabilities of actual occupancy or coverage-guaranteed intervals.
    If provided, threshold must be chosen *outside* this projection workflow and
    shared across all fits. Model predictions must already be comparable.
    """
    if len(models) < 2:
        raise ValueError("At least two independently fitted models are required.")
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie between 0 and 1.")
    if not np.isfinite(nodata) or 0.0 <= nodata <= 1.0:
        raise ValueError("nodata must be finite and outside the suitability range.")

    features = list(models[0]["feature_columns"])
    if not features or len(features) != len(set(features)):
        raise ValueError("Feature names must be nonempty and unique.")
    for bundle in models:
        if list(bundle["feature_columns"]) != features:
            raise ValueError("All models must use exactly the same ordered features.")
        if not hasattr(bundle["model"], "predict_proba"):
            raise ValueError("Every fitted model must expose predict_proba.")
    _validate_feature_mapping(layer_paths, features)

    rasterio = _require_rasterio()
    from rasterio.windows import Window

    directory = Path(output_dir)
    paths = {key: Path(layer_paths[key]) for key in features}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing raster for {key}: {path}")
    output_paths = {
        "mean": directory / "model_fit_mean.tif",
        "sd": directory / "model_fit_sd.tif",
        "q025": directory / "model_fit_q025.tif",
        "q975": directory / "model_fit_q975.tif",
    }
    if threshold is not None:
        output_paths["agreement"] = directory / "model_fit_suitable_fraction.tif"

    valid_cells = 0
    with ExitStack() as stack:
        sources = {
            feature: stack.enter_context(rasterio.open(paths[feature]))
            for feature in features
        }
        profile = _validate_open_datasets(sources, features)
        height, width = sources[features[0]].shape
        directory.mkdir(parents=True, exist_ok=True)
        outputs = {
            name: stack.enter_context(
                rasterio.open(path, "w", **_output_profile(profile, nodata))
            )
            for name, path in output_paths.items()
        }
        for name, dataset in outputs.items():
            dataset.set_band_description(1, name)
            dataset.update_tags(
                rangeshift_output="model_fit_variability",
                model_count=str(len(models)),
                statistic=name,
            )
        for row in range(0, height, window_size):
            for col in range(0, width, window_size):
                window = Window(
                    col, row, min(window_size, width - col),
                    min(window_size, height - row),
                )
                arrays = {}
                valid = np.ones((int(window.height), int(window.width)), dtype=bool)
                for feature in features:
                    band = sources[feature].read(1, window=window, masked=True)
                    values = np.asarray(band.filled(np.nan), dtype=float)
                    valid &= ~np.ma.getmaskarray(band) & np.isfinite(values)
                    arrays[feature] = values
                computed = {}
                if valid.any():
                    frame = pd.DataFrame({
                        feature: arrays[feature][valid] for feature in features
                    })
                    predictions = []
                    for bundle in models:
                        model = bundle["model"]
                        classes = list(model.classes_)
                        if 1 not in classes:
                            raise ValueError("Model must expose positive target class 1.")
                        values = np.asarray(model.predict_proba(frame), dtype=float)
                        scores = values[:, classes.index(1)]
                        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                            raise ValueError("Model outputs must be finite values in [0, 1].")
                        predictions.append(scores)
                    ensemble = np.stack(predictions)
                    computed = {
                        "mean": ensemble.mean(axis=0),
                        "sd": ensemble.std(axis=0, ddof=1),
                        "q025": np.quantile(ensemble, 0.025, axis=0),
                        "q975": np.quantile(ensemble, 0.975, axis=0),
                    }
                    if threshold is not None:
                        computed["agreement"] = (ensemble >= threshold).mean(axis=0)
                    valid_cells += int(valid.sum())
                for name, dataset in outputs.items():
                    result = np.full(valid.shape, nodata, dtype=np.float32)
                    if computed:
                        result[valid] = computed[name].astype(np.float32)
                    dataset.write(result, 1, window=window)
    if valid_cells == 0:
        for path in output_paths.values():
            path.unlink(missing_ok=True)
        raise ValueError("Predictor rasters have no complete valid cells.")
    return ModelRasterUncertaintyResult(
        mean_path=output_paths["mean"],
        sd_path=output_paths["sd"],
        q025_path=output_paths["q025"],
        q975_path=output_paths["q975"],
        agreement_path=output_paths.get("agreement"),
        model_count=len(models),
        valid_cells=valid_cells,
        threshold=threshold,
    )
