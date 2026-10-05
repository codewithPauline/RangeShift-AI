"""Balanced crossed climate-scenario x fitted-model raster uncertainty.

With equal weights, total variance equals between-scenario variance of the
model-mean predictions plus mean within-scenario between-model variance.
Components are descriptive, not probabilistic forecast uncertainty.
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
class CrossedUncertaintyResult:
    """Cellwise decomposition of crossed fitted-model and scenario variability."""

    mean_path: Path
    total_sd_path: Path
    scenario_sd_path: Path
    model_sd_path: Path
    suitable_fraction_path: Path | None
    model_count: int
    scenario_count: int
    valid_cells: int
    threshold: float | None


def summarize_crossed_uncertainty(
    models: Sequence[dict],
    scenarios: Mapping[str, Mapping[str, str | Path]],
    output_dir: str | Path,
    *,
    threshold: float | None = None,
    window_size: int = 256,
    nodata: float = -9999.0,
) -> CrossedUncertaintyResult:
    """Project every model into every scenario and decompose equal-weight spread.

    A common cell is valid only if *every* scenario has finite predictor values.
    Scenario labels represent a supplied, equally weighted ensemble, not a
    probability distribution over future climates. Each model must have been
    fitted independently using scientifically compatible settings; this
    routine does not refit, calibrate, or validate supplied models.
    """
    if len(models) < 2 or len(scenarios) < 2:
        raise ValueError("Require at least two models and two future scenarios.")
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must lie between 0 and 1.")
    if not np.isfinite(nodata) or 0 <= nodata <= 1:
        raise ValueError("nodata must be finite and outside suitability range.")

    features = list(models[0]["feature_columns"])
    if not features or len(set(features)) != len(features):
        raise ValueError("Model feature names must be nonempty and unique.")
    for bundle in models:
        if list(bundle["feature_columns"]) != features:
            raise ValueError("All model feature sequences must match exactly.")
        model = bundle["model"]
        if not hasattr(model, "predict_proba") or 1 not in list(model.classes_):
            raise ValueError("Every model needs predict_proba and positive class 1.")
    for name, layers in scenarios.items():
        if not name:
            raise ValueError("Scenario labels must not be empty.")
        _validate_feature_mapping(layers, features)

    rasterio = _require_rasterio()
    from rasterio.windows import Window

    root = Path(output_dir)
    paths = {
        "mean": root / "crossed_mean.tif",
        "total_sd": root / "crossed_total_sd.tif",
        "scenario_sd": root / "crossed_scenario_sd.tif",
        "model_sd": root / "crossed_model_sd.tif",
    }
    if threshold is not None:
        paths["suitable_fraction"] = root / "crossed_suitable_fraction.tif"
    valid_cells = 0

    with ExitStack() as stack:
        opened = {}
        common_grid = None
        for name, layers in scenarios.items():
            datasets = {
                feature: stack.enter_context(rasterio.open(Path(layers[feature])))
                for feature in features
            }
            profile = _validate_open_datasets(datasets, features)
            ref = datasets[features[0]]
            grid = (ref.shape, ref.transform, ref.crs)
            if common_grid is None:
                common_grid = grid
                common_profile = profile
            elif grid != common_grid:
                raise ValueError(
                    "Every future scenario must have identical shape, transform, and CRS."
                )
            opened[name] = datasets
        height, width = common_grid[0]
        root.mkdir(parents=True, exist_ok=True)
        outputs = {
            name: stack.enter_context(
                rasterio.open(path, "w", **_output_profile(common_profile, nodata))
            )
            for name, path in paths.items()
        }
        for name, dest in outputs.items():
            dest.set_band_description(1, name)
            dest.update_tags(
                rangeshift_output="crossed_model_scenario_variability",
                scenario_count=str(len(scenarios)),
                model_count=str(len(models)),
            )
        for row in range(0, height, window_size):
            for col in range(0, width, window_size):
                window = Window(
                    col, row, min(window_size, width - col),
                    min(window_size, height - row),
                )
                valid = np.ones((int(window.height), int(window.width)), dtype=bool)
                frames = []
                scenario_arrays = []
                for datasets in opened.values():
                    values = {}
                    for feature, dataset in datasets.items():
                        band = dataset.read(1, window=window, masked=True)
                        arr = np.asarray(band.filled(np.nan), dtype=float)
                        valid &= ~np.ma.getmaskarray(band) & np.isfinite(arr)
                        values[feature] = arr
                    scenario_arrays.append(values)
                calculated = {}
                if valid.any():
                    for values in scenario_arrays:
                        frames.append(
                            pd.DataFrame({
                                feature: values[feature][valid] for feature in features
                            })
                        )
                    predictions = []
                    for frame in frames:
                        fitted = []
                        for bundle in models:
                            model = bundle["model"]
                            index = list(model.classes_).index(1)
                            proba = np.asarray(model.predict_proba(frame), dtype=float)
                            scores = proba[:, index]
                            if (not np.isfinite(scores).all()
                                or ((scores < 0) | (scores > 1)).any()):
                                raise ValueError("Model scores must be finite within [0, 1].")
                            fitted.append(scores)
                        predictions.append(np.stack(fitted))
                    cube = np.stack(predictions)  # scenario x model x valid cells
                    mean = cube.mean(axis=(0, 1))
                    scenario_means = cube.mean(axis=1)
                    scenario_var = scenario_means.var(axis=0, ddof=0)
                    within_model_var = cube.var(axis=1, ddof=0).mean(axis=0)
                    total_var = cube.var(axis=(0, 1), ddof=0)
                    calculated = {
                        "mean": mean,
                        "total_sd": np.sqrt(total_var),
                        "scenario_sd": np.sqrt(scenario_var),
                        "model_sd": np.sqrt(within_model_var),
                    }
                    if threshold is not None:
                        calculated["suitable_fraction"] = (cube >= threshold).mean(axis=(0, 1))
                    valid_cells += int(valid.sum())
                for name, dest in outputs.items():
                    data = np.full(valid.shape, nodata, dtype=np.float32)
                    if calculated:
                        data[valid] = calculated[name].astype(np.float32)
                    dest.write(data, 1, window=window)

    if valid_cells == 0:
        for path in paths.values():
            path.unlink(missing_ok=True)
        raise ValueError("No common valid predictor cells across future scenarios.")
    return CrossedUncertaintyResult(
        mean_path=paths["mean"],
        total_sd_path=paths["total_sd"],
        scenario_sd_path=paths["scenario_sd"],
        model_sd_path=paths["model_sd"],
        suitable_fraction_path=paths.get("suitable_fraction"),
        model_count=len(models),
        scenario_count=len(scenarios),
        valid_cells=valid_cells,
        threshold=threshold,
    )
