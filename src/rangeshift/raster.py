"""Raster-based habitat-suitability prediction for RangeShift AI.

The v0.3 raster engine expects one single-band raster per model predictor. All
layers must already be aligned to the same grid. RangeShift validates that
assumption instead of silently resampling ecological predictors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .prediction import predict_suitability


@dataclass
class RasterStack:
    """Validated in-memory environmental raster stack."""

    arrays: dict[str, np.ndarray]
    valid_mask: np.ndarray
    profile: dict
    feature_columns: list[str]


@dataclass
class RasterPredictionResult:
    """Summary of one raster suitability prediction."""

    output_path: Path
    valid_cells: int
    total_cells: int
    feature_columns: list[str]
    crs: str


def _require_rasterio():
    """Import Rasterio with an actionable optional-dependency error."""
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "Rasterio is required for raster prediction. "
            "Install RangeShift with the 'geo' optional dependency."
        ) from exc
    return rasterio


def parse_layer_specs(specs: Sequence[str]) -> dict[str, Path]:
    """Parse repeated ``FEATURE=PATH`` CLI specifications."""
    if not specs:
        raise ValueError("At least one raster layer specification is required.")

    layers: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid layer specification '{spec}'. Expected FEATURE=PATH."
            )
        feature, raw_path = spec.split("=", 1)
        feature = feature.strip()
        raw_path = raw_path.strip()
        if not feature or not raw_path:
            raise ValueError(
                f"Invalid layer specification '{spec}'. Expected FEATURE=PATH."
            )
        if feature in layers:
            raise ValueError(f"Duplicate raster layer for predictor '{feature}'.")
        layers[feature] = Path(raw_path)
    return layers


def _validate_feature_mapping(
    layer_paths: Mapping[str, str | Path],
    feature_columns: Sequence[str],
) -> None:
    """Require raster predictors to match the trained model exactly."""
    expected = list(feature_columns)
    provided = set(layer_paths)
    missing = [feature for feature in expected if feature not in provided]
    extras = sorted(provided.difference(expected))

    problems: list[str] = []
    if missing:
        problems.append(f"missing predictors: {', '.join(missing)}")
    if extras:
        problems.append(f"unexpected predictors: {', '.join(extras)}")
    if problems:
        raise ValueError("Raster layer mapping does not match model; " + "; ".join(problems))


def load_aligned_raster_stack(
    layer_paths: Mapping[str, str | Path],
    feature_columns: Sequence[str],
) -> RasterStack:
    """Load and validate aligned single-band predictor rasters.

    RangeShift v0.3 is intentionally strict: it does not automatically resample,
    reproject, or crop predictor layers because doing so without explicit choices
    can hide scientifically important preprocessing decisions.
    """
    rasterio = _require_rasterio()
    feature_columns = list(feature_columns)
    _validate_feature_mapping(layer_paths, feature_columns)

    arrays: dict[str, np.ndarray] = {}
    valid_mask: np.ndarray | None = None
    reference_profile: dict | None = None
    reference_shape: tuple[int, int] | None = None
    reference_transform = None
    reference_crs = None

    for feature in feature_columns:
        path = Path(layer_paths[feature])
        if not path.exists():
            raise FileNotFoundError(f"Raster layer does not exist for '{feature}': {path}")

        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise ValueError(
                    f"Raster layer '{feature}' must contain exactly one band; "
                    f"found {dataset.count}."
                )
            if dataset.crs is None:
                raise ValueError(f"Raster layer '{feature}' has no coordinate reference system.")

            shape = (dataset.height, dataset.width)
            if reference_profile is None:
                reference_profile = dataset.profile.copy()
                reference_shape = shape
                reference_transform = dataset.transform
                reference_crs = dataset.crs
                valid_mask = np.ones(shape, dtype=bool)
            else:
                if shape != reference_shape:
                    raise ValueError(
                        f"Raster layer '{feature}' has shape {shape}, expected {reference_shape}."
                    )
                if dataset.transform != reference_transform:
                    raise ValueError(
                        f"Raster layer '{feature}' is not aligned to the reference pixel grid."
                    )
                if dataset.crs != reference_crs:
                    raise ValueError(
                        f"Raster layer '{feature}' CRS {dataset.crs} does not match "
                        f"reference CRS {reference_crs}."
                    )

            masked = dataset.read(1, masked=True)
            values = np.asarray(masked.filled(np.nan), dtype=float)
            layer_valid = ~np.ma.getmaskarray(masked) & np.isfinite(values)
            valid_mask &= layer_valid
            arrays[feature] = values

    if reference_profile is None or valid_mask is None:
        raise ValueError("No raster predictors were loaded.")
    if not valid_mask.any():
        raise ValueError("Raster predictors have no cells with complete finite data.")

    return RasterStack(
        arrays=arrays,
        valid_mask=valid_mask,
        profile=reference_profile,
        feature_columns=feature_columns,
    )


def raster_stack_to_frame(stack: RasterStack) -> pd.DataFrame:
    """Convert valid raster cells to model predictor rows in trained feature order."""
    return pd.DataFrame(
        {
            feature: stack.arrays[feature][stack.valid_mask]
            for feature in stack.feature_columns
        }
    )


def predict_suitability_raster(
    bundle: dict,
    layer_paths: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    nodata: float = -9999.0,
) -> RasterPredictionResult:
    """Predict habitat suitability for aligned environmental raster cells."""
    rasterio = _require_rasterio()
    feature_columns = list(bundle["feature_columns"])
    stack = load_aligned_raster_stack(layer_paths, feature_columns)
    predictors = raster_stack_to_frame(stack)
    suitability = predict_suitability(predictors, bundle).to_numpy(dtype=np.float32)

    output = np.full(stack.valid_mask.shape, nodata, dtype=np.float32)
    output[stack.valid_mask] = suitability

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = stack.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=float(nodata),
        compress="deflate",
    )

    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(output, 1)
        destination.set_band_description(1, "habitat_suitability")
        destination.update_tags(
            rangeshift_output="habitat_suitability_probability",
            suitability_min="0",
            suitability_max="1",
        )

    return RasterPredictionResult(
        output_path=output_path,
        valid_cells=int(stack.valid_mask.sum()),
        total_cells=int(stack.valid_mask.size),
        feature_columns=feature_columns,
        crs=str(stack.profile["crs"]),
    )
