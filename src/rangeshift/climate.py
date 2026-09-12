"""Climate-data helpers for reproducible RangeShift projection workflows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import numpy as np

WORLDCLIM_CMIP6_BASE = "https://geodata.ucdavis.edu/cmip6"
WORLDCLIM_CMIP6_PERIODS = {
    "2021-2040",
    "2041-2060",
    "2061-2080",
    "2081-2100",
}
WORLDCLIM_CMIP6_SSPS = {"ssp126", "ssp245", "ssp370", "ssp585"}
_WORLDCLIM_GCM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def worldclim_cmip6_bioc_url(
    gcm: str,
    ssp: str,
    period: str,
    *,
    resolution: str = "10m",
) -> str:
    """Build an official WorldClim 2.1 CMIP6 bioclimatic-raster URL.

    RangeShift currently supports the documented 10-minute archive for automated
    retrieval because it is compact enough for a reproducible public case study.
    """
    if resolution != "10m":
        raise ValueError("Automated WorldClim CMIP6 retrieval currently supports resolution='10m'.")
    if ssp not in WORLDCLIM_CMIP6_SSPS:
        raise ValueError(
            "ssp must be one of: " + ", ".join(sorted(WORLDCLIM_CMIP6_SSPS))
        )
    if period not in WORLDCLIM_CMIP6_PERIODS:
        raise ValueError(
            "period must be one of: " + ", ".join(sorted(WORLDCLIM_CMIP6_PERIODS))
        )
    if not _WORLDCLIM_GCM_PATTERN.fullmatch(gcm):
        raise ValueError("gcm contains unsupported characters.")

    filename = f"wc2.1_{resolution}_bioc_{gcm}_{ssp}_{period}.tif"
    return f"{WORLDCLIM_CMIP6_BASE}/{resolution}/{gcm}/{ssp}/{filename}"


def crop_raster_band_to_bounds(
    source_path: str | Path,
    output_path: str | Path,
    *,
    bounds: tuple[float, float, float, float],
    band: int = 1,
) -> Path:
    """Crop one raster band to geographic bounds while preserving its native grid."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise ImportError("Install RangeShift with the 'geo' extra for climate rasters.") from exc

    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    left, bottom, right, top = bounds
    if not (left < right and bottom < top):
        raise ValueError("bounds must be (left, bottom, right, top) with positive extent.")

    with rasterio.open(source_path) as source:
        if band < 1 or band > source.count:
            raise ValueError(
                f"Requested band {band} is outside source band range 1..{source.count}."
            )
        window = from_bounds(left, bottom, right, top, transform=source.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(
            rasterio.windows.Window(0, 0, source.width, source.height)
        )
        if window.width <= 0 or window.height <= 0:
            raise ValueError("Requested bounds do not overlap the source raster.")
        values = source.read(band, window=window)
        profile = source.profile.copy()
        profile.update(
            count=1,
            height=int(window.height),
            width=int(window.width),
            transform=source.window_transform(window),
            compress="deflate",
        )
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(values, 1)
    return output_path


def align_raster_band_to_reference(
    source_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    band: int = 1,
) -> Path:
    """Resample one continuous raster band exactly onto a reference raster grid."""
    try:
        import rasterio
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:
        raise ImportError("Install RangeShift with the 'geo' extra for climate rasters.") from exc

    source_path = Path(source_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as reference, rasterio.open(source_path) as source:
        if band < 1 or band > source.count:
            raise ValueError(
                f"Requested band {band} is outside source band range 1..{source.count}."
            )
        if source.crs is None or reference.crs is None:
            raise ValueError("Source and reference rasters must both define a CRS.")

        nodata = source.nodata
        destination_nodata = -9999.0 if nodata is None else float(nodata)
        destination = np.full(
            (reference.height, reference.width),
            destination_nodata,
            dtype=np.float32,
        )
        reproject(
            source=rasterio.band(source, band),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            dst_nodata=destination_nodata,
            resampling=Resampling.bilinear,
        )

        profile = reference.profile.copy()
        profile.update(
            count=1,
            dtype="float32",
            nodata=destination_nodata,
            compress="deflate",
        )
        with rasterio.open(output_path, "w", **profile) as output:
            output.write(destination, 1)
    return output_path


def extract_bioclim_bands_to_reference(
    source_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path,
    *,
    features: Mapping[str, int],
) -> dict[str, Path]:
    """Extract selected bands from a WorldClim bioclim raster onto one shared grid."""
    if not features:
        raise ValueError("features must contain at least one bioclimatic variable.")
    output_dir = Path(output_dir)
    outputs: dict[str, Path] = {}
    for feature, band in features.items():
        if not feature or int(band) < 1:
            raise ValueError("features must map non-empty names to positive band numbers.")
        path = output_dir / f"{feature}.tif"
        outputs[feature] = align_raster_band_to_reference(
            source_path,
            reference_path,
            path,
            band=int(band),
        )
    return outputs


def validate_aligned_rasters(raster_paths: Mapping[str, str | Path]) -> None:
    """Require an exact shared grid across a set of raster layers."""
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("Install RangeShift with the 'geo' extra for climate rasters.") from exc

    if not raster_paths:
        raise ValueError("At least one raster is required for alignment validation.")
    reference = None
    for label, path in raster_paths.items():
        with rasterio.open(path) as dataset:
            signature = (dataset.shape, dataset.transform, dataset.crs)
            if reference is None:
                reference = signature
            elif signature != reference:
                raise ValueError(f"Raster {label!r} is not aligned to the shared grid.")
