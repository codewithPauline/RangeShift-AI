"""Projected geospatial helpers for RangeShift AI.

These utilities are optional because coordinate projection depends on the geospatial
stack. They provide a more physically interpretable alternative to degree-based
blocks for local and regional studies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spatial import validate_coordinates


@dataclass
class ProjectedBlockResult:
    """Spatial blocks created after projecting WGS84 coordinates to a metric CRS."""

    blocks: pd.Series
    crs_epsg: int
    block_size_km: float


def estimate_local_utm_epsg(
    frame: pd.DataFrame,
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
) -> int:
    """Estimate a UTM EPSG code from the mean coordinate of a regional dataset.

    The helper is intended for local or regional studies contained reasonably well
    within one UTM zone. It rejects latitudes outside the standard UTM coverage.
    """
    validate_coordinates(frame, latitude_column, longitude_column)

    latitude = float(frame[latitude_column].mean())
    longitude = float(frame[longitude_column].mean())
    if latitude < -80 or latitude > 84:
        raise ValueError("Automatic UTM estimation requires latitudes between -80 and 84.")

    zone = int(np.floor((longitude + 180.0) / 6.0)) + 1
    zone = min(60, max(1, zone))
    return (32600 if latitude >= 0 else 32700) + zone


def assign_projected_blocks(
    frame: pd.DataFrame,
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    block_size_km: float = 100.0,
    crs_epsg: int | None = None,
) -> ProjectedBlockResult:
    """Project coordinates and assign them to square blocks measured in kilometers.

    When ``crs_epsg`` is omitted, a local UTM CRS is estimated from the dataset's
    mean coordinate. For studies spanning multiple UTM zones or very large extents,
    users should provide an appropriate projected CRS explicitly.
    """
    validate_coordinates(frame, latitude_column, longitude_column)
    if block_size_km <= 0:
        raise ValueError("block_size_km must be greater than 0.")

    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError(
            "pyproj is required for kilometer-scale projected blocks. "
            "Install RangeShift with the 'geo' optional dependency."
        ) from exc

    resolved_epsg = crs_epsg or estimate_local_utm_epsg(
        frame,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
    )
    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{resolved_epsg}",
        always_xy=True,
    )
    x, y = transformer.transform(
        frame[longitude_column].to_numpy(),
        frame[latitude_column].to_numpy(),
    )

    block_size_m = block_size_km * 1000.0
    x_bins = np.floor(np.asarray(x) / block_size_m).astype(int)
    y_bins = np.floor(np.asarray(y) / block_size_m).astype(int)
    blocks = pd.Series(
        [
            f"epsg{resolved_epsg}_x{x_bin}_y{y_bin}"
            for x_bin, y_bin in zip(x_bins, y_bins, strict=True)
        ],
        index=frame.index,
        name="spatial_block",
    )

    return ProjectedBlockResult(
        blocks=blocks,
        crs_epsg=resolved_epsg,
        block_size_km=block_size_km,
    )
