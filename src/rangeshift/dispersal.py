"""Dispersal-constrained future suitability for RangeShift AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neighbors import BallTree

from .range_shift import (
    _cell_area_grid_km2,
    _cell_centres,
    _load_suitability_pair,
    _require_geo,
    _validate_threshold,
)

DISPERSAL_UNSUITABLE = 0
DISPERSAL_ACCESSIBLE = 1
DISPERSAL_INACCESSIBLE = 2
DISPERSAL_NODATA = 255
EARTH_RADIUS_KM = 6371.0088


@dataclass
class DispersalConstraintResult:
    """Summary of an explicit maximum-distance dispersal constraint."""

    accessibility_path: Path
    constrained_future_path: Path
    threshold: float
    max_distance_km: float
    future_suitable_cells: int
    accessible_suitable_cells: int
    inaccessible_suitable_cells: int
    accessible_fraction: float
    accessible_suitable_area_km2: float
    inaccessible_suitable_area_km2: float
    crs: str

    def to_dict(self) -> dict:
        payload = self.__dict__.copy()
        payload["accessibility_path"] = str(self.accessibility_path)
        payload["constrained_future_path"] = str(self.constrained_future_path)
        return payload


def _geographic_points(mask: np.ndarray, transform, crs) -> np.ndarray:
    """Return masked raster-cell centres as radians in latitude/longitude order."""
    _, _, Transformer = _require_geo()
    x, y = _cell_centres(mask.shape, transform)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(x[mask], y[mask])
    return np.column_stack(
        [
            np.radians(np.asarray(latitude, dtype=float)),
            np.radians(np.asarray(longitude, dtype=float)),
        ]
    )


def apply_dispersal_constraint(
    current_path: str | Path,
    future_path: str | Path,
    accessibility_output_path: str | Path,
    constrained_future_output_path: str | Path,
    *,
    threshold: float,
    max_distance_km: float,
    query_chunk_size: int = 100_000,
) -> DispersalConstraintResult:
    """Limit future suitable habitat to cells reachable within a maximum distance.

    Future cells at or above ``threshold`` are considered accessible only when
    their geodesic distance to the nearest currently suitable cell is no greater
    than ``max_distance_km``. The assumption is explicit and does not imply that
    organisms will actually occupy every reachable cell.
    """
    rasterio, _, _ = _require_geo()
    threshold = _validate_threshold(threshold)
    if max_distance_km <= 0:
        raise ValueError("max_distance_km must be greater than 0.")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be at least 1.")

    current_path = Path(current_path)
    future_path = Path(future_path)
    accessibility_output_path = Path(accessibility_output_path)
    constrained_future_output_path = Path(constrained_future_output_path)

    current, future, valid, profile, transform, crs = _load_suitability_pair(
        current_path,
        future_path,
    )
    current_suitable = valid & (current >= threshold)
    future_suitable = valid & (future >= threshold)
    if not current_suitable.any():
        raise ValueError("Dispersal constraints require at least one currently suitable cell.")

    current_points = _geographic_points(current_suitable, transform, crs)
    future_points = _geographic_points(future_suitable, transform, crs)
    tree = BallTree(current_points, metric="haversine")

    nearest_km = np.empty(len(future_points), dtype=float)
    for start in range(0, len(future_points), query_chunk_size):
        stop = min(start + query_chunk_size, len(future_points))
        distances, _ = tree.query(future_points[start:stop], k=1)
        nearest_km[start:stop] = distances[:, 0] * EARTH_RADIUS_KM

    future_indices = np.flatnonzero(future_suitable.ravel())
    accessible_flat = np.zeros(valid.size, dtype=bool)
    accessible_flat[future_indices] = nearest_km <= max_distance_km
    accessible = accessible_flat.reshape(valid.shape)
    inaccessible = future_suitable & ~accessible

    classes = np.full(valid.shape, DISPERSAL_NODATA, dtype=np.uint8)
    classes[valid & ~future_suitable] = DISPERSAL_UNSUITABLE
    classes[accessible] = DISPERSAL_ACCESSIBLE
    classes[inaccessible] = DISPERSAL_INACCESSIBLE

    accessibility_output_path.parent.mkdir(parents=True, exist_ok=True)
    class_profile = profile.copy()
    class_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=DISPERSAL_NODATA,
        compress="deflate",
    )
    with rasterio.open(accessibility_output_path, "w", **class_profile) as destination:
        destination.write(classes, 1)
        destination.set_band_description(1, "dispersal_accessibility")
        destination.update_tags(
            rangeshift_output="dispersal_accessibility",
            threshold=str(threshold),
            max_distance_km=str(float(max_distance_km)),
            class_0="future_unsuitable",
            class_1="future_suitable_accessible",
            class_2="future_suitable_beyond_distance_constraint",
        )

    constrained = np.full(valid.shape, -9999.0, dtype=np.float32)
    constrained[valid] = future[valid].astype(np.float32)
    constrained[inaccessible] = 0.0
    constrained_future_output_path.parent.mkdir(parents=True, exist_ok=True)
    constrained_profile = profile.copy()
    constrained_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=-9999.0,
        compress="deflate",
    )
    with rasterio.open(
        constrained_future_output_path,
        "w",
        **constrained_profile,
    ) as destination:
        destination.write(constrained, 1)
        destination.set_band_description(1, "dispersal_constrained_future_suitability")
        destination.update_tags(
            rangeshift_output="dispersal_constrained_future_suitability",
            threshold=str(threshold),
            max_distance_km=str(float(max_distance_km)),
            inaccessible_cells="forced_to_zero_suitability",
        )

    area_km2 = _cell_area_grid_km2(valid.shape, transform, crs)
    accessible_area = float(area_km2[accessible].sum())
    inaccessible_area = float(area_km2[inaccessible].sum())
    suitable_count = int(future_suitable.sum())
    accessible_count = int(accessible.sum())
    inaccessible_count = int(inaccessible.sum())

    return DispersalConstraintResult(
        accessibility_path=accessibility_output_path,
        constrained_future_path=constrained_future_output_path,
        threshold=threshold,
        max_distance_km=float(max_distance_km),
        future_suitable_cells=suitable_count,
        accessible_suitable_cells=accessible_count,
        inaccessible_suitable_cells=inaccessible_count,
        accessible_fraction=(float(accessible_count / suitable_count) if suitable_count else 0.0),
        accessible_suitable_area_km2=accessible_area,
        inaccessible_suitable_area_km2=inaccessible_area,
        crs=str(crs),
    )
