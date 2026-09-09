"""Current-versus-future range-shift analysis for RangeShift AI.

This module converts two aligned continuous suitability rasters into explicit
stable/lost/gained habitat classes using a user-supplied threshold. It also
summarizes physically meaningful area and geographic centroid movement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


STABLE_UNSUITABLE = 0
LOST_SUITABLE = 1
GAINED_SUITABLE = 2
STABLE_SUITABLE = 3
CLASS_NODATA = 255


@dataclass
class RangeShiftResult:
    """Summary statistics and outputs from a current/future comparison."""

    classes_path: Path
    difference_path: Path | None
    threshold: float
    valid_cells: int
    total_cells: int
    total_valid_area_km2: float
    current_suitable_area_km2: float
    future_suitable_area_km2: float
    stable_suitable_area_km2: float
    gained_area_km2: float
    lost_area_km2: float
    stable_unsuitable_area_km2: float
    net_change_area_km2: float
    percent_change_from_current: float | None
    jaccard_overlap: float | None
    current_centroid_longitude: float | None
    current_centroid_latitude: float | None
    future_centroid_longitude: float | None
    future_centroid_latitude: float | None
    centroid_shift_km: float | None
    centroid_bearing_degrees: float | None
    crs: str

    def to_dict(self) -> dict:
        """Return a JSON-serializable result dictionary."""
        payload = self.__dict__.copy()
        payload["classes_path"] = str(self.classes_path)
        payload["difference_path"] = (
            str(self.difference_path) if self.difference_path is not None else None
        )
        return payload


def _require_geo():
    """Import raster/projection dependencies with an actionable error."""
    try:
        import rasterio
        from pyproj import CRS, Transformer
    except ImportError as exc:
        raise ImportError(
            "Range-shift raster analysis requires Rasterio and PyProj. "
            "Install RangeShift with the 'geo' optional dependency."
        ) from exc
    return rasterio, CRS, Transformer


def _validate_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be explicitly set between 0 and 1.")
    return threshold


def _load_suitability_pair(current_path: Path, future_path: Path):
    rasterio, _, _ = _require_geo()

    for label, path in (("current", current_path), ("future", future_path)):
        if not path.exists():
            raise FileNotFoundError(
                f"{label.capitalize()} suitability raster does not exist: {path}"
            )

    with rasterio.open(current_path) as current_ds, rasterio.open(future_path) as future_ds:
        for label, dataset in (("current", current_ds), ("future", future_ds)):
            if dataset.count != 1:
                raise ValueError(f"{label.capitalize()} suitability raster must be single-band.")
            if dataset.crs is None:
                raise ValueError(f"{label.capitalize()} suitability raster must define a CRS.")

        if (current_ds.height, current_ds.width) != (future_ds.height, future_ds.width):
            raise ValueError(
                "Current and future suitability rasters must have identical dimensions."
            )
        if current_ds.transform != future_ds.transform:
            raise ValueError(
                "Current and future suitability rasters are not on the same pixel grid."
            )
        if current_ds.crs != future_ds.crs:
            raise ValueError("Current and future suitability rasters must use the same CRS.")

        current = current_ds.read(1, masked=True)
        future = future_ds.read(1, masked=True)
        current_values = np.asarray(current.filled(np.nan), dtype=float)
        future_values = np.asarray(future.filled(np.nan), dtype=float)
        valid = (
            ~np.ma.getmaskarray(current)
            & ~np.ma.getmaskarray(future)
            & np.isfinite(current_values)
            & np.isfinite(future_values)
        )

        if not valid.any():
            raise ValueError("Current and future rasters have no shared valid cells.")

        current_valid = current_values[valid]
        future_valid = future_values[valid]
        if np.any((current_valid < 0.0) | (current_valid > 1.0)):
            raise ValueError("Current suitability values must fall between 0 and 1.")
        if np.any((future_valid < 0.0) | (future_valid > 1.0)):
            raise ValueError("Future suitability values must fall between 0 and 1.")

        profile = current_ds.profile.copy()
        transform = current_ds.transform
        crs = current_ds.crs

    return current_values, future_values, valid, profile, transform, crs


def _cell_area_grid_km2(shape: tuple[int, int], transform, crs) -> np.ndarray:
    """Calculate cell areas in km² for projected or geographic rasters."""
    _, CRS, _ = _require_geo()
    pyproj_crs = CRS.from_user_input(crs)
    height, width = shape

    if pyproj_crs.is_projected:
        axis_info = pyproj_crs.axis_info
        if not axis_info or axis_info[0].unit_conversion_factor is None:
            raise ValueError("Projected CRS does not expose a linear unit conversion factor.")
        metres_per_unit = float(axis_info[0].unit_conversion_factor)
        pixel_area_units2 = abs(transform.a * transform.e - transform.b * transform.d)
        pixel_area_km2 = pixel_area_units2 * metres_per_unit**2 / 1_000_000.0
        return np.full(shape, pixel_area_km2, dtype=float)

    if not pyproj_crs.is_geographic:
        raise ValueError("Raster CRS must be geographic or projected for area calculation.")

    geod = pyproj_crs.get_geod()
    areas = np.empty(shape, dtype=float)

    for row in range(height):
        for col in range(width):
            corners = [
                transform * (col, row),
                transform * (col + 1, row),
                transform * (col + 1, row + 1),
                transform * (col, row + 1),
            ]
            lons = [point[0] for point in corners]
            lats = [point[1] for point in corners]
            area_m2, _ = geod.polygon_area_perimeter(lons, lats)
            areas[row, col] = abs(area_m2) / 1_000_000.0

    return areas


def _cell_centres(shape: tuple[int, int], transform) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices(shape, dtype=float)
    cols += 0.5
    rows += 0.5
    x = transform.a * cols + transform.b * rows + transform.c
    y = transform.d * cols + transform.e * rows + transform.f
    return x, y


def _weighted_geographic_centroid(
    mask: np.ndarray,
    area_km2: np.ndarray,
    transform,
    crs,
) -> tuple[float, float] | None:
    """Calculate an area-weighted centroid on the sphere, returned as lon/lat."""
    if not mask.any():
        return None

    _, _, Transformer = _require_geo()
    x, y = _cell_centres(mask.shape, transform)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x[mask], y[mask])

    weights = area_km2[mask]
    lon_rad = np.deg2rad(np.asarray(lon, dtype=float))
    lat_rad = np.deg2rad(np.asarray(lat, dtype=float))

    x_cart = np.cos(lat_rad) * np.cos(lon_rad)
    y_cart = np.cos(lat_rad) * np.sin(lon_rad)
    z_cart = np.sin(lat_rad)

    x_mean = np.average(x_cart, weights=weights)
    y_mean = np.average(y_cart, weights=weights)
    z_mean = np.average(z_cart, weights=weights)

    lon_centroid = np.rad2deg(np.arctan2(y_mean, x_mean))
    hyp = np.hypot(x_mean, y_mean)
    lat_centroid = np.rad2deg(np.arctan2(z_mean, hyp))
    return float(lon_centroid), float(lat_centroid)


def _centroid_shift(
    current_centroid: tuple[float, float] | None,
    future_centroid: tuple[float, float] | None,
) -> tuple[float | None, float | None]:
    if current_centroid is None or future_centroid is None:
        return None, None

    _, CRS, _ = _require_geo()
    geod = CRS.from_epsg(4326).get_geod()
    azimuth, _, distance_m = geod.inv(
        current_centroid[0],
        current_centroid[1],
        future_centroid[0],
        future_centroid[1],
    )
    return float(distance_m / 1000.0), float(azimuth % 360.0)


def compare_suitability_rasters(
    current_path: str | Path,
    future_path: str | Path,
    classes_output_path: str | Path,
    *,
    threshold: float,
    difference_output_path: str | Path | None = None,
) -> RangeShiftResult:
    """Compare aligned current and future suitability rasters.

    Class codes written to the output GeoTIFF are:

    - 0: stable unsuitable
    - 1: lost suitable habitat
    - 2: gained suitable habitat
    - 3: stable suitable habitat
    - 255: nodata

    The threshold is intentionally required rather than silently defaulted.
    """
    rasterio, _, _ = _require_geo()
    threshold = _validate_threshold(threshold)
    current_path = Path(current_path)
    future_path = Path(future_path)
    classes_output_path = Path(classes_output_path)
    difference_path = (
        Path(difference_output_path) if difference_output_path is not None else None
    )

    current, future, valid, profile, transform, crs = _load_suitability_pair(
        current_path,
        future_path,
    )

    current_suitable = valid & (current >= threshold)
    future_suitable = valid & (future >= threshold)

    stable_unsuitable = valid & ~current_suitable & ~future_suitable
    lost = valid & current_suitable & ~future_suitable
    gained = valid & ~current_suitable & future_suitable
    stable_suitable = valid & current_suitable & future_suitable

    classes = np.full(valid.shape, CLASS_NODATA, dtype=np.uint8)
    classes[stable_unsuitable] = STABLE_UNSUITABLE
    classes[lost] = LOST_SUITABLE
    classes[gained] = GAINED_SUITABLE
    classes[stable_suitable] = STABLE_SUITABLE

    classes_output_path.parent.mkdir(parents=True, exist_ok=True)
    class_profile = profile.copy()
    class_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=CLASS_NODATA,
        compress="deflate",
    )
    with rasterio.open(classes_output_path, "w", **class_profile) as destination:
        destination.write(classes, 1)
        destination.set_band_description(1, "range_shift_class")
        destination.update_tags(
            rangeshift_output="range_shift_classification",
            threshold=str(threshold),
            class_0="stable_unsuitable",
            class_1="lost_suitable",
            class_2="gained_suitable",
            class_3="stable_suitable",
        )

    if difference_path is not None:
        difference = np.full(valid.shape, -9999.0, dtype=np.float32)
        difference[valid] = (future[valid] - current[valid]).astype(np.float32)
        difference_path.parent.mkdir(parents=True, exist_ok=True)
        difference_profile = profile.copy()
        difference_profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=-9999.0,
            compress="deflate",
        )
        with rasterio.open(difference_path, "w", **difference_profile) as destination:
            destination.write(difference, 1)
            destination.set_band_description(1, "future_minus_current_suitability")
            destination.update_tags(
                rangeshift_output="suitability_difference",
                difference_definition="future_minus_current",
            )

    area_km2 = _cell_area_grid_km2(valid.shape, transform, crs)

    def area(mask: np.ndarray) -> float:
        return float(area_km2[mask].sum())

    current_area = area(current_suitable)
    future_area = area(future_suitable)
    stable_area = area(stable_suitable)
    gained_area = area(gained)
    lost_area = area(lost)
    stable_unsuitable_area = area(stable_unsuitable)
    total_valid_area = area(valid)
    net_change = future_area - current_area
    percent_change = (net_change / current_area * 100.0) if current_area > 0 else None

    union_area = stable_area + gained_area + lost_area
    jaccard = stable_area / union_area if union_area > 0 else None

    current_centroid = _weighted_geographic_centroid(
        current_suitable,
        area_km2,
        transform,
        crs,
    )
    future_centroid = _weighted_geographic_centroid(
        future_suitable,
        area_km2,
        transform,
        crs,
    )
    shift_km, bearing = _centroid_shift(current_centroid, future_centroid)

    return RangeShiftResult(
        classes_path=classes_output_path,
        difference_path=difference_path,
        threshold=threshold,
        valid_cells=int(valid.sum()),
        total_cells=int(valid.size),
        total_valid_area_km2=total_valid_area,
        current_suitable_area_km2=current_area,
        future_suitable_area_km2=future_area,
        stable_suitable_area_km2=stable_area,
        gained_area_km2=gained_area,
        lost_area_km2=lost_area,
        stable_unsuitable_area_km2=stable_unsuitable_area,
        net_change_area_km2=net_change,
        percent_change_from_current=percent_change,
        jaccard_overlap=jaccard,
        current_centroid_longitude=(current_centroid[0] if current_centroid else None),
        current_centroid_latitude=(current_centroid[1] if current_centroid else None),
        future_centroid_longitude=(future_centroid[0] if future_centroid else None),
        future_centroid_latitude=(future_centroid[1] if future_centroid else None),
        centroid_shift_km=shift_km,
        centroid_bearing_degrees=bearing,
        crs=str(crs),
    )
