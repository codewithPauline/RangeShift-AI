"""Pseudo-absence and background-point generation for RangeShift AI."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

from .spatial import validate_coordinates

EARTH_RADIUS_KM = 6371.0088
SUPPORTED_BACKGROUND_METHODS = {"random", "spatial_stratified", "candidate_pool"}


@dataclass
class BackgroundGenerationResult:
    """Generated background coordinates plus provenance metadata."""

    points: pd.DataFrame
    method: str
    seed: int
    requested_points: int
    min_distance_km: float
    bounds: tuple[float, float, float, float]


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1_r = radians(lat1)
    lon1_r = radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _validate_bounds(bounds: tuple[float, float, float, float]) -> None:
    min_lon, min_lat, max_lon, max_lat = bounds
    if not (-180.0 <= min_lon < max_lon <= 180.0):
        raise ValueError("Longitude bounds must satisfy -180 <= min < max <= 180.")
    if not (-90.0 <= min_lat < max_lat <= 90.0):
        raise ValueError("Latitude bounds must satisfy -90 <= min < max <= 90.")


def _default_bounds(
    presences: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    buffer_degrees: float,
) -> tuple[float, float, float, float]:
    if buffer_degrees < 0:
        raise ValueError("buffer_degrees cannot be negative.")
    min_lon = max(-180.0, float(presences[longitude_column].min()) - buffer_degrees)
    max_lon = min(180.0, float(presences[longitude_column].max()) + buffer_degrees)
    min_lat = max(-90.0, float(presences[latitude_column].min()) - buffer_degrees)
    max_lat = min(90.0, float(presences[latitude_column].max()) + buffer_degrees)
    bounds = (min_lon, min_lat, max_lon, max_lat)
    _validate_bounds(bounds)
    return bounds


def _far_enough(
    latitude: float,
    longitude: float,
    presence_latitudes: np.ndarray,
    presence_longitudes: np.ndarray,
    min_distance_km: float,
) -> bool:
    if min_distance_km <= 0:
        return True
    distances = _haversine_km(
        latitude,
        longitude,
        presence_latitudes,
        presence_longitudes,
    )
    return bool(np.min(distances) >= min_distance_km)


def _random_equal_area_coordinate(
    rng: np.random.Generator,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bounds
    longitude = float(rng.uniform(min_lon, max_lon))
    sin_min = sin(radians(min_lat))
    sin_max = sin(radians(max_lat))
    latitude = float(np.degrees(asin(float(rng.uniform(sin_min, sin_max)))))
    return latitude, longitude


def _random_background(
    *,
    rng: np.random.Generator,
    n_points: int,
    bounds: tuple[float, float, float, float],
    presence_latitudes: np.ndarray,
    presence_longitudes: np.ndarray,
    min_distance_km: float,
    max_attempts: int,
) -> list[tuple[float, float]]:
    selected: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    attempts = 0
    while len(selected) < n_points and attempts < max_attempts:
        attempts += 1
        latitude, longitude = _random_equal_area_coordinate(rng, bounds)
        key = (round(latitude, 8), round(longitude, 8))
        if key in seen:
            continue
        if not _far_enough(
            latitude,
            longitude,
            presence_latitudes,
            presence_longitudes,
            min_distance_km,
        ):
            continue
        selected.append((latitude, longitude))
        seen.add(key)
    if len(selected) < n_points:
        raise RuntimeError(
            f"Only {len(selected)} of {n_points} background points could be generated. "
            "Reduce min_distance_km, enlarge the bounds, or increase max_attempts."
        )
    return selected


def _stratified_background(
    *,
    rng: np.random.Generator,
    n_points: int,
    bounds: tuple[float, float, float, float],
    presence_latitudes: np.ndarray,
    presence_longitudes: np.ndarray,
    min_distance_km: float,
    strata_size_degrees: float,
    max_attempts: int,
) -> list[tuple[float, float]]:
    if strata_size_degrees <= 0:
        raise ValueError("strata_size_degrees must be greater than 0.")
    min_lon, min_lat, max_lon, max_lat = bounds
    lat_edges = np.arange(min_lat, max_lat, strata_size_degrees)
    lon_edges = np.arange(min_lon, max_lon, strata_size_degrees)
    cells = [
        (
            float(lon),
            float(lat),
            min(float(lon + strata_size_degrees), max_lon),
            min(float(lat + strata_size_degrees), max_lat),
        )
        for lat in lat_edges
        for lon in lon_edges
    ]
    if not cells:
        raise ValueError("No spatial strata were created for the requested bounds.")

    order = rng.permutation(len(cells)).tolist()
    selected: list[tuple[float, float]] = []
    attempts = 0
    while len(selected) < n_points and attempts < max_attempts:
        if not order:
            order = rng.permutation(len(cells)).tolist()
        cell = cells[order.pop()]
        attempts += 1
        latitude, longitude = _random_equal_area_coordinate(rng, cell)
        if not _far_enough(
            latitude,
            longitude,
            presence_latitudes,
            presence_longitudes,
            min_distance_km,
        ):
            continue
        selected.append((latitude, longitude))
    if len(selected) < n_points:
        raise RuntimeError(
            f"Only {len(selected)} of {n_points} stratified background points could be generated."
        )
    return selected


def _candidate_pool_background(
    candidate_pool: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_points: int,
    bounds: tuple[float, float, float, float],
    latitude_column: str,
    longitude_column: str,
    presence_latitudes: np.ndarray,
    presence_longitudes: np.ndarray,
    min_distance_km: float,
) -> list[tuple[float, float]]:
    validate_coordinates(candidate_pool, latitude_column, longitude_column)
    min_lon, min_lat, max_lon, max_lat = bounds
    pool = candidate_pool.loc[
        candidate_pool[longitude_column].between(min_lon, max_lon)
        & candidate_pool[latitude_column].between(min_lat, max_lat)
    ].copy()
    if pool.empty:
        raise ValueError("candidate_pool contains no coordinates inside the requested bounds.")

    keep = []
    for index, row in pool.iterrows():
        if _far_enough(
            float(row[latitude_column]),
            float(row[longitude_column]),
            presence_latitudes,
            presence_longitudes,
            min_distance_km,
        ):
            keep.append(index)
    pool = pool.loc[keep].drop_duplicates([latitude_column, longitude_column])
    if len(pool) < n_points:
        raise ValueError(
            f"candidate_pool contains only {len(pool)} eligible unique points; "
            f"{n_points} were requested."
        )
    chosen = rng.choice(pool.index.to_numpy(), size=n_points, replace=False)
    return [
        (float(pool.loc[index, latitude_column]), float(pool.loc[index, longitude_column]))
        for index in chosen
    ]


def generate_background_points(
    presences: pd.DataFrame,
    n_points: int,
    *,
    method: str = "random",
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    min_distance_km: float = 0.0,
    buffer_degrees: float = 1.0,
    bounds: tuple[float, float, float, float] | None = None,
    strata_size_degrees: float = 1.0,
    candidate_pool: pd.DataFrame | None = None,
    seed: int = 42,
    max_attempts: int | None = None,
) -> BackgroundGenerationResult:
    """Generate reproducible pseudo-absence/background coordinates.

    Methods
    -------
    random
        Equal-area random sampling inside the geographic bounds.
    spatial_stratified
        Samples across geographic grid strata to reduce extreme spatial clumping.
    candidate_pool
        Samples from user-supplied effort/target-group coordinates.
    """
    validate_coordinates(presences, latitude_column, longitude_column)
    if presences.empty:
        raise ValueError("At least one presence coordinate is required.")
    if n_points < 1:
        raise ValueError("n_points must be at least 1.")
    if method not in SUPPORTED_BACKGROUND_METHODS:
        supported = ", ".join(sorted(SUPPORTED_BACKGROUND_METHODS))
        raise ValueError(f"Unsupported method '{method}'. Choose from: {supported}.")
    if min_distance_km < 0:
        raise ValueError("min_distance_km cannot be negative.")

    resolved_bounds = bounds or _default_bounds(
        presences,
        latitude_column,
        longitude_column,
        buffer_degrees,
    )
    _validate_bounds(resolved_bounds)
    attempts = max_attempts or max(10000, n_points * 200)
    if attempts < n_points:
        raise ValueError("max_attempts must be at least n_points.")

    rng = np.random.default_rng(seed)
    presence_latitudes = presences[latitude_column].to_numpy(dtype=float)
    presence_longitudes = presences[longitude_column].to_numpy(dtype=float)

    if method == "random":
        selected = _random_background(
            rng=rng,
            n_points=n_points,
            bounds=resolved_bounds,
            presence_latitudes=presence_latitudes,
            presence_longitudes=presence_longitudes,
            min_distance_km=min_distance_km,
            max_attempts=attempts,
        )
    elif method == "spatial_stratified":
        selected = _stratified_background(
            rng=rng,
            n_points=n_points,
            bounds=resolved_bounds,
            presence_latitudes=presence_latitudes,
            presence_longitudes=presence_longitudes,
            min_distance_km=min_distance_km,
            strata_size_degrees=strata_size_degrees,
            max_attempts=attempts,
        )
    else:
        if candidate_pool is None:
            raise ValueError("candidate_pool must be provided when method='candidate_pool'.")
        selected = _candidate_pool_background(
            candidate_pool,
            rng=rng,
            n_points=n_points,
            bounds=resolved_bounds,
            latitude_column=latitude_column,
            longitude_column=longitude_column,
            presence_latitudes=presence_latitudes,
            presence_longitudes=presence_longitudes,
            min_distance_km=min_distance_km,
        )

    points = pd.DataFrame(selected, columns=[latitude_column, longitude_column])
    points["presence"] = 0
    points["background_method"] = method
    return BackgroundGenerationResult(
        points=points,
        method=method,
        seed=seed,
        requested_points=n_points,
        min_distance_km=float(min_distance_km),
        bounds=tuple(float(value) for value in resolved_bounds),
    )
