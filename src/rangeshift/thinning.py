"""Spatial thinning utilities for clustered occurrence records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spatial import validate_coordinates

EARTH_RADIUS_KM = 6371.0088


@dataclass
class SpatialThinningResult:
    """Result of deterministic minimum-distance spatial thinning."""

    thinned_frame: pd.DataFrame
    kept_indices: list[object]
    removed_indices: list[object]
    original_count: int
    retained_count: int
    retention_fraction: float
    min_distance_km: float
    seed: int
    priority_column: str | None


def _haversine_to_many(
    latitude: float,
    longitude: float,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    lat1 = np.radians(latitude)
    lon1 = np.radians(longitude)
    lat2 = np.radians(latitudes)
    lon2 = np.radians(longitudes)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def thin_spatial_points(
    frame: pd.DataFrame,
    min_distance_km: float,
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    priority_column: str | None = None,
    seed: int = 42,
) -> SpatialThinningResult:
    """Greedily retain records separated by at least ``min_distance_km``.

    Without a priority column, candidate order is reproducibly randomized to avoid
    privileging the original row order. With a priority column, larger values are
    considered first and random jitter breaks ties reproducibly.
    """
    validate_coordinates(frame, latitude_column, longitude_column)
    if frame.empty:
        raise ValueError("Spatial thinning requires at least one record.")
    if min_distance_km <= 0:
        raise ValueError("min_distance_km must be greater than 0.")
    if priority_column is not None:
        if priority_column not in frame.columns:
            raise ValueError(f"Missing priority column: {priority_column}")
        if not pd.api.types.is_numeric_dtype(frame[priority_column]):
            raise ValueError("priority_column must be numeric.")
        if frame[priority_column].isna().any():
            raise ValueError("priority_column cannot contain missing values.")

    rng = np.random.default_rng(seed)
    working = frame.copy()
    working["__rangeshift_tie"] = rng.random(len(working))
    if priority_column is None:
        working = working.sort_values("__rangeshift_tie", kind="stable")
    else:
        working = working.sort_values(
            [priority_column, "__rangeshift_tie"],
            ascending=[False, True],
            kind="stable",
        )

    kept_indices: list[object] = []
    kept_latitudes: list[float] = []
    kept_longitudes: list[float] = []

    for index, row in working.iterrows():
        latitude = float(row[latitude_column])
        longitude = float(row[longitude_column])
        if not kept_indices:
            keep = True
        else:
            distances = _haversine_to_many(
                latitude,
                longitude,
                np.asarray(kept_latitudes, dtype=float),
                np.asarray(kept_longitudes, dtype=float),
            )
            keep = bool(np.all(distances >= min_distance_km))
        if keep:
            kept_indices.append(index)
            kept_latitudes.append(latitude)
            kept_longitudes.append(longitude)

    kept_set = set(kept_indices)
    removed_indices = [index for index in frame.index if index not in kept_set]
    thinned = frame.loc[kept_indices].copy()
    retained_count = len(thinned)
    original_count = len(frame)

    return SpatialThinningResult(
        thinned_frame=thinned,
        kept_indices=kept_indices,
        removed_indices=removed_indices,
        original_count=original_count,
        retained_count=retained_count,
        retention_fraction=float(retained_count / original_count),
        min_distance_km=float(min_distance_km),
        seed=seed,
        priority_column=priority_column,
    )
