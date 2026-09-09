"""Sampling-bias diagnostics for georeferenced occurrence/background data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .spatial import assign_spatial_blocks, validate_coordinates

EARTH_RADIUS_KM = 6371.0088


@dataclass
class SamplingBiasResult:
    """Summary statistics describing geographic clustering in sampled records."""

    metrics: dict[str, float]
    block_counts: pd.DataFrame
    nearest_neighbor_km: pd.Series
    block_size_degrees: float


def diagnose_sampling_bias(
    frame: pd.DataFrame,
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    block_size_degrees: float = 1.0,
) -> SamplingBiasResult:
    """Quantify geographic clustering without imposing an arbitrary pass/fail rule.

    Reported diagnostics include duplicate-coordinate fraction, occupied spatial
    blocks, concentration of records among blocks, an effective-number-of-blocks
    diversity measure, and nearest-neighbor distances on the sphere.
    """
    validate_coordinates(frame, latitude_column, longitude_column)
    if len(frame) < 2:
        raise ValueError("Sampling-bias diagnostics require at least two observations.")

    blocks = assign_spatial_blocks(
        frame,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        block_size_degrees=block_size_degrees,
    )
    block_counts = (
        blocks.value_counts()
        .rename_axis("spatial_block")
        .reset_index(name="observations")
        .sort_values("observations", ascending=False, ignore_index=True)
    )

    counts = block_counts["observations"].to_numpy(dtype=float)
    proportions = counts / counts.sum()
    occupied_blocks = len(block_counts)
    effective_blocks = float(1.0 / np.square(proportions).sum())
    block_count_cv = float(counts.std(ddof=0) / counts.mean()) if counts.mean() else 0.0

    coordinates = frame[[latitude_column, longitude_column]].to_numpy(dtype=float)
    duplicate_fraction = float(
        1.0 - len(np.unique(coordinates, axis=0)) / len(coordinates)
    )

    radians = np.radians(coordinates)
    tree = BallTree(radians, metric="haversine")
    distances, _ = tree.query(radians, k=2)
    nearest_neighbor_km = pd.Series(
        distances[:, 1] * EARTH_RADIUS_KM,
        index=frame.index,
        name="nearest_neighbor_km",
    )

    metrics = {
        "observations": float(len(frame)),
        "occupied_blocks": float(occupied_blocks),
        "max_block_fraction": float(proportions.max()),
        "block_count_cv": block_count_cv,
        "effective_blocks": effective_blocks,
        "effective_block_fraction": float(effective_blocks / occupied_blocks),
        "duplicate_coordinate_fraction": duplicate_fraction,
        "median_nearest_neighbor_km": float(nearest_neighbor_km.median()),
        "p10_nearest_neighbor_km": float(nearest_neighbor_km.quantile(0.10)),
        "p90_nearest_neighbor_km": float(nearest_neighbor_km.quantile(0.90)),
    }

    block_counts["fraction"] = block_counts["observations"] / len(frame)
    return SamplingBiasResult(
        metrics=metrics,
        block_counts=block_counts,
        nearest_neighbor_km=nearest_neighbor_km,
        block_size_degrees=block_size_degrees,
    )
