"""Tests for optional projected geospatial utilities."""

import pandas as pd
import pytest

pyproj = pytest.importorskip("pyproj")

from rangeshift.geospatial import assign_projected_blocks, estimate_local_utm_epsg


def test_estimate_local_utm_epsg_for_ohio_coordinates() -> None:
    frame = pd.DataFrame(
        {
            "latitude": [39.2, 39.5, 39.8],
            "longitude": [-84.9, -84.5, -84.1],
        }
    )

    assert estimate_local_utm_epsg(frame) == 32616


def test_projected_blocks_use_kilometer_grid() -> None:
    frame = pd.DataFrame(
        {
            "latitude": [39.20, 39.22, 40.50, 40.52],
            "longitude": [-84.90, -84.88, -83.10, -83.08],
        }
    )
    result = assign_projected_blocks(frame, block_size_km=50.0)

    assert result.crs_epsg == 32616
    assert result.block_size_km == 50.0
    assert result.blocks.nunique() >= 2
    assert result.blocks.index.equals(frame.index)


def test_utm_estimation_rejects_polar_latitudes() -> None:
    frame = pd.DataFrame(
        {
            "latitude": [85.0, 85.2],
            "longitude": [10.0, 10.2],
        }
    )

    with pytest.raises(ValueError, match="UTM"):
        estimate_local_utm_epsg(frame)
