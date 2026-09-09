"""Tests for RangeShift raster map rendering."""

from pathlib import Path

import numpy as np
import pytest

from rangeshift.visualization import plot_range_shift_map, plot_suitability_map

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
rasterio = pytest.importorskip("rasterio")
rasterio_transform = pytest.importorskip("rasterio.transform")
from_origin = rasterio_transform.from_origin


def _write_suitability(path: Path) -> None:
    values = np.array(
        [
            [0.05, 0.20, 0.45, 0.70],
            [0.10, 0.35, 0.65, 0.90],
            [0.15, 0.40, -9999.0, 0.95],
        ],
        dtype=np.float32,
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-85.0, 41.0, 0.25, 0.25),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(values, 1)


def _write_range_shift(path: Path) -> None:
    values = np.array(
        [
            [0, 0, 1, 3],
            [0, 2, 3, 3],
            [255, 2, 2, 3],
        ],
        dtype=np.uint8,
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(-85.0, 41.0, 0.25, 0.25),
        nodata=255,
    ) as dataset:
        dataset.write(values, 1)


def test_plot_suitability_map_creates_png(tmp_path: Path) -> None:
    raster_path = tmp_path / "suitability.tif"
    output_path = tmp_path / "suitability.png"
    _write_suitability(raster_path)

    result = plot_suitability_map(
        raster_path,
        output_path,
        title="Synthetic suitability",
        dpi=120,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_range_shift_map_creates_png(tmp_path: Path) -> None:
    raster_path = tmp_path / "range_shift.tif"
    output_path = tmp_path / "range_shift.png"
    _write_range_shift(raster_path)

    result = plot_range_shift_map(
        raster_path,
        output_path,
        title="Synthetic range shift",
        dpi=120,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_suitability_map_rejects_low_dpi(tmp_path: Path) -> None:
    raster_path = tmp_path / "suitability.tif"
    _write_suitability(raster_path)

    with pytest.raises(ValueError, match="dpi"):
        plot_suitability_map(raster_path, tmp_path / "map.png", dpi=50)


def test_plot_range_shift_map_rejects_unknown_classes(tmp_path: Path) -> None:
    raster_path = tmp_path / "bad_classes.tif"
    values = np.array([[0, 4]], dtype=np.uint8)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=1,
        width=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(-85.0, 41.0, 0.25, 0.25),
        nodata=255,
    ) as dataset:
        dataset.write(values, 1)

    with pytest.raises(ValueError, match="Unexpected"):
        plot_range_shift_map(raster_path, tmp_path / "bad.png", dpi=120)
