"""Tests for current-versus-future range-shift analysis."""

from pathlib import Path

import numpy as np
import pytest

from rangeshift.range_shift import (
    CLASS_NODATA,
    GAINED_SUITABLE,
    LOST_SUITABLE,
    STABLE_SUITABLE,
    STABLE_UNSUITABLE,
    compare_suitability_rasters,
)

rasterio = pytest.importorskip("rasterio")


def _write_suitability(
    path: Path,
    values: np.ndarray,
    *,
    transform,
    crs: str = "EPSG:3857",
    nodata: float = -9999.0,
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(values.astype(np.float32), 1)


def test_range_shift_classification_and_area(tmp_path: Path) -> None:
    transform = rasterio.transform.from_origin(0.0, 2000.0, 1000.0, 1000.0)
    current = np.array(
        [
            [0.2, 0.8, 0.7],
            [0.1, 0.9, 0.4],
        ],
        dtype=np.float32,
    )
    future = np.array(
        [
            [0.3, 0.2, 0.9],
            [0.8, 0.95, 0.1],
        ],
        dtype=np.float32,
    )

    current_path = tmp_path / "current.tif"
    future_path = tmp_path / "future.tif"
    classes_path = tmp_path / "classes.tif"
    difference_path = tmp_path / "difference.tif"
    _write_suitability(current_path, current, transform=transform)
    _write_suitability(future_path, future, transform=transform)

    result = compare_suitability_rasters(
        current_path,
        future_path,
        classes_path,
        threshold=0.5,
        difference_output_path=difference_path,
    )

    expected = np.array(
        [
            [STABLE_UNSUITABLE, LOST_SUITABLE, STABLE_SUITABLE],
            [GAINED_SUITABLE, STABLE_SUITABLE, STABLE_UNSUITABLE],
        ],
        dtype=np.uint8,
    )

    with rasterio.open(classes_path) as dataset:
        observed = dataset.read(1)
        assert np.array_equal(observed, expected)
        assert dataset.nodata == CLASS_NODATA
        assert dataset.dtypes == ("uint8",)

    with rasterio.open(difference_path) as dataset:
        difference = dataset.read(1)
        assert np.allclose(difference, future - current, atol=1e-6)

    assert result.valid_cells == 6
    assert result.total_cells == 6
    assert result.total_valid_area_km2 == pytest.approx(6.0)
    assert result.current_suitable_area_km2 == pytest.approx(3.0)
    assert result.future_suitable_area_km2 == pytest.approx(3.0)
    assert result.stable_suitable_area_km2 == pytest.approx(2.0)
    assert result.gained_area_km2 == pytest.approx(1.0)
    assert result.lost_area_km2 == pytest.approx(1.0)
    assert result.stable_unsuitable_area_km2 == pytest.approx(2.0)
    assert result.net_change_area_km2 == pytest.approx(0.0)
    assert result.percent_change_from_current == pytest.approx(0.0)
    assert result.jaccard_overlap == pytest.approx(0.5)
    assert result.centroid_shift_km is not None
    assert result.centroid_shift_km > 0.0
    assert result.centroid_bearing_degrees is not None
    assert 0.0 <= result.centroid_bearing_degrees < 360.0


def test_range_shift_propagates_shared_nodata(tmp_path: Path) -> None:
    transform = rasterio.transform.from_origin(-85.0, 41.0, 0.1, 0.1)
    current = np.array([[0.8, -9999.0], [0.2, 0.7]], dtype=np.float32)
    future = np.array([[0.9, 0.6], [0.1, 0.4]], dtype=np.float32)

    current_path = tmp_path / "current.tif"
    future_path = tmp_path / "future.tif"
    classes_path = tmp_path / "classes.tif"
    _write_suitability(current_path, current, transform=transform, crs="EPSG:4326")
    _write_suitability(future_path, future, transform=transform, crs="EPSG:4326")

    result = compare_suitability_rasters(
        current_path,
        future_path,
        classes_path,
        threshold=0.5,
    )

    with rasterio.open(classes_path) as dataset:
        classes = dataset.read(1)
        assert classes[0, 1] == CLASS_NODATA

    assert result.valid_cells == 3
    assert result.total_valid_area_km2 > 0.0


def test_range_shift_requires_explicit_valid_threshold(tmp_path: Path) -> None:
    transform = rasterio.transform.from_origin(0.0, 1000.0, 1000.0, 1000.0)
    values = np.array([[0.4, 0.8]], dtype=np.float32)
    current_path = tmp_path / "current.tif"
    future_path = tmp_path / "future.tif"
    _write_suitability(current_path, values, transform=transform)
    _write_suitability(future_path, values, transform=transform)

    for threshold in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="threshold"):
            compare_suitability_rasters(
                current_path,
                future_path,
                tmp_path / "classes.tif",
                threshold=threshold,
            )


def test_range_shift_rejects_misaligned_rasters(tmp_path: Path) -> None:
    current_path = tmp_path / "current.tif"
    future_path = tmp_path / "future.tif"
    values = np.array([[0.2, 0.8]], dtype=np.float32)

    _write_suitability(
        current_path,
        values,
        transform=rasterio.transform.from_origin(0.0, 1000.0, 1000.0, 1000.0),
    )
    _write_suitability(
        future_path,
        values,
        transform=rasterio.transform.from_origin(100.0, 1000.0, 1000.0, 1000.0),
    )

    with pytest.raises(ValueError, match="same pixel grid"):
        compare_suitability_rasters(
            current_path,
            future_path,
            tmp_path / "classes.tif",
            threshold=0.5,
        )
