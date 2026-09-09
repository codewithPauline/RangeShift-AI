from pathlib import Path

import numpy as np
import pytest

from rangeshift.dispersal import apply_dispersal_constraint

rasterio = pytest.importorskip("rasterio")
rasterio_transform = pytest.importorskip("rasterio.transform")
from_origin = rasterio_transform.from_origin


def _write(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-85.0, 40.0, 0.1, 0.1),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(values.astype(np.float32), 1)


def test_dispersal_constraint_separates_reachable_and_unreachable_future_cells(
    tmp_path: Path,
) -> None:
    current = np.array([[0.9, 0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
    future = np.array([[0.9, 0.8, 0.1, 0.1, 0.9]], dtype=np.float32)
    current_path = tmp_path / "current.tif"
    future_path = tmp_path / "future.tif"
    accessibility = tmp_path / "accessibility.tif"
    constrained = tmp_path / "constrained.tif"
    _write(current_path, current)
    _write(future_path, future)

    result = apply_dispersal_constraint(
        current_path,
        future_path,
        accessibility,
        constrained,
        threshold=0.5,
        max_distance_km=20.0,
    )

    assert result.future_suitable_cells == 3
    assert result.accessible_suitable_cells == 2
    assert result.inaccessible_suitable_cells == 1
    assert result.accessible_fraction == pytest.approx(2 / 3)

    with rasterio.open(accessibility) as dataset:
        classes = dataset.read(1)
        assert classes.tolist() == [[1, 1, 0, 0, 2]]

    with rasterio.open(constrained) as dataset:
        values = dataset.read(1)
        assert values[0, 0] == pytest.approx(0.9)
        assert values[0, 1] == pytest.approx(0.8)
        assert values[0, 4] == pytest.approx(0.0)
