from pathlib import Path

import numpy as np
import pytest

from rangeshift.scenarios import summarize_suitability_scenarios


def test_summarize_suitability_scenarios_writes_mean_sd_and_agreement(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    transform = from_origin(-85.0, 41.0, 0.25, 0.25)

    def write_raster(path: Path, values: np.ndarray) -> None:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=values.shape[0],
            width=values.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=-9999.0,
        ) as dataset:
            dataset.write(values.astype(np.float32), 1)

    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    write_raster(first, np.array([[0.2, 0.8], [0.6, -9999.0]], dtype=float))
    write_raster(second, np.array([[0.4, 0.6], [0.2, 0.9]], dtype=float))

    result = summarize_suitability_scenarios(
        {"scenario_a": first, "scenario_b": second},
        tmp_path / "summary",
        threshold=0.5,
    )

    assert result.scenario_count == 2
    assert result.threshold == 0.5

    with rasterio.open(result.mean_suitability_path) as dataset:
        mean = dataset.read(1)
        nodata = dataset.nodata
    with rasterio.open(result.suitability_sd_path) as dataset:
        sd = dataset.read(1)
    with rasterio.open(result.suitable_fraction_path) as dataset:
        agreement = dataset.read(1)

    np.testing.assert_allclose(mean[:1, :], np.array([[0.3, 0.7]]), atol=1e-6)
    np.testing.assert_allclose(sd[:1, :], np.array([[0.1, 0.1]]), atol=1e-6)
    np.testing.assert_allclose(agreement[0], np.array([0.0, 1.0]), atol=1e-6)
    assert agreement[1, 0] == pytest.approx(0.5)
    assert mean[1, 1] == pytest.approx(nodata)
    assert sd[1, 1] == pytest.approx(nodata)
    assert agreement[1, 1] == pytest.approx(nodata)


def test_summarize_suitability_scenarios_rejects_single_scenario(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least two scenario rasters"):
        summarize_suitability_scenarios(
            {"only": tmp_path / "only.tif"},
            tmp_path / "summary",
            threshold=0.5,
        )


def test_summarize_suitability_scenarios_rejects_invalid_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        summarize_suitability_scenarios(
            {"a": tmp_path / "a.tif", "b": tmp_path / "b.tif"},
            tmp_path / "summary",
            threshold=1.5,
        )
