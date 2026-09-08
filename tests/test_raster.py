"""Tests for RangeShift raster suitability prediction."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rangeshift.model import train_habitat_model
from rangeshift.raster import parse_layer_specs, predict_suitability_raster

rasterio = pytest.importorskip("rasterio")
rasterio_transform = pytest.importorskip("rasterio.transform")
from_origin = rasterio_transform.from_origin


def _bundle() -> dict:
    rng = np.random.default_rng(12)
    bio1 = rng.normal(14.0, 3.0, 100)
    bio12 = rng.normal(900.0, 180.0, 100)
    score = -(bio1 - 13.5) ** 2 / 8.0 + (bio12 - 850.0) / 250.0
    presence = (score + rng.normal(0.0, 0.5, 100) > 0).astype(int)
    frame = pd.DataFrame({"presence": presence, "bio1": bio1, "bio12": bio12})

    result = train_habitat_model(
        frame,
        feature_columns=["bio1", "bio12"],
        n_estimators=30,
        random_state=7,
    )
    return {
        "model": result.model,
        "feature_columns": result.feature_columns,
        "target_column": result.target_column,
        "metrics": result.metrics,
    }


def _write_raster(
    path: Path,
    values: np.ndarray,
    *,
    transform=None,
    nodata: float = -9999.0,
) -> None:
    if transform is None:
        transform = from_origin(-85.0, 41.0, 0.25, 0.25)

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
        nodata=nodata,
    ) as dataset:
        dataset.write(values.astype(np.float32), 1)


def test_parse_layer_specs() -> None:
    layers = parse_layer_specs(["bio1=temp.tif", "bio12=rain.tif"])
    assert layers == {"bio1": Path("temp.tif"), "bio12": Path("rain.tif")}

    with pytest.raises(ValueError, match="Duplicate"):
        parse_layer_specs(["bio1=a.tif", "bio1=b.tif"])

    with pytest.raises(ValueError, match="FEATURE=PATH"):
        parse_layer_specs(["broken-spec"])


def test_predict_suitability_raster_preserves_nodata(tmp_path: Path) -> None:
    bio1 = np.array(
        [
            [11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0],
            [12.5, 13.5, 14.5, 15.5],
        ],
        dtype=np.float32,
    )
    bio12 = np.array(
        [
            [1100.0, 1000.0, 900.0, 800.0],
            [750.0, -9999.0, 650.0, 600.0],
            [1050.0, 950.0, 850.0, 700.0],
        ],
        dtype=np.float32,
    )

    bio1_path = tmp_path / "bio1.tif"
    bio12_path = tmp_path / "bio12.tif"
    output_path = tmp_path / "suitability.tif"
    _write_raster(bio1_path, bio1)
    _write_raster(bio12_path, bio12)

    result = predict_suitability_raster(
        _bundle(),
        {"bio1": bio1_path, "bio12": bio12_path},
        output_path,
    )

    assert result.valid_cells == 11
    assert result.total_cells == 12
    assert result.crs == "EPSG:4326"

    with rasterio.open(output_path) as dataset:
        output = dataset.read(1)
        assert dataset.dtypes == ("float32",)
        assert dataset.nodata == -9999.0
        assert dataset.descriptions == ("habitat_suitability",)
        assert output[1, 1] == -9999.0
        valid = output[output != dataset.nodata]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)


def test_raster_prediction_rejects_misaligned_layers(tmp_path: Path) -> None:
    values = np.ones((3, 4), dtype=np.float32)
    bio1_path = tmp_path / "bio1.tif"
    bio12_path = tmp_path / "bio12.tif"
    _write_raster(bio1_path, values)
    _write_raster(
        bio12_path,
        values,
        transform=from_origin(-84.9, 41.0, 0.25, 0.25),
    )

    with pytest.raises(ValueError, match="not aligned"):
        predict_suitability_raster(
            _bundle(),
            {"bio1": bio1_path, "bio12": bio12_path},
            tmp_path / "output.tif",
        )


def test_raster_prediction_requires_exact_model_features(tmp_path: Path) -> None:
    values = np.ones((2, 2), dtype=np.float32)
    bio1_path = tmp_path / "bio1.tif"
    _write_raster(bio1_path, values)

    with pytest.raises(ValueError, match="missing predictors"):
        predict_suitability_raster(
            _bundle(),
            {"bio1": bio1_path},
            tmp_path / "output.tif",
        )
