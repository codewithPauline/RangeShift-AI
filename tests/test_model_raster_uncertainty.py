from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rangeshift.model_raster_uncertainty import summarize_model_fit_rasters


class FakeModel:
    classes_ = np.array([0, 1])

    def __init__(self, offset):
        self.offset = offset

    def predict_proba(self, frame):
        p = np.clip(frame["bio1"].to_numpy() + self.offset, 0, 1)
        return np.column_stack([1 - p, p])


def test_windowed_model_fit_raster_uncertainty(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    tif = tmp_path / "bio1.tif"
    with rasterio.open(
        tif, "w", driver="GTiff", height=2, width=2, count=1,
        dtype="float32", nodata=-9999.0,
        crs="EPSG:4326", transform=from_origin(-85, 40, 0.1, 0.1),
    ) as ds:
        ds.write(np.array([[0.2, 0.6], [0.4, -9999.0]], dtype=np.float32), 1)
    bundles = [
        {"model": FakeModel(0.0), "feature_columns": ["bio1"]},
        {"model": FakeModel(0.2), "feature_columns": ["bio1"]},
    ]
    result = summarize_model_fit_rasters(
        bundles, {"bio1": tif}, tmp_path / "result", window_size=1, threshold=0.5
    )
    assert result.valid_cells == 3
    with rasterio.open(result.mean_path) as ds:
        values = ds.read(1)
        assert values[0, 0] == pytest.approx(0.3)
        assert values[1, 1] == ds.nodata
    with rasterio.open(result.sd_path) as ds:
        assert ds.read(1)[0, 0] == pytest.approx(np.sqrt(0.02))
    with rasterio.open(result.agreement_path) as ds:
        assert ds.read(1)[1, 0] == pytest.approx(0.5)
    with rasterio.open(result.q025_path) as ds:
        assert 0.2 < ds.read(1)[0, 0] < 0.3


def test_model_fit_raster_rejects_bad_ensembles(tmp_path: Path):
    bundle = {"model": FakeModel(0), "feature_columns": ["bio1"]}
    with pytest.raises(ValueError, match="At least two"):
        summarize_model_fit_rasters([bundle], {}, tmp_path)
    with pytest.raises(ValueError, match="ordered features"):
        summarize_model_fit_rasters(
            [bundle, {"model": FakeModel(0), "feature_columns": ["bio12"]}],
            {}, tmp_path,
        )
