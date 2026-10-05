import json

import numpy as np
import pandas as pd
import pytest

from rangeshift.bootstrap_raster import run_bootstrap_raster_uncertainty


def test_bootstrap_to_raster_is_reproducible(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    rng = np.random.default_rng(10)
    y = np.array([0, 1] * 25)
    training = pd.DataFrame({
        "bio1": rng.normal(size=50) + y,
        "presence": y,
    })
    raster = tmp_path / "predictor.tif"
    with rasterio.open(
        raster, "w", driver="GTiff", height=2, width=2, count=1,
        dtype="float32", nodata=-9999.0,
        crs="EPSG:4326", transform=from_origin(-85, 40, 0.1, 0.1),
    ) as ds:
        ds.write(np.array([[0.2, 0.8], [0.6, -9999.0]], dtype="float32"), 1)
    groups = [str(i // 5) for i in range(50)]
    inputs = (training, ["bio1"], {"bio1": raster})
    a = run_bootstrap_raster_uncertainty(
        *inputs, tmp_path / "a", training_groups=groups,
        n_resamples=3, n_estimators=10, threshold=0.5, window_size=1,
    )
    b = run_bootstrap_raster_uncertainty(
        *inputs, tmp_path / "b", training_groups=groups,
        n_resamples=3, n_estimators=10, threshold=0.5, window_size=2,
    )
    with rasterio.open(a.raster.mean_path) as first:
        with rasterio.open(b.raster.mean_path) as second:
            np.testing.assert_allclose(first.read(1), second.read(1))
            assert first.read(1)[1, 1] == first.nodata
    assert a.raster.model_count == 3
    manifest = json.loads(a.manifest_path.read_text())
    assert manifest["n_groups"] == 10
    assert manifest["resampling_unit"] == "spatial_group"


def test_bootstrap_raster_validation(tmp_path):
    training = pd.DataFrame({"bio1": [1, 2, 3, 4], "presence": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="n_resamples"):
        run_bootstrap_raster_uncertainty(
            training, ["bio1"], {}, tmp_path, n_resamples=1,
        )
    with pytest.raises(ValueError, match="row count"):
        run_bootstrap_raster_uncertainty(
            training, ["bio1"], {}, tmp_path, training_groups=["a"],
        )
