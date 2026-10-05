from pathlib import Path

import numpy as np
import pytest

from rangeshift.crossed_uncertainty import summarize_crossed_uncertainty


class ShiftModel:
    classes_ = np.array([0, 1])

    def __init__(self, shift):
        self.shift = shift

    def predict_proba(self, frame):
        p = np.clip(frame["bio1"].to_numpy() + self.shift, 0, 1)
        return np.column_stack((1 - p, p))


def test_balanced_variance_decomposition_and_common_mask(tmp_path: Path):
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    rasters = {}
    for name, values in {
        "first": [[0.2, 0.4], [0.3, -9999]],
        "second": [[0.4, -9999], [0.5, 0.7]],
    }.items():
        p = tmp_path / (name + ".tif")
        with rio.open(
            p, "w", driver="GTiff", height=2, width=2, count=1,
            dtype="float32", crs="EPSG:4326",
            transform=from_origin(-85, 40, 0.1, 0.1), nodata=-9999.0,
        ) as dst:
            dst.write(np.array(values, dtype=np.float32), 1)
        rasters[name] = {"bio1": p}

    models = [
        {"model": ShiftModel(0.0), "feature_columns": ["bio1"]},
        {"model": ShiftModel(0.2), "feature_columns": ["bio1"]},
    ]
    result = summarize_crossed_uncertainty(
        models, rasters, tmp_path / "outputs", threshold=0.5, window_size=1
    )
    assert result.valid_cells == 2
    with rio.open(result.scenario_sd_path) as scenario:
        with rio.open(result.model_sd_path) as model:
            with rio.open(result.total_sd_path) as total:
                between = scenario.read(1)
                within = model.read(1)
                combined = total.read(1)
                assert between[0, 0] == pytest.approx(0.1)
                assert within[0, 0] == pytest.approx(0.1)
                assert combined[0, 0] ** 2 == pytest.approx(
                    between[0, 0] ** 2 + within[0, 0] ** 2
                )
                assert combined[0, 1] == total.nodata
                assert combined[1, 1] == total.nodata
    with rio.open(result.suitable_fraction_path) as ds:
        assert ds.read(1)[0, 0] == pytest.approx(0.25)


def test_crossed_requires_two_of_each(tmp_path: Path):
    model = {"model": ShiftModel(0), "feature_columns": ["bio1"]}
    with pytest.raises(ValueError, match="at least two models"):
        summarize_crossed_uncertainty([model], {}, tmp_path)
