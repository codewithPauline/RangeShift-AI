import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def test_crossed_figure_writes_real_png_and_stats(tmp_path):
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    script = (
        Path(__file__).resolve().parents[1]
        / "examples/case_studies/plethodon_cinereus/make_crossed_figure.py"
    )
    spec = importlib.util.spec_from_file_location("crossed_figure", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rasters = tmp_path / "rasters"
    rasters.mkdir()
    for name, *_ in module.PANELS:
        with rio.open(
            rasters / name, "w", driver="GTiff", height=2, width=2, count=1,
            dtype="float32", crs="EPSG:4326", nodata=-9999.0,
            transform=from_origin(-85, 42, 0.1, 0.1),
        ) as dataset:
            dataset.write(
                np.array([[0.1, 0.3], [0.2, -9999]], dtype=np.float32), 1
            )
    png = tmp_path / "figure.png"
    summary = tmp_path / "stats.json"
    module.render(rasters, png, summary)
    assert png.is_file() and png.stat().st_size > 1000
    assert json.loads(summary.read_text())["crossed_mean.tif"]["valid_cells"] == 3
