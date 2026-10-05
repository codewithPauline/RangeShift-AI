import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _script():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/case_studies/plethodon_cinereus/run_crossed_uncertainty.py"
    )
    spec = importlib.util.spec_from_file_location("plethodon_crossed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_case_study_crossed_end_to_end(tmp_path):
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    rng = np.random.default_rng(20)
    n = 40
    y = np.array([0, 1] * 20)
    training = pd.DataFrame({
        "bio1": rng.uniform(1, 15, n) + y,
        "bio12": rng.uniform(300, 600, n),
        "bio15": rng.uniform(10, 30, n),
        "presence": y,
        "longitude": np.tile([-84.5, -81.5, -78.5, -75.5], 10),
        "latitude": np.tile([38.2, 40.2, 42.2, 44.2], 10),
    })
    train_path = tmp_path / "training.csv"
    training.to_csv(train_path, index=False)
    scenarios = {}
    for index in range(2):
        layers = {}
        for feature, baseline in [("bio1", 10), ("bio12", 420), ("bio15", 18)]:
            path = tmp_path / f"{index}_{feature}.tif"
            with rio.open(
                path, "w", driver="GTiff", height=2, width=2,
                count=1, dtype="float32", crs="EPSG:4326",
                transform=from_origin(-85, 42, 0.1, 0.1), nodata=-9999.0,
            ) as dataset:
                dataset.write(
                    np.array([
                        [baseline + index, baseline + index + 1],
                        [baseline - 1, -9999],
                    ], dtype=np.float32), 1,
                )
            layers[feature] = str(path)
        scenarios[f"scenario_{index}"] = layers
    scenario_path = tmp_path / "scenario_layers.json"
    scenario_path.write_text(json.dumps(scenarios))
    result = _script().run_case_study(
        train_path, scenario_path, tmp_path / "results",
        n_models=2, n_estimators=5,
    )
    manifest = json.loads(result.read_text())
    assert manifest["n_models"] == 2
    assert manifest["n_scenarios"] == 2
    assert manifest["common_valid_cells"] == 3
    assert Path(manifest["model_sd_raster"]).is_file()
