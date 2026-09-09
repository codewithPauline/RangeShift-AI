import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rangeshift.config import RunConfig, load_run_config, run_configured_analysis


def test_load_run_config_validates_feature_layer_contract(tmp_path: Path) -> None:
    payload = {
        "training_csv": "training.csv",
        "features": ["bio1", "bio12"],
        "current_layers": {"bio1": "current1.tif", "bio12": "current12.tif"},
        "future_layers": {"bio1": "future1.tif", "bio12": "future12.tif"},
        "output_dir": "out",
        "spatial_validation": False,
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))
    config = load_run_config(path)
    assert config.features == ["bio1", "bio12"]
    assert config.output_dir == Path("out")

    payload["future_layers"] = {"bio1": "future1.tif"}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Future layer mapping"):
        load_run_config(path)


def test_configured_analysis_produces_manifest_and_range_outputs(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    rng = np.random.default_rng(22)
    bio1 = rng.normal(13.0, 2.0, 160)
    bio12 = rng.normal(900.0, 120.0, 160)
    signal = -(bio1 - 13.0) ** 2 / 5.0 + (bio12 - 880.0) / 150.0
    presence = (signal + rng.normal(0.0, 0.7, 160) > 0.0).astype(int)
    training = pd.DataFrame(
        {
            "presence": presence,
            "bio1": bio1,
            "bio12": bio12,
            "latitude": rng.uniform(38.0, 41.0, 160),
            "longitude": rng.uniform(-86.0, -82.0, 160),
        }
    )
    training_path = tmp_path / "training.csv"
    training.to_csv(training_path, index=False)

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

    current_bio1 = tmp_path / "current_bio1.tif"
    current_bio12 = tmp_path / "current_bio12.tif"
    future_bio1 = tmp_path / "future_bio1.tif"
    future_bio12 = tmp_path / "future_bio12.tif"
    write_raster(current_bio1, np.array([[11, 12, 13], [14, 15, 16]], dtype=float))
    write_raster(current_bio12, np.array([[1050, 980, 900], [850, 780, 720]], dtype=float))
    write_raster(future_bio1, np.array([[12, 13, 14], [15, 16, 17]], dtype=float))
    write_raster(future_bio12, np.array([[1000, 930, 860], [810, 740, 680]], dtype=float))

    config = RunConfig(
        training_csv=training_path,
        features=["bio1", "bio12"],
        current_layers={"bio1": current_bio1, "bio12": current_bio12},
        future_layers={"bio1": future_bio1, "bio12": future_bio12},
        output_dir=tmp_path / "run",
        calibration_cv=3,
        spatial_validation=False,
        raster_window_size=2,
    )
    result = run_configured_analysis(config)

    assert result.manifest_path.exists()
    assert result.model_path.exists()
    assert result.current_suitability_path.exists()
    assert result.future_suitability_path.exists()
    assert result.range_shift_classes_path.exists()
    assert result.range_shift_summary_path.exists()
    assert len(result.config_sha256) == 64

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["config_sha256"] == result.config_sha256
    assert 0.0 < manifest["selected_threshold"] < 1.0
