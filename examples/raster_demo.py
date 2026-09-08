"""Run an end-to-end synthetic RangeShift raster prediction demo.

Install the geospatial extras before running:

    pip install -e ".[geo]"
    python examples/raster_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from rangeshift.model import train_habitat_model
from rangeshift.raster import predict_suitability_raster


def _write_raster(path: Path, values: np.ndarray) -> None:
    """Write one synthetic environmental predictor raster."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-86.0, 42.0, 0.1, 0.1),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(values.astype(np.float32), 1)


def main() -> None:
    """Train a small model and project it over synthetic environmental rasters."""
    rng = np.random.default_rng(44)
    n = 300
    temperature = rng.normal(14.0, 3.0, n)
    precipitation = rng.normal(900.0, 170.0, n)
    suitability_signal = (
        -((temperature - 13.5) ** 2) / 7.0
        + (precipitation - 850.0) / 230.0
        + rng.normal(0.0, 0.55, n)
    )
    presence = (suitability_signal > 0).astype(int)

    training = pd.DataFrame(
        {
            "presence": presence,
            "temperature": temperature,
            "precipitation": precipitation,
        }
    )
    trained = train_habitat_model(
        training,
        feature_columns=["temperature", "precipitation"],
        n_estimators=150,
        random_state=44,
    )
    bundle = {
        "model": trained.model,
        "feature_columns": trained.feature_columns,
        "target_column": trained.target_column,
        "metrics": trained.metrics,
    }

    rows, cols = 40, 60
    x_gradient = np.linspace(9.0, 20.0, cols)
    y_gradient = np.linspace(0.0, 3.0, rows)[:, None]
    temperature_grid = x_gradient[None, :] + y_gradient

    precipitation_west_east = np.linspace(1150.0, 600.0, cols)
    precipitation_north_south = np.linspace(80.0, -80.0, rows)[:, None]
    precipitation_grid = precipitation_west_east[None, :] + precipitation_north_south

    output_dir = Path("rangeshift_demo_output")
    output_dir.mkdir(exist_ok=True)
    temperature_path = output_dir / "temperature.tif"
    precipitation_path = output_dir / "precipitation.tif"
    suitability_path = output_dir / "habitat_suitability.tif"

    _write_raster(temperature_path, temperature_grid)
    _write_raster(precipitation_path, precipitation_grid)

    result = predict_suitability_raster(
        bundle,
        {
            "temperature": temperature_path,
            "precipitation": precipitation_path,
        },
        suitability_path,
    )

    print("Model metrics:")
    print(trained.metrics)
    print(f"Predicted {result.valid_cells:,} raster cells")
    print(f"CRS: {result.crs}")
    print(f"Suitability GeoTIFF: {result.output_path}")


if __name__ == "__main__":
    main()
