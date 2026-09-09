"""Run an end-to-end synthetic current-to-future RangeShift demonstration.

Install the geospatial extras before running:

    pip install -e ".[geo]"
    python examples/range_shift_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from rangeshift.model import train_habitat_model
from rangeshift.range_shift import compare_suitability_rasters
from rangeshift.raster import predict_suitability_raster
from rangeshift.visualization import plot_range_shift_map, plot_suitability_map


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
        transform=from_origin(-88.0, 43.0, 0.08, 0.08),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(values.astype(np.float32), 1)


def main() -> None:
    """Train, project current/future environments, and quantify the range shift."""
    rng = np.random.default_rng(82)
    n = 500
    temperature = rng.normal(14.0, 3.0, n)
    precipitation = rng.normal(900.0, 180.0, n)
    elevation = rng.normal(260.0, 100.0, n)

    ecological_signal = (
        -((temperature - 13.5) ** 2) / 8.0
        + (precipitation - 850.0) / 260.0
        - ((elevation - 280.0) ** 2) / 50000.0
        + rng.normal(0.0, 0.55, n)
    )
    presence = (ecological_signal > 0).astype(int)
    training = pd.DataFrame(
        {
            "presence": presence,
            "temperature": temperature,
            "precipitation": precipitation,
            "elevation": elevation,
        }
    )

    trained = train_habitat_model(
        training,
        feature_columns=["temperature", "precipitation", "elevation"],
        n_estimators=180,
        random_state=82,
    )
    bundle = {
        "model": trained.model,
        "feature_columns": trained.feature_columns,
        "target_column": trained.target_column,
        "metrics": trained.metrics,
    }

    rows, cols = 55, 80
    west_east = np.linspace(0.0, 1.0, cols)[None, :]
    north_south = np.linspace(0.0, 1.0, rows)[:, None]

    current_temperature = 10.0 + 7.0 * west_east + 2.0 * north_south
    current_precipitation = 1120.0 - 420.0 * west_east - 90.0 * north_south
    elevation = 180.0 + 220.0 * north_south + 60.0 * np.sin(west_east * np.pi)

    future_temperature = current_temperature + 2.3 + 0.8 * north_south
    future_precipitation = current_precipitation - 85.0 + 40.0 * north_south

    output_dir = Path("rangeshift_demo_output")
    output_dir.mkdir(exist_ok=True)

    current_layers = {
        "temperature": output_dir / "current_temperature.tif",
        "precipitation": output_dir / "current_precipitation.tif",
        "elevation": output_dir / "elevation.tif",
    }
    future_layers = {
        "temperature": output_dir / "future_temperature.tif",
        "precipitation": output_dir / "future_precipitation.tif",
        "elevation": output_dir / "elevation.tif",
    }

    _write_raster(current_layers["temperature"], current_temperature)
    _write_raster(current_layers["precipitation"], current_precipitation)
    _write_raster(current_layers["elevation"], elevation)
    _write_raster(future_layers["temperature"], future_temperature)
    _write_raster(future_layers["precipitation"], future_precipitation)

    current_suitability = output_dir / "current_suitability.tif"
    future_suitability = output_dir / "future_suitability.tif"
    transition_raster = output_dir / "range_shift_classes.tif"
    difference_raster = output_dir / "suitability_change.tif"

    predict_suitability_raster(bundle, current_layers, current_suitability)
    predict_suitability_raster(bundle, future_layers, future_suitability)

    threshold = 0.55
    result = compare_suitability_rasters(
        current_suitability,
        future_suitability,
        transition_raster,
        threshold=threshold,
        difference_output_path=difference_raster,
    )

    plot_suitability_map(
        current_suitability,
        output_dir / "current_suitability.png",
        title="Synthetic current habitat suitability",
    )
    plot_suitability_map(
        future_suitability,
        output_dir / "future_suitability.png",
        title="Synthetic future habitat suitability",
    )
    plot_range_shift_map(
        transition_raster,
        output_dir / "range_shift_map.png",
        title=f"Synthetic projected range shift — threshold {threshold:.2f}",
    )

    print("Model metrics:")
    print(trained.metrics)
    print("\nRange-shift summary:")
    for key, value in result.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
