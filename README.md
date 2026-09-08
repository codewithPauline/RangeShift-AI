# RangeShift AI

> Machine-learning tools for predicting species habitat suitability and exploring potential geographic range shifts under environmental change.

[![CI](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

## Why RangeShift AI?

Species ranges are not static. Climate, land use, topography, and other environmental pressures can change where suitable habitat exists. RangeShift AI is being built as a transparent, reproducible machine-learning toolkit for moving from species occurrence and environmental data to habitat-suitability predictions and, ultimately, projected geographic range shifts.

The project is developed in stages so that each ecological assumption and machine-learning decision is visible, testable, and explainable rather than hidden inside a black-box workflow.

## Project status

**Active development — v0.2 spatial intelligence.**

RangeShift AI can now answer two different model-evaluation questions:

1. **Random holdout:** how well does the model predict randomly withheld observations?
2. **Spatial holdout:** how well does the model predict observations from geographic blocks that were completely absent from training?

That distinction matters because nearby samples often share both environment and spatial structure. A random split can therefore produce an optimistic estimate of how well a species-distribution model will transfer to genuinely new geography.

### Current capabilities

- load occurrence/background CSV data;
- validate binary targets and numeric environmental predictors;
- validate latitude/longitude coordinates;
- train a class-balanced `RandomForestClassifier`;
- report ROC-AUC, accuracy, precision, recall, and F1;
- rank environmental predictors by feature importance;
- save and reload trained model bundles;
- predict continuous habitat-suitability probabilities;
- assign observations to non-overlapping spatial grid blocks;
- evaluate a model with complete spatial blocks held out;
- compare random-split and spatial-split performance directly;
- guarantee that spatial train/test blocks do not overlap;
- optionally convert coordinate tables to GeoPandas `GeoDataFrame` objects;
- run training, prediction, and spatial comparison from the command line;
- test the core workflow automatically with GitHub Actions.

## Roadmap

### Phase 1 — Current habitat suitability
- [x] Project architecture
- [x] Baseline Random Forest classifier
- [x] Model evaluation
- [x] Feature importance
- [x] Model persistence and prediction
- [x] CLI entry point
- [x] Automated tests and CI
- [ ] Real ecological example dataset

### Phase 2 — Spatial intelligence
- [x] Coordinate validation
- [x] Spatial block assignment
- [x] Spatially separated train/test evaluation
- [x] Random-vs-spatial performance comparison
- [x] Optional GeoPandas conversion
- [ ] CRS-aware projected spatial blocks
- [ ] Repeated spatial cross-validation
- [ ] Sampling-bias diagnostics

### Phase 3 — Environmental rasters and mapping
- [ ] Read raster predictor layers
- [ ] Predict suitability across a geographic grid
- [ ] Export GeoTIFF predictions
- [ ] Publication-quality suitability maps

### Phase 4 — Future range-shift projection
- [ ] Future climate/environmental scenario input
- [ ] Current vs. future suitability comparison
- [ ] Stable, gained, and lost habitat classes
- [ ] Range contraction/expansion estimates
- [ ] Geographic centroid movement and shift distance

### Phase 5 — Explainable and robust ML
- [ ] Hyperparameter tuning
- [ ] Gradient-boosted comparison model
- [ ] SHAP-based model interpretation
- [ ] Calibration and threshold analysis
- [ ] Environmental extrapolation diagnostics

### Phase 6 — User-facing tool
- [ ] Reproducible end-to-end workflow
- [ ] Interactive visualization interface
- [ ] Documentation/tutorials
- [ ] Packaged release

See the detailed learning and scientific plan in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Input data format

The baseline model expects a binary response column and numeric environmental predictors. Spatial evaluation additionally requires latitude and longitude.

```csv
presence,bio1,bio12,elevation,latitude,longitude
1,13.2,1050,310,39.51,-84.74
1,14.1,980,270,39.72,-84.10
0,18.9,620,120,37.98,-86.12
0,20.4,540,80,36.43,-85.51
```

`presence` should contain `1` for observed/presence records and `0` for background or absence records.

> **Scientific note:** pseudo-absence/background generation can strongly affect species-distribution models. RangeShift AI does not yet generate these points automatically. That ecological decision will remain explicit rather than being silently made for the user.

## Installation

```bash
git clone https://github.com/codewithPauline/RangeShift-AI.git
cd RangeShift-AI
python -m venv .venv
pip install -e .
```

For GeoPandas support:

```bash
pip install -e ".[geo]"
```

For development and testing:

```bash
pip install -e ".[dev]"
```

## Command-line usage

### Train and save a model

```bash
rangeshift train data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --output model.joblib
```

### Predict habitat suitability

```bash
rangeshift predict model.joblib future_environment.csv \
  --output suitability_predictions.csv
```

The output preserves the input rows and adds a `suitability` probability between 0 and 1.

### Compare random and spatial evaluation

```bash
rangeshift compare-spatial data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --latitude latitude \
  --longitude longitude \
  --block-size 1.0 \
  --output spatial_comparison.json
```

The command reports three groups of results:

- `random`: conventional stratified random holdout metrics;
- `spatial`: metrics from complete held-out geographic blocks;
- `spatial_minus_random`: the change in each metric under the harder spatial test.

A negative `spatial_minus_random` value means performance declined when the model had to transfer to geography that was not represented in training.

## Python example

```python
import pandas as pd
from rangeshift.spatial import compare_random_and_spatial

frame = pd.read_csv("data.csv")

result = compare_random_and_spatial(
    frame,
    feature_columns=["bio1", "bio12", "elevation"],
    target_column="presence",
    latitude_column="latitude",
    longitude_column="longitude",
    block_size_degrees=1.0,
)

print(result.random.metrics)
print(result.spatial.metrics)
print(result.spatial_minus_random)
```

Runnable demonstrations are available in:

- [`examples/train_demo.py`](examples/train_demo.py)
- [`examples/spatial_demo.py`](examples/spatial_demo.py)

## How spatial blocking currently works

RangeShift v0.2 divides latitude/longitude coordinates into fixed geographic grid cells and uses those cells as groups during model splitting. Entire blocks are assigned to either training or testing, so no spatial block appears in both partitions.

This is deliberately a **baseline spatial diagnostic**. Geographic degrees are not equal-area units: one degree of longitude represents different physical distances at different latitudes. The next spatial milestone will add projected, CRS-aware blocking for analyses where block size needs to correspond to real distances.

## GeoPandas integration

The lightweight ML core does not require a full GIS stack. Users who install the optional `geo` dependency can convert coordinate tables into a WGS84 `GeoDataFrame`:

```python
from rangeshift.spatial import to_geodataframe

gdf = to_geodataframe(frame)
print(gdf.crs)
```

This keeps the core package easy to install while providing a clean path toward projections, spatial joins, raster sampling, and mapping.

## Repository structure

```text
RangeShift-AI/
├── .github/workflows/     # continuous integration
├── docs/                  # project design and scientific roadmap
├── examples/              # runnable demonstrations
├── src/rangeshift/        # Python package
├── tests/                 # automated tests
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## Design principles

RangeShift AI is being developed around four principles:

1. **Reproducibility** — the same inputs and random seed should reproduce the same baseline result.
2. **Scientific transparency** — ecological assumptions should be explicit.
3. **Software quality** — tested, modular code rather than a single analysis notebook.
4. **Interpretability** — model performance and predictor effects should be inspectable.

## What this project is not

RangeShift AI does not assume that a machine-learning suitability score is automatically equivalent to a realized species range. Dispersal limits, biotic interactions, sampling bias, evolutionary adaptation, land-use change, detectability, and extrapolation into novel environmental space can all affect real distributions and future range shifts.

## Author

**Pauline Owusu-Ansah**  
Ph.D. researcher in computational and evolutionary biology

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for the full license text.
