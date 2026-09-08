# RangeShift AI

> Machine-learning tools for predicting species habitat suitability and exploring potential geographic range shifts under environmental change.

[![CI](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

## Why RangeShift AI?

Species ranges are not static. Climate, land use, topography, and other environmental pressures can change where suitable habitat exists. RangeShift AI is being built as a transparent, reproducible machine-learning toolkit for moving from species occurrence and environmental data to habitat-suitability predictions and, ultimately, projected geographic range shifts.

The project is developed in stages so that ecological assumptions and machine-learning decisions remain visible, testable, and explainable rather than hidden inside a black-box workflow.

## Project status

**Active development — v0.3 raster suitability prediction and mapping.**

RangeShift AI now connects four parts of a species-distribution workflow:

1. **Model learning** — train and evaluate a habitat-suitability classifier from tabular environmental data.
2. **Spatial validation** — test whether performance survives geographic separation between training and evaluation samples.
3. **Raster projection** — apply a saved model to aligned environmental GeoTIFFs and export continuous habitat-suitability probabilities across a geographic grid.
4. **Map rendering** — convert the suitability GeoTIFF into a high-resolution map with a fixed 0–1 scale and optional study-area boundary overlay.

### Current capabilities

- load occurrence/background CSV data;
- validate binary targets and numeric environmental predictors;
- train a class-balanced `RandomForestClassifier`;
- report ROC-AUC, accuracy, precision, recall, and F1;
- rank predictors by feature importance;
- save and reload trained model bundles;
- predict continuous habitat-suitability probabilities for tabular data;
- validate latitude/longitude coordinates;
- assign observations to non-overlapping degree-based spatial blocks;
- compare random and spatial holdout performance;
- run repeated spatial block cross-validation;
- create kilometer-scale projected blocks using a local UTM CRS or a user-specified EPSG code;
- optionally convert coordinate tables to GeoPandas `GeoDataFrame` objects;
- validate aligned single-band environmental raster stacks;
- require raster predictor names to match the trained model exactly;
- propagate nodata and non-finite raster cells into the output mask;
- predict suitability probability for every complete raster cell;
- export compressed `float32` GeoTIFF suitability predictions;
- render 300-DPI suitability maps with a fixed 0–1 color scale;
- optionally overlay a vector study-area boundary after CRS reprojection;
- use CRS-aware longitude/latitude or easting/northing axis labels;
- run the core workflows through a command-line interface;
- test core and geospatial functionality automatically with GitHub Actions.

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
- [x] Degree-based spatial blocks
- [x] Spatially separated train/test evaluation
- [x] Random-vs-spatial performance comparison
- [x] Repeated spatial cross-validation
- [x] Optional GeoPandas conversion
- [x] CRS-aware projected spatial blocks measured in kilometers
- [ ] Sampling-bias diagnostics
- [ ] Train/test block visualization

### Phase 3 — Environmental rasters and mapping
- [x] Read aligned raster predictor layers
- [x] Validate shape, CRS, transform, nodata, and predictor names
- [x] Predict suitability across a geographic grid
- [x] Export GeoTIFF predictions
- [x] High-resolution suitability-map rendering
- [x] Optional study-area boundary overlay
- [ ] Chunked/windowed prediction for very large rasters
- [ ] Additional publication-map refinements and observation overlays

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

See the detailed scientific and development plan in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Tabular input format

The baseline model expects a binary response column and numeric environmental predictors. Spatial evaluation additionally requires latitude and longitude.

```csv
presence,bio1,bio12,elevation,latitude,longitude
1,13.2,1050,310,39.51,-84.74
1,14.1,980,270,39.72,-84.10
0,18.9,620,120,37.98,-86.12
0,20.4,540,80,36.43,-85.51
```

`presence` should contain `1` for observed/presence records and `0` for background or absence records.

> **Scientific note:** pseudo-absence/background generation can strongly affect species-distribution models. RangeShift AI does not yet generate these points automatically. That ecological decision remains explicit rather than being silently made for the user.

## Raster input requirements

Raster prediction expects **one single-band raster per trained model feature**. Before prediction, RangeShift verifies that all layers have:

- the same number of rows and columns;
- the same coordinate reference system;
- the same affine transform and pixel grid;
- complete, finite predictor values for every cell that will be predicted;
- predictor names that match the saved model exactly.

RangeShift deliberately does **not** silently crop, reproject, or resample mismatched predictors. Those preprocessing choices can materially affect ecological inference and should be made explicitly before model projection.

The current v0.3 implementation loads the complete predictor stack into memory. Windowed/chunked prediction is planned for large continental or global raster datasets.

## Installation

Clone and install the lightweight ML core:

```bash
git clone https://github.com/codewithPauline/RangeShift-AI.git
cd RangeShift-AI
python -m venv .venv
pip install -e .
```

Install geospatial support, including GeoPandas, Matplotlib, PyProj, and Rasterio:

```bash
pip install -e ".[geo]"
```

For development and testing:

```bash
pip install -e ".[dev,geo]"
```

## Command-line usage

### Train and save a model

```bash
rangeshift train data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --output model.joblib
```

### Predict tabular habitat suitability

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

A negative `spatial_minus_random` metric means performance declined when the model had to transfer to geography that was not represented in training.

### Run repeated spatial cross-validation

```bash
rangeshift spatial-cv data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --block-size 1.0 \
  --splits 5 \
  --output spatial_cv.csv
```

### Create kilometer-scale projected blocks

```bash
rangeshift project-blocks data.csv \
  --block-km 100 \
  --output projected_blocks.csv
```

For regional data within UTM coverage, RangeShift can estimate a local UTM CRS automatically. A specific projected CRS can also be supplied with `--epsg`.

### Predict a habitat-suitability GeoTIFF

```bash
rangeshift predict-raster model.joblib \
  --layer bio1=rasters/bio1.tif \
  --layer bio12=rasters/bio12.tif \
  --layer elevation=rasters/elevation.tif \
  --output habitat_suitability.tif
```

Each `--layer` maps one trained feature name to one aligned environmental raster. The output is a single-band `float32` GeoTIFF containing suitability probabilities from 0 to 1, with invalid predictor cells written as nodata.

### Render the suitability map

```bash
rangeshift plot-raster habitat_suitability.tif \
  --title "Predicted habitat suitability" \
  --output habitat_suitability.png \
  --dpi 300
```

An optional vector boundary can be added with `--boundary study_area.gpkg`. RangeShift reprojects the boundary to the raster CRS before drawing it.

## Python example

```python
from rangeshift.prediction import load_model_bundle
from rangeshift.raster import predict_suitability_raster
from rangeshift.visualization import plot_suitability_map

bundle = load_model_bundle("model.joblib")

result = predict_suitability_raster(
    bundle,
    {
        "bio1": "rasters/bio1.tif",
        "bio12": "rasters/bio12.tif",
        "elevation": "rasters/elevation.tif",
    },
    "habitat_suitability.tif",
)

plot_suitability_map(
    result.output_path,
    "habitat_suitability.png",
    title="Predicted habitat suitability",
)
```

A fully synthetic end-to-end demonstration is available at [`examples/raster_demo.py`](examples/raster_demo.py). Running it creates environmental rasters, trains a model, generates a suitability GeoTIFF, and renders a 300-DPI PNG without requiring external ecological data.

## Scientific interpretation

A suitability raster is a **model projection**, not a guaranteed realized species distribution. High predicted suitability does not mean a species will necessarily occupy a cell. Dispersal limits, biotic interactions, demographic processes, sampling bias, land-use barriers, detectability, adaptation, and environmental novelty can all separate potential suitability from realized range occupancy.

Similarly, model performance from random train/test splitting can be optimistic when observations are spatially autocorrelated. RangeShift therefore exposes spatial validation alongside conventional evaluation rather than treating geography as an afterthought.

## Design principles

RangeShift AI is being developed around four principles:

1. **Reproducibility** — identical inputs and seeds should reproduce baseline results.
2. **Scientific transparency** — ecological and geospatial assumptions should be explicit.
3. **Software quality** — tested, modular code rather than one monolithic notebook.
4. **Interpretability** — performance, predictors, spatial transfer, and outputs should be inspectable.

## Author

**Pauline Owusu-Ansah**  
Ph.D. researcher in computational and evolutionary biology

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for the full license text.
