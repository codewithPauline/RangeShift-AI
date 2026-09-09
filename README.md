# RangeShift AI

> Geospatial machine-learning tools for habitat suitability modeling and transparent current-to-future range-shift analysis.

[![CI](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

## Why RangeShift AI?

Species ranges are not static. Climate, land use, topography, and other environmental pressures can alter where suitable habitat exists. RangeShift AI is being built as a transparent, reproducible toolkit that connects species occurrence data, environmental predictors, machine learning, spatial validation, raster projection, and future range-change analysis.

The project is deliberately designed so ecological assumptions remain visible. RangeShift does not hide spatial validation, threshold choice, raster preprocessing, or range-change definitions inside a black box.

## Project status

**Active development — v0.4 current-to-future range-shift analysis.**

RangeShift AI now supports this end-to-end workflow:

```text
Occurrence/background data
        ↓
Habitat-suitability model
        ↓
Spatial validation
        ↓
Current environmental rasters ──→ Current suitability GeoTIFF
        ↓
Future environmental rasters  ──→ Future suitability GeoTIFF
        ↓
Explicit suitability threshold
        ↓
Stable / Lost / Gained / Stable-suitable habitat
        ↓
Area change + overlap + centroid shift
        ↓
GeoTIFFs + JSON summary + high-resolution maps
```

### Current capabilities

- load and validate occurrence/background CSV data;
- train a class-balanced `RandomForestClassifier`;
- report ROC-AUC, accuracy, precision, recall, and F1;
- rank environmental predictors by feature importance;
- save and reload trained model bundles;
- predict continuous habitat-suitability probabilities for tabular data;
- validate latitude/longitude coordinates;
- compare random and spatial holdout performance;
- run repeated spatial block cross-validation;
- create projected kilometer-scale spatial blocks;
- validate aligned environmental raster stacks;
- require raster predictor names to match the trained model exactly;
- propagate raster nodata and non-finite values;
- predict 0–1 suitability across geographic raster grids;
- export compressed suitability GeoTIFFs;
- render high-resolution suitability maps;
- compare aligned current and future suitability rasters;
- require an explicit suitability threshold for range classification;
- classify cells as stable unsuitable, lost, gained, or stable suitable;
- export a continuous future-minus-current suitability raster;
- calculate suitable area under current and future scenarios;
- calculate gained, lost, stable, and net area change;
- calculate area-weighted Jaccard overlap;
- calculate area-weighted current and future geographic centroids;
- report centroid shift distance and compass bearing;
- calculate cell area correctly for both projected and geographic CRSs;
- render discrete range-shift transition maps;
- expose all major workflows through a command-line interface;
- test core and geospatial functionality automatically with GitHub Actions.

## Roadmap

### Phase 1 — Current habitat suitability
- [x] Project architecture
- [x] Random Forest baseline
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
- [x] GeoPandas integration
- [x] Projected spatial blocks measured in kilometers
- [ ] Sampling-bias diagnostics
- [ ] Train/test block visualization

### Phase 3 — Environmental rasters and mapping
- [x] Read aligned raster predictors
- [x] Validate dimensions, CRS, transform, nodata, and feature names
- [x] Predict suitability across a raster grid
- [x] Export suitability GeoTIFFs
- [x] Render high-resolution suitability maps
- [x] Optional vector boundary overlay
- [ ] Windowed prediction for very large rasters

### Phase 4 — Future range-shift projection
- [x] Current and future suitability raster comparison
- [x] Explicit threshold requirement
- [x] Stable / gained / lost habitat classes
- [x] Continuous suitability-change raster
- [x] Area gained and lost
- [x] Net range-area change
- [x] Jaccard overlap
- [x] Current/future suitable-range centroids
- [x] Centroid shift distance and bearing
- [x] Range-shift transition map
- [x] JSON summary output

### Phase 5 — Explainable and robust ML
- [ ] Threshold-selection methods
- [ ] Probability calibration
- [ ] Hyperparameter tuning
- [ ] Gradient-boosted comparison model
- [ ] SHAP-based interpretation
- [ ] Partial dependence / response curves
- [ ] Environmental extrapolation diagnostics

### Phase 6 — Ecological safeguards and productization
- [ ] Pseudo-absence/background generation strategies
- [ ] Spatial thinning
- [ ] Environmental collinearity diagnostics
- [ ] Novel-climate warnings
- [ ] Dispersal constraints
- [ ] Reproducible configuration files
- [ ] Interactive visualization interface
- [ ] Packaged release

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the detailed scientific and development plan.

## Installation

Install the lightweight ML core:

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

## Tabular input format

```csv
presence,bio1,bio12,elevation,latitude,longitude
1,13.2,1050,310,39.51,-84.74
1,14.1,980,270,39.72,-84.10
0,18.9,620,120,37.98,-86.12
0,20.4,540,80,36.43,-85.51
```

`presence` should contain `1` for observed/presence records and `0` for background or absence records.

> **Scientific note:** pseudo-absence/background generation can materially affect species-distribution models. RangeShift does not yet generate these points automatically; that choice remains explicit.

## Raster requirements

Raster prediction expects one single-band raster per trained feature. RangeShift verifies that all predictor layers share the same:

- dimensions;
- coordinate reference system;
- affine transform and pixel grid;
- valid coverage for predicted cells;
- model feature names.

RangeShift deliberately does **not** silently reproject, crop, or resample mismatched environmental layers.

Current and future suitability rasters used in range-shift analysis must also be aligned to exactly the same grid and CRS.

## Command-line usage

### Train a model

```bash
rangeshift train data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --output model.joblib
```

### Compare random and spatial evaluation

```bash
rangeshift compare-spatial data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --block-size 1.0 \
  --output spatial_comparison.json
```

### Run repeated spatial cross-validation

```bash
rangeshift spatial-cv data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --splits 5 \
  --output spatial_cv.csv
```

### Predict current suitability

```bash
rangeshift predict-raster model.joblib \
  --layer bio1=current/bio1.tif \
  --layer bio12=current/bio12.tif \
  --layer elevation=current/elevation.tif \
  --output current_suitability.tif
```

### Predict future suitability

Use the **same fitted model** with future environmental layers:

```bash
rangeshift predict-raster model.joblib \
  --layer bio1=future/bio1.tif \
  --layer bio12=future/bio12.tif \
  --layer elevation=future/elevation.tif \
  --output future_suitability.tif
```

### Quantify the range shift

The threshold is required. RangeShift never silently assumes that `0.5` defines suitable habitat.

```bash
rangeshift range-shift \
  current_suitability.tif \
  future_suitability.tif \
  --threshold 0.55 \
  --classes-output range_shift_classes.tif \
  --difference-output suitability_change.tif \
  --summary-output range_shift_summary.json
```

The transition raster uses these class codes:

| Code | Interpretation |
| ---: | --- |
| 0 | Stable unsuitable |
| 1 | Lost suitable habitat |
| 2 | Gained suitable habitat |
| 3 | Stable suitable habitat |
| 255 | Nodata |

The JSON summary includes suitable area under each scenario, gained/lost/stable area, net change, Jaccard overlap, current and future geographic centroids, centroid shift distance, and bearing.

### Render the range-shift map

```bash
rangeshift plot-range-shift range_shift_classes.tif \
  --title "Projected habitat range shift" \
  --output range_shift_map.png \
  --dpi 300
```

An optional study-area boundary can be overlaid using `--boundary study_area.gpkg`.

## Area and centroid calculations

RangeShift does not treat map degrees as physical distance.

- For **projected CRSs**, cell area is calculated from the affine transform and the CRS linear-unit conversion factor.
- For **geographic CRSs**, each raster cell area is calculated geodesically on the ellipsoid using PyProj.
- Suitable-range centroids are area-weighted and calculated geographically.
- Centroid movement is reported as geodesic distance in kilometers and forward bearing in degrees.

This matters because a one-degree raster cell does not represent the same physical area at every latitude.

## Python example

```python
from rangeshift.range_shift import compare_suitability_rasters
from rangeshift.visualization import plot_range_shift_map

result = compare_suitability_rasters(
    "current_suitability.tif",
    "future_suitability.tif",
    "range_shift_classes.tif",
    threshold=0.55,
    difference_output_path="suitability_change.tif",
)

print(result.to_dict())

plot_range_shift_map(
    result.classes_path,
    "range_shift_map.png",
    title="Projected habitat range shift",
)
```

## Runnable demonstrations

- [`examples/train_demo.py`](examples/train_demo.py) — baseline supervised learning.
- [`examples/spatial_demo.py`](examples/spatial_demo.py) — random vs. spatial evaluation.
- [`examples/raster_demo.py`](examples/raster_demo.py) — raster prediction and suitability map.
- [`examples/range_shift_demo.py`](examples/range_shift_demo.py) — complete synthetic current-to-future range-shift workflow.

The v0.4 demo trains one model, creates synthetic current and future environmental rasters, predicts both suitability surfaces, calculates the range shift, and renders all major outputs without external ecological data.

## Scientific interpretation

A predicted range shift is a **scenario-conditioned suitability projection**, not a guaranteed future distribution. The result depends on the fitted model, environmental predictors, climate/environmental scenario, threshold choice, sampling design, and the assumption that modeled environment–occurrence relationships transfer to the future.

Realized ranges may also be constrained by dispersal, barriers, biotic interactions, demography, adaptation, land-use change, detectability, and environmental novelty.

RangeShift therefore keeps the threshold visible, preserves the continuous suitability difference, and reports spatial transfer performance rather than presenting the classified map as certainty.

## Design principles

1. **Reproducibility** — identical inputs and seeds should reproduce baseline results.
2. **Scientific transparency** — assumptions and thresholds should remain visible.
3. **Software quality** — modular, tested code instead of one monolithic notebook.
4. **Interpretability** — performance, spatial transfer, predictors, and range-change outputs should be inspectable.

## Author

**Pauline Owusu-Ansah**  
Ph.D. researcher in computational and evolutionary biology

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for the full license text.
