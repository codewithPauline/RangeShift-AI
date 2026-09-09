# RangeShift AI

> Geospatial machine-learning tools for habitat suitability modeling and transparent current-to-future range-shift analysis.

[![CI](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

## Why RangeShift AI?

Species ranges are not static. Climate, land use, topography, and other environmental pressures can alter where suitable habitat exists. RangeShift AI is being built as a transparent, reproducible toolkit that connects species occurrence data, environmental predictors, machine learning, spatial validation, probability calibration, raster projection, and future range-change analysis.

The project is deliberately designed so ecological assumptions remain visible. RangeShift does not hide spatial validation, threshold choice, probability calibration, raster preprocessing, or range-change definitions inside a black box.

## Project status

**Active development — v0.5 probability calibration and defensible threshold selection.**

RangeShift AI now supports this end-to-end workflow:

```text
Occurrence/background data
        ↓
Train / validation / test separation
        ↓
Random Forest + probability calibration
        ↓
Threshold selection on validation data
        ↓
Final test evaluation
        ↓
Spatial validation
        ↓
Current environmental rasters ──→ Current suitability GeoTIFF
        ↓
Future environmental rasters  ──→ Future suitability GeoTIFF
        ↓
Explicit selected suitability threshold
        ↓
Stable / Lost / Gained / Stable-suitable habitat
        ↓
Area change + overlap + centroid shift
        ↓
GeoTIFFs + CSV/JSON diagnostics + high-resolution maps
```

### Current capabilities

- load and validate occurrence/background CSV data;
- train a class-balanced `RandomForestClassifier`;
- report ROC-AUC, accuracy, precision, recall, and F1;
- rank environmental predictors by feature importance;
- save and reload trained model bundles;
- calibrate Random Forest probabilities with sigmoid or isotonic calibration;
- keep training, validation, and final test partitions separate;
- select thresholds from validation predictions using TSS, Youden J, F1, or balanced accuracy;
- report sensitivity, specificity, precision, recall, F1, balanced accuracy, and predicted-positive rate across thresholds;
- report Brier score, log loss, and ROC-AUC for calibrated and uncalibrated test probabilities;
- save calibrated models using the same bundle contract as the raster pipeline;
- predict continuous habitat-suitability probabilities for tabular data;
- compare random and spatial holdout performance;
- run repeated spatial block cross-validation;
- create projected kilometer-scale spatial blocks;
- validate aligned environmental raster stacks;
- predict 0–1 suitability across geographic raster grids;
- export compressed suitability GeoTIFFs and high-resolution maps;
- compare aligned current and future suitability rasters;
- classify cells as stable unsuitable, lost, gained, or stable suitable;
- calculate gained, lost, stable, and net suitable area;
- calculate area-weighted Jaccard overlap;
- calculate current/future geographic centroids and shift distance/bearing;
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
- [x] Threshold diagnostics across candidate cutoffs
- [x] TSS / Youden J threshold selection
- [x] F1 and balanced-accuracy threshold selection
- [x] Separate train / validation / test workflow
- [x] Sigmoid and isotonic probability calibration
- [x] Brier score and log-loss reporting
- [x] Calibration diagnostic table
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

Install geospatial support:

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

## Threshold and calibration design

RangeShift does not choose a threshold from the final test set.

The v0.5 calibrated workflow uses three partitions:

1. **Training** — fit the Random Forest and perform cross-validated probability calibration.
2. **Validation** — choose the suitability threshold using an explicit metric.
3. **Test** — report final probability and threshold-based performance once, after the threshold is frozen.

Supported threshold criteria are `tss`, `youden_j`, `f1`, and `balanced_accuracy`. Probability calibration supports `sigmoid` and `isotonic` methods.

A lower Brier score or log loss indicates better probabilistic accuracy, but calibration is not assumed to improve every dataset. RangeShift reports both calibrated and uncalibrated test probability metrics so the effect remains inspectable.

## Raster requirements

Raster prediction expects one single-band raster per trained feature. RangeShift verifies matching dimensions, CRS, affine transform, valid coverage, and model feature names. RangeShift deliberately does **not** silently reproject, crop, or resample mismatched environmental layers.

Current and future suitability rasters used in range-shift analysis must also be aligned to exactly the same grid and CRS.

## Command-line usage

### Train a baseline model

```bash
rangeshift train data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --output model.joblib
```

### Train a calibrated model and select a threshold

```bash
rangeshift calibrate data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --calibration-method sigmoid \
  --threshold-method tss \
  --output calibrated_model.joblib \
  --threshold-output threshold_diagnostics.csv \
  --calibration-output calibration_diagnostics.csv \
  --summary-output calibration_summary.json
```

The selected threshold is chosen from the validation partition and then evaluated on an untouched final test partition.

### Select a threshold from existing validation predictions

If a CSV already contains observed labels and validation probabilities:

```bash
rangeshift select-threshold validation_predictions.csv \
  --target presence \
  --probability suitability \
  --method tss \
  --output threshold_diagnostics.csv
```

### Compare random and spatial evaluation

```bash
rangeshift compare-spatial data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --block-size 1.0 \
  --output spatial_comparison.json
```

### Predict current and future suitability

```bash
rangeshift predict-raster calibrated_model.joblib \
  --layer bio1=current/bio1.tif \
  --layer bio12=current/bio12.tif \
  --layer elevation=current/elevation.tif \
  --output current_suitability.tif

rangeshift predict-raster calibrated_model.joblib \
  --layer bio1=future/bio1.tif \
  --layer bio12=future/bio12.tif \
  --layer elevation=future/elevation.tif \
  --output future_suitability.tif
```

### Quantify the range shift

Pass the selected threshold explicitly:

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

### Render the range-shift map

```bash
rangeshift plot-range-shift range_shift_classes.tif \
  --title "Projected habitat range shift" \
  --output range_shift_map.png \
  --dpi 300
```

## Area and centroid calculations

RangeShift does not treat map degrees as physical distance.

- For **projected CRSs**, cell area is calculated from the affine transform and CRS linear units.
- For **geographic CRSs**, each cell area is calculated geodesically on the ellipsoid.
- Suitable-range centroids are area-weighted and calculated geographically.
- Centroid movement is reported as geodesic distance in kilometers and forward bearing in degrees.

## Python example

```python
from rangeshift.calibration import train_calibrated_habitat_model

result = train_calibrated_habitat_model(
    frame,
    feature_columns=["bio1", "bio12", "elevation"],
    calibration_method="sigmoid",
    threshold_method="tss",
)

print(result.selected_threshold)
print(result.test_probability_metrics)
print(result.test_classification_metrics)
```

## Runnable demonstrations

- [`examples/train_demo.py`](examples/train_demo.py) — baseline supervised learning.
- [`examples/spatial_demo.py`](examples/spatial_demo.py) — random vs. spatial evaluation.
- [`examples/raster_demo.py`](examples/raster_demo.py) — raster prediction and suitability map.
- [`examples/range_shift_demo.py`](examples/range_shift_demo.py) — synthetic current-to-future range shift.
- [`examples/calibration_demo.py`](examples/calibration_demo.py) — calibration and validation-based threshold selection.

## Scientific interpretation

A predicted range shift is a **scenario-conditioned suitability projection**, not a guaranteed future distribution. The result depends on the fitted model, environmental predictors, scenario, threshold choice, sampling design, and transferability of environment–occurrence relationships.

Calibration improves interpretation of model probabilities only when it is supported by the data; RangeShift therefore reports before/after probability metrics rather than assuming calibration is beneficial. Likewise, threshold choice is treated as a model decision that must be selected on validation data and disclosed.

Realized ranges may also be constrained by dispersal, barriers, biotic interactions, demography, adaptation, land-use change, detectability, and environmental novelty.

## Design principles

1. **Reproducibility** — identical inputs and seeds should reproduce baseline results.
2. **Scientific transparency** — assumptions, calibration choices, and thresholds remain visible.
3. **Software quality** — modular, tested code instead of one monolithic notebook.
4. **Interpretability** — performance, spatial transfer, probabilities, and range-change outputs are inspectable.

## Author

**Pauline Owusu-Ansah**  
Ph.D. researcher in computational and evolutionary biology

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for the full license text.
