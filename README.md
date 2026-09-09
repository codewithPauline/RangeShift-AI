# RangeShift AI

> Geospatial machine-learning tools for habitat suitability modeling and transparent current-to-future range-shift analysis.

[![CI](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.6.0-informational.svg)](pyproject.toml)

## Why RangeShift AI?

Species ranges are not static. Climate, topography, land use, and other environmental pressures can change where suitable habitat exists. RangeShift AI is a reproducible Python toolkit that connects species occurrence data, environmental predictors, geospatial machine learning, spatial validation, probability calibration, raster projection, model interpretation, and future range-change analysis.

The project is deliberately designed so ecological assumptions remain visible. RangeShift does not hide spatial validation, threshold choice, probability calibration, raster alignment, environmental extrapolation, or range-change definitions inside a black box.

## Project status

**Active development — v0.6.0. Phases 1–5 of the core roadmap are complete and covered by automated tests.**

RangeShift now supports an end-to-end workflow from real occurrence/environmental data to interpretable range-shift outputs:

```text
GBIF / occurrence records + environmental predictors
                     ↓
          Input and coordinate validation
                     ↓
       Sampling-bias diagnostics + spatial blocks
                     ↓
 Random Forest ↔ Gradient Boosting model comparison
                     ↓
     Hyperparameter tuning / spatial-group CV
                     ↓
       Probability calibration + threshold selection
                     ↓
    SHAP + partial-dependence interpretation
                     ↓
        Environmental extrapolation checks
                     ↓
       Current environmental raster stack
                     ↓
        Current suitability probability
                     │
                     ├──────── compare ────────┐
                     │                          │
       Future environmental raster stack       │
                     ↓                          │
        Future suitability probability          │
                     └──────────────────────────┘
                                ↓
                     Explicit selected threshold
                                ↓
              Stable / Lost / Gained / Stable-suitable
                                ↓
                  Area + overlap + centroid shift
                                ↓
             GeoTIFFs + CSV/JSON + high-resolution maps
```

## Current capabilities

### Data and ecological diagnostics

- validate binary presence/background training tables;
- validate latitude/longitude coordinates;
- quantify duplicate coordinates and spatial sampling concentration;
- report nearest-neighbor distance distributions;
- provide a reproducible real ecological example using GBIF occurrences and WorldClim 2.1 predictors;
- preserve provenance information instead of committing large downloaded climate files.

### Spatial validation

- degree-based geographic blocking;
- projected kilometer-scale blocks;
- local UTM estimation and user-specified projected EPSG support;
- random-vs-spatial holdout comparison;
- repeated spatial cross-validation;
- grouped model tuning that keeps spatial groups together;
- explicit validation that both target classes occur in every model-selection fold;
- high-resolution train/test spatial split visualization.

### Machine learning and model selection

- class-balanced Random Forest baseline;
- Gradient Boosting comparison model;
- grid-search hyperparameter tuning;
- ROC-AUC, accuracy, precision, recall, F1, and balanced-accuracy evaluation;
- feature-importance reporting;
- common saved-model bundle contract across supported estimators.

### Calibration and threshold selection

- separate training, validation, and final test partitions;
- sigmoid or isotonic probability calibration;
- TSS / Youden J threshold selection;
- F1 and balanced-accuracy threshold selection;
- sensitivity, specificity, precision, recall, F1, and predicted-positive-rate diagnostics across candidate thresholds;
- Brier score, log loss, and ROC-AUC for calibrated and uncalibrated probabilities;
- threshold selection on validation data only, never the final test set.

### Explainability and environmental transfer

- one-dimensional partial-dependence response curves;
- optional Tree SHAP feature attribution for supported fitted tree models;
- training-envelope environmental extrapolation diagnostics;
- row-level counts/fractions of predictors outside the training range;
- feature-level summaries of projection values beyond the observed training envelope.

### Raster prediction and mapping

- strict predictor-name matching between model and rasters;
- dimensions, CRS, affine-transform, nodata, and alignment validation;
- continuous 0–1 suitability prediction across raster grids;
- in-memory prediction for moderate grids;
- bounded-memory windowed prediction for large rasters;
- nodata propagation;
- compressed `float32` GeoTIFF export;
- high-resolution suitability maps and optional vector-boundary overlays.

### Current-to-future range shifts

- aligned current/future suitability comparison;
- explicit threshold requirement;
- stable unsuitable, lost, gained, and stable suitable classes;
- continuous future-minus-current suitability raster;
- physically meaningful area calculations for projected and geographic CRSs;
- current/future suitable area, gained area, lost area, stable area, and net change;
- Jaccard overlap;
- area-weighted geographic centroids;
- centroid shift distance and bearing;
- JSON summary output and discrete transition maps.

## Roadmap

### Phase 1 — Current habitat suitability ✅
- [x] Project architecture
- [x] Random Forest baseline
- [x] Model evaluation
- [x] Feature importance
- [x] Model persistence and prediction
- [x] CLI entry point
- [x] Automated tests and CI
- [x] Reproducible real ecological example using GBIF + WorldClim

### Phase 2 — Spatial intelligence ✅
- [x] Coordinate validation
- [x] Degree-based spatial blocks
- [x] Spatially separated train/test evaluation
- [x] Random-vs-spatial performance comparison
- [x] Repeated spatial cross-validation
- [x] GeoPandas integration
- [x] Projected spatial blocks measured in kilometers
- [x] Sampling-bias diagnostics
- [x] Train/test block visualization

### Phase 3 — Environmental rasters and mapping ✅
- [x] Read aligned raster predictors
- [x] Validate dimensions, CRS, transform, nodata, and feature names
- [x] Predict suitability across a raster grid
- [x] Export suitability GeoTIFFs
- [x] Render high-resolution suitability maps
- [x] Optional vector boundary overlay
- [x] Windowed prediction for very large rasters

### Phase 4 — Future range-shift projection ✅
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

### Phase 5 — Explainable and robust ML ✅
- [x] Threshold diagnostics across candidate cutoffs
- [x] TSS / Youden J threshold selection
- [x] F1 and balanced-accuracy threshold selection
- [x] Separate train / validation / test workflow
- [x] Sigmoid and isotonic probability calibration
- [x] Brier score and log-loss reporting
- [x] Calibration diagnostic table
- [x] Hyperparameter tuning
- [x] Gradient-boosted comparison model
- [x] SHAP-based interpretation
- [x] Partial dependence / response curves
- [x] Environmental extrapolation diagnostics

### Phase 6 — Ecological safeguards and productization
- [ ] Pseudo-absence/background generation strategies
- [ ] Spatial thinning
- [ ] Environmental collinearity diagnostics
- [ ] Multivariate novel-climate / MESS-style warnings
- [ ] Dispersal constraints
- [ ] Reproducible configuration files
- [ ] Interactive visualization interface
- [ ] Packaged release

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the detailed scientific and development history.

## Installation

Clone the repository and install the lightweight ML core:

```bash
git clone https://github.com/codewithPauline/RangeShift-AI.git
cd RangeShift-AI
python -m venv .venv
pip install -e .
```

Geospatial support:

```bash
pip install -e ".[geo]"
```

Optional SHAP support:

```bash
pip install -e ".[explain]"
```

Full geospatial + explainability environment:

```bash
pip install -e ".[geo,explain]"
```

Development environment:

```bash
pip install -e ".[dev,geo,explain]"
```

## Real ecological example

A real-data preparation workflow is included at [`examples/real_ecology/`](examples/real_ecology/README.md).

By default it uses the spotted salamander, *Ambystoma maculatum*, and:

1. resolves the species using GBIF's current species matcher;
2. retrieves georeferenced GBIF presence records;
3. downloads WorldClim 2.1 bioclimatic layers;
4. extracts BIO1, BIO12, and BIO15 at occurrence locations;
5. creates climate-valid background points within the observed study extent;
6. writes a RangeShift-ready training CSV and provenance JSON.

```bash
python examples/real_ecology/prepare_gbif_worldclim.py \
  --species "Ambystoma maculatum" \
  --country US \
  --max-records 500 \
  --background 500 \
  --output-dir real_ecology_output
```

Downloaded data are ignored by Git. The script is the reproducible artifact; large climate rasters and changing API search results are not frozen into repository history.

> **Research-use note:** the example is designed for transparent demonstration, not as a universal publication-ready SDM protocol. Publication work should use a citable GBIF download DOI and justify taxonomic filters, background sampling, spatial thinning, predictor selection, and validation design.

## Expected tabular input

```csv
presence,bio1,bio12,elevation,latitude,longitude
1,13.2,1050,310,39.51,-84.74
1,14.1,980,270,39.72,-84.10
0,18.9,620,120,37.98,-86.12
0,20.4,540,80,36.43,-85.51
```

`presence` must contain `1` for observed/presence records and `0` for absence or background records.

## Command-line examples

### Diagnose spatial sampling bias

```bash
rangeshift diagnose-bias data.csv \
  --block-size 1.0 \
  --summary-output sampling_bias_summary.json
```

### Compare random and spatial evaluation and render the split

```bash
rangeshift compare-spatial data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --block-size 1.0 \
  --output spatial_comparison.json \
  --plot-output spatial_holdout.png
```

### Tune Random Forest vs Gradient Boosting

```bash
rangeshift tune-models data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --spatial-groups \
  --block-size 1.0 \
  --scoring roc_auc \
  --output selected_model.joblib
```

When `--spatial-groups` is used, entire geographic blocks stay together during tuning. RangeShift rejects a grouped split if any train/test fold lacks one of the target classes.

### Calibrate probabilities and select a threshold

```bash
rangeshift calibrate data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --calibration-method sigmoid \
  --threshold-method tss \
  --output calibrated_model.joblib
```

Thresholds are selected from the validation partition and evaluated only afterward on the untouched final test partition.

### Partial-dependence response curves

```bash
rangeshift response-curves selected_model.joblib data.csv \
  --table-output partial_dependence.csv \
  --plot-output response_curves.png
```

### Tree SHAP importance

```bash
rangeshift shap-importance selected_model.joblib data.csv \
  --max-samples 500 \
  --output shap_importance.csv
```

Tree SHAP is an optional extra and is intended for supported fitted tree estimators such as RangeShift's Random Forest and Gradient Boosting models.

### Diagnose environmental extrapolation

```bash
rangeshift diagnose-extrapolation training_environment.csv future_environment.csv \
  --features bio1 bio12 elevation \
  --row-output extrapolation_rows.csv \
  --feature-output extrapolation_features.csv
```

The current diagnostic is deliberately transparent: it identifies predictor values outside the observed univariate training envelope. It is not presented as a full multivariate MESS analysis.

### Predict a large raster in bounded-memory windows

```bash
rangeshift predict-raster selected_model.joblib \
  --layer bio1=current/bio1.tif \
  --layer bio12=current/bio12.tif \
  --layer elevation=current/elevation.tif \
  --window-size 512 \
  --output current_suitability.tif
```

RangeShift still requires predictor rasters to be aligned before prediction. It deliberately does **not** silently crop, reproject, or resample ecological predictors.

### Quantify current-to-future range change

```bash
rangeshift range-shift \
  current_suitability.tif \
  future_suitability.tif \
  --threshold 0.55 \
  --classes-output range_shift_classes.tif \
  --difference-output suitability_change.tif \
  --summary-output range_shift_summary.json
```

The transition raster uses:

| Code | Interpretation |
| ---: | --- |
| 0 | Stable unsuitable |
| 1 | Lost suitable habitat |
| 2 | Gained suitable habitat |
| 3 | Stable suitable habitat |
| 255 | Nodata |

## Scientific design decisions

### Spatial validation

Random train/test splits can overstate performance when nearby records share environmental and spatial structure. RangeShift therefore supports complete spatial-block holdouts, repeated spatial validation, and spatial-group model tuning.

### Thresholds

RangeShift never silently assumes `0.5` defines suitable habitat. Threshold selection is an explicit validation-stage decision using TSS, Youden J, F1, or balanced accuracy.

### Calibration

Calibration is not assumed to improve every dataset. RangeShift reports calibrated and uncalibrated probability metrics so the effect can be inspected rather than presumed.

### Raster area

Map degrees are not treated as physical distance. Projected rasters use CRS linear units, while geographic raster cells use ellipsoidal geodesic area calculations.

### Extrapolation

A model can output a probability in an environmental regime it never encountered during training. RangeShift therefore exposes predictor-envelope extrapolation before those projections are interpreted biologically.

## Scientific interpretation

A predicted range shift is a **scenario-conditioned habitat-suitability projection**, not a guaranteed future distribution. Results depend on occurrence sampling, background/absence design, environmental predictors, model choice, validation design, calibration, threshold choice, future scenario, and transferability of modeled environment–occurrence relationships.

Realized distributions may also be constrained by dispersal, barriers, biotic interactions, demography, adaptation, detectability, land-use change, and environmental novelty. Those limitations are part of the analysis, not footnotes to hide.

## Automated quality checks

GitHub Actions tests:

- Python 3.10;
- Python 3.11;
- Python 3.12;
- geospatial extras including Rasterio/GeoPandas/PyProj/Matplotlib workflows;
- explainability extras including SHAP.

Ruff linting runs on the core Python matrix before tests.

## Runnable demonstrations

- [`examples/train_demo.py`](examples/train_demo.py) — baseline supervised learning.
- [`examples/spatial_demo.py`](examples/spatial_demo.py) — random vs spatial evaluation.
- [`examples/raster_demo.py`](examples/raster_demo.py) — raster suitability prediction.
- [`examples/range_shift_demo.py`](examples/range_shift_demo.py) — synthetic current-to-future range shift.
- [`examples/calibration_demo.py`](examples/calibration_demo.py) — calibration and validation-based threshold selection.
- [`examples/real_ecology/`](examples/real_ecology/README.md) — real GBIF + WorldClim ecological data preparation.

## Design principles

1. **Reproducibility** — inputs, seeds, assumptions, and provenance should be inspectable.
2. **Scientific transparency** — spatial validation, calibration, thresholds, and extrapolation stay visible.
3. **Software quality** — modular functions, CLI workflows, tests, and CI instead of a monolithic notebook.
4. **Interpretability** — performance, feature effects, probabilities, and range-change outputs should be explainable.
5. **Explicit preprocessing** — RangeShift validates geospatial assumptions instead of silently altering data.

## Author

**Pauline Owusu-Ansah**  
Ph.D. researcher in computational and evolutionary biology

## License

RangeShift AI is released under the **MIT License**. See [LICENSE](LICENSE) for the full license text.
