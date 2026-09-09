# RangeShift AI

> **Released geospatial machine-learning software for habitat suitability modeling and transparent current-to-future range-shift analysis.**

[![Latest Release](https://img.shields.io/github/v/release/codewithPauline/RangeShift-AI?label=Latest%20Release&sort=semver)](https://github.com/codewithPauline/RangeShift-AI/releases/latest)
[![CI](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Released · CI tested · Installable · MIT licensed**  
**Latest release:** [`v0.7.0`](https://github.com/codewithPauline/RangeShift-AI/releases/tag/v0.7.0) — download the tested Python wheel or source distribution from the release page.

## Why RangeShift AI?

Species ranges are not static. Climate, topography, land use, and other environmental pressures can change where suitable habitat exists. RangeShift AI is a reproducible Python toolkit that connects species occurrence data, ecological safeguards, environmental predictors, geospatial machine learning, spatial validation, probability calibration, raster projection, model interpretation, and future range-change analysis.

The project is deliberately designed so ecological assumptions remain visible. RangeShift does not hide background-point design, spatial thinning, predictor collinearity, spatial validation, threshold choice, probability calibration, raster alignment, environmental novelty, dispersal assumptions, or range-change definitions inside a black box.

## Project status

**v0.7.0 — Phases 1–6 of the public roadmap are complete and covered by automated validation.**

RangeShift now supports an end-to-end workflow from occurrence/environmental data to ecologically qualified, reproducible range-shift outputs:

```text
Occurrence records
       ↓
Background / pseudo-absence strategy
       ↓
Spatial thinning + sampling-bias diagnostics
       ↓
Environmental collinearity diagnostics
       ↓
Spatial validation and block-aware model selection
       ↓
Random Forest ↔ Gradient Boosting comparison
       ↓
Probability calibration + validation-only threshold selection
       ↓
SHAP + partial-dependence interpretation
       ↓
Environmental extrapolation + multivariate novelty warnings
       ↓
Current and future aligned environmental rasters
       ↓
Current / future habitat-suitability probabilities
       ↓
Optional explicit dispersal-distance constraint
       ↓
Stable / Lost / Gained / Stable-suitable habitat
       ↓
Area + overlap + centroid shift
       ↓
Run manifest + GeoTIFFs + CSV/JSON + maps
       ↓
CLI workflows / packaged Streamlit explorer
```

## Current capabilities

### Data preparation and ecological safeguards

- validate binary presence/background training tables;
- validate latitude/longitude coordinates;
- generate reproducible equal-area random background points;
- generate spatially stratified background points;
- sample background points from user-provided candidate/target-group pools;
- enforce an explicit minimum distance between generated backgrounds and presences;
- thin clustered occurrence records by minimum geodesic distance;
- optionally prioritize higher-quality records during spatial thinning;
- diagnose Pearson correlations among environmental predictors;
- calculate variance-inflation factors (VIF);
- report high-correlation predictor pairs without silently dropping variables;
- quantify duplicate coordinates and spatial sampling concentration;
- report nearest-neighbor distance distributions;
- provide a reproducible real ecological example using GBIF occurrences and WorldClim 2.1 predictors.

### Spatial validation

- degree-based geographic blocking;
- projected kilometer-scale blocks;
- local UTM estimation and user-specified projected EPSG support;
- random-vs-spatial holdout comparison;
- repeated spatial cross-validation;
- grouped model tuning that keeps spatial groups together;
- explicit validation that both target classes occur in every grouped model-selection fold;
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
- univariate training-envelope extrapolation diagnostics;
- multivariate standardized nearest-neighbor novelty diagnostics;
- a training-derived environmental-distance warning threshold;
- row-level `novel_climate_warning` output that combines envelope and multivariate novelty evidence.

### Raster prediction and mapping

- strict predictor-name matching between models and rasters;
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
- optional maximum-distance dispersal constraint;
- separate accessible and beyond-distance future-suitable classes;
- dispersal-constrained future suitability surface;
- JSON summary output and discrete transition maps.

### Reproducibility and productization

- JSON configuration-driven current-to-future analyses;
- SHA-256 hash of the normalized configuration;
- machine-readable run manifest containing model choices, threshold, diagnostics, assumptions, and output paths;
- optional spatial validation inside the configuration runner;
- optional dispersal constraints inside the configuration runner;
- `rangeshift` core modeling CLI;
- `rangeshift-eco` ecological-safeguard CLI;
- `rangeshift-run` configuration-driven workflow CLI;
- packaged `rangeshift-app` Streamlit explorer;
- wheel and source-distribution builds validated in CI;
- clean-wheel installation test in CI;
- automated GitHub Release workflow for versioned distribution artifacts.

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

### Phase 6 — Ecological safeguards and productization ✅
- [x] Pseudo-absence/background generation strategies
- [x] Spatial thinning
- [x] Environmental collinearity diagnostics
- [x] Multivariate novel-climate warnings
- [x] Dispersal constraints
- [x] Reproducible configuration files and run manifests
- [x] Interactive visualization interface
- [x] Packaged release workflow and validated distributions

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

Interactive explorer support:

```bash
pip install -e ".[geo,app]"
```

Full user environment:

```bash
pip install -e ".[geo,explain,app]"
```

Development environment:

```bash
pip install -e ".[dev,geo,explain,app]"
```

Versioned wheel and source-distribution artifacts are attached to GitHub Releases. RangeShift is **not currently claimed as a PyPI release**.

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

## Ecological safeguard examples

### Generate background points

```bash
rangeshift-eco background presences.csv \
  --n 500 \
  --method spatial_stratified \
  --min-distance-km 5 \
  --strata-size-degrees 1.0 \
  --output background_points.csv
```

Supported strategies are `random`, `spatial_stratified`, and `candidate_pool`. The candidate-pool strategy is intended for user-supplied effort or target-group coordinates.

### Spatially thin occurrences

```bash
rangeshift-eco thin presences.csv \
  --min-distance-km 5 \
  --priority record_quality \
  --output thinned_occurrences.csv
```

RangeShift reports the retained and removed counts; thinning is not presented as universally required or universally optimal.

### Diagnose predictor collinearity

```bash
rangeshift-eco collinearity training.csv \
  --features bio1 bio12 elevation \
  --correlation-threshold 0.7 \
  --vif-threshold 5
```

RangeShift reports correlations and VIF values. It does **not** silently delete a predictor on the user's behalf.

### Diagnose multivariate environmental novelty

```bash
rangeshift-eco novel-climate training_environment.csv future_environment.csv \
  --features bio1 bio12 elevation \
  --distance-quantile 0.99
```

This combines univariate training-envelope violations with standardized multivariate distance to the nearest training environment. It is a transparent novelty warning, not a claim to reproduce a particular proprietary or black-box MESS implementation.

### Apply an explicit dispersal constraint

```bash
rangeshift-eco dispersal \
  current_suitability.tif \
  future_suitability.tif \
  --threshold 0.55 \
  --max-distance-km 100 \
  --accessibility-output dispersal_accessibility.tif \
  --constrained-output future_suitability_constrained.tif
```

The distance constraint answers a limited question: which future suitable cells fall within the stated maximum distance of currently suitable habitat? It does not guarantee colonization or estimate species-specific dispersal unless the user supplies a biologically justified distance.

## Core command-line examples

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

### Predict a large raster in bounded-memory windows

```bash
rangeshift predict-raster selected_model.joblib \
  --layer bio1=current/bio1.tif \
  --layer bio12=current/bio12.tif \
  --layer elevation=current/elevation.tif \
  --window-size 512 \
  --output current_suitability.tif
```

RangeShift requires predictor rasters to be aligned before prediction. It deliberately does **not** silently crop, reproject, or resample ecological predictors.

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

## Reproducible configuration workflow

An example configuration is provided at [`examples/config/run_config.example.json`](examples/config/run_config.example.json).

```bash
rangeshift-run examples/config/run_config.example.json
```

A configured run can train/calibrate the model, perform optional spatial validation, project current and future rasters, optionally impose a dispersal-distance constraint, quantify the range shift, and write a `run_manifest.json` containing a SHA-256 hash of the normalized configuration plus key model decisions and output locations.

## Interactive RangeShift Explorer

Install the `geo` and `app` extras, then launch the packaged interface:

```bash
rangeshift-app
```

The Streamlit explorer can display suitability GeoTIFFs, four-class range-shift GeoTIFFs, `range_shift_summary.json`, and configuration run manifests. It is deliberately an **output explorer**, not a one-click modeling black box.

## Scientific design decisions

### Background and pseudo-absence design

Background selection can materially alter species-distribution models. RangeShift therefore exposes multiple reproducible strategies and records the chosen method rather than presenting one sampler as universally correct.

### Spatial thinning and sampling bias

Spatial clustering can overrepresent heavily sampled localities. RangeShift can diagnose clustering and perform explicit minimum-distance thinning, but both are user-visible decisions rather than hidden preprocessing.

### Predictor collinearity

Highly redundant environmental predictors can destabilize interpretation and inflate variable-importance narratives. RangeShift reports correlation and VIF diagnostics without automatically deciding which ecological variable should be removed.

### Spatial validation

Random train/test splits can overstate performance when nearby records share environmental and spatial structure. RangeShift therefore supports complete spatial-block holdouts, repeated spatial validation, and spatial-group model tuning.

### Thresholds

RangeShift never silently assumes `0.5` defines suitable habitat. Threshold selection is an explicit validation-stage decision using TSS, Youden J, F1, or balanced accuracy.

### Calibration

Calibration is not assumed to improve every dataset. RangeShift reports calibrated and uncalibrated probability metrics so the effect can be inspected rather than presumed.

### Environmental novelty

A model can output a probability in an environmental regime it never encountered during training. RangeShift exposes both univariate extrapolation and multivariate novelty warnings before those projections are interpreted biologically.

### Dispersal

Climatically suitable habitat is not automatically reachable habitat. RangeShift keeps the maximum-distance dispersal assumption explicit and outputs both accessible and beyond-distance suitable cells.

### Raster area

Map degrees are not treated as physical distance. Projected rasters use CRS linear units, while geographic raster cells use ellipsoidal geodesic area calculations.

## Scientific interpretation

A predicted range shift is a **scenario-conditioned habitat-suitability projection**, not a guaranteed future distribution. Results depend on occurrence sampling, background/absence design, thinning choices, environmental predictors, model choice, validation design, calibration, threshold choice, future scenario, environmental novelty, and transferability of modeled environment–occurrence relationships.

Realized distributions may also be constrained by dispersal, barriers, biotic interactions, demography, adaptation, detectability, land-use change, and future-scenario uncertainty. Those limitations are part of the analysis, not footnotes to hide.

## Automated quality checks

GitHub Actions validates:

- Python 3.10;
- Python 3.11;
- Python 3.12;
- Ruff linting;
- geospatial extras including Rasterio, GeoPandas, PyProj, Matplotlib, dispersal, and configured-run workflows;
- explainability extras including SHAP;
- the packaged Streamlit interface;
- wheel and source-distribution builds;
- Twine package metadata checks;
- clean installation of the built wheel.

A separate release workflow builds the distributions again and attaches them to a versioned GitHub Release when the package version changes on `main`.

## Runnable demonstrations

- [`examples/train_demo.py`](examples/train_demo.py) — baseline supervised learning.
- [`examples/spatial_demo.py`](examples/spatial_demo.py) — random vs spatial evaluation.
- [`examples/raster_demo.py`](examples/raster_demo.py) — raster suitability prediction.
- [`examples/range_shift_demo.py`](examples/range_shift_demo.py) — synthetic current-to-future range shift.
- [`examples/calibration_demo.py`](examples/calibration_demo.py) — calibration and validation-based threshold selection.
- [`examples/real_ecology/`](examples/real_ecology/README.md) — real GBIF + WorldClim ecological data preparation.
- [`examples/config/run_config.example.json`](examples/config/run_config.example.json) — configuration-driven reproducible workflow.

## Next milestone — toward v1.0

The six-phase public roadmap is complete. The next work is **hardening rather than checkbox expansion**: uncertainty summaries across resamples/models/scenarios, batch climate-model/SSP comparison, richer provenance embedded in outputs, additional ecological case studies, performance profiling, API stabilization, documentation hardening, and broader external validation before a v1.0 release.

## Design principles

1. **Reproducibility** — inputs, seeds, assumptions, configurations, and provenance should be inspectable.
2. **Scientific transparency** — preprocessing, spatial validation, calibration, thresholds, novelty, and dispersal stay visible.
3. **Software quality** — modular functions, CLIs, tests, CI, and installable packages instead of a monolithic notebook.
4. **Interpretability** — performance, feature effects, probabilities, transfer warnings, and range-change outputs should be explainable.
5. **Explicit geospatial processing** — RangeShift validates alignment assumptions instead of silently altering environmental data.

## Author

**Pauline Owusu-Ansah**  
Ph.D. researcher in computational and evolutionary biology

## License

RangeShift AI is released under the **MIT License**. See [LICENSE](LICENSE) for the full license text.
