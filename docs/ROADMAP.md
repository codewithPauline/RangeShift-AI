# RangeShift AI Development Roadmap

RangeShift AI is being developed as scientific software, not as a collection of disconnected notebooks. Each milestone adds a testable capability while preserving explicit ecological assumptions and reproducible outputs.

## Core roadmap status

| Phase | Focus | Status |
| --- | --- | --- |
| Phase 1 | Current habitat suitability | Complete ✅ |
| Phase 2 | Spatial intelligence | Complete ✅ |
| Phase 3 | Environmental rasters and mapping | Complete ✅ |
| Phase 4 | Future range-shift projection | Complete ✅ |
| Phase 5 | Explainable and robust ML | Complete ✅ |
| Phase 6 | Ecological safeguards and productization | Planned / active next |

## v0.1 — Baseline habitat suitability

**Goal:** implement the complete supervised-learning loop using tabular environmental predictors.

Implemented:

- CSV input and validation;
- binary presence/background target;
- reproducible train/test split;
- class-balanced Random Forest;
- ROC-AUC, accuracy, precision, recall, and F1;
- feature importance;
- model persistence with `joblib`;
- continuous suitability probabilities;
- command-line interface;
- automated tests and GitHub Actions CI.

## v0.2 — Spatial intelligence

**Goal:** test whether apparent model performance survives geographic separation between training and evaluation data.

Implemented:

- latitude/longitude validation;
- geographic spatial blocks;
- complete held-out block evaluation;
- random-vs-spatial performance comparison;
- repeated spatial cross-validation;
- GeoPandas conversion;
- local UTM estimation;
- projected kilometer-scale spatial blocks;
- user-specified projected EPSG support.

Extended in v0.6:

- duplicate-coordinate fraction;
- nearest-neighbor distance diagnostics;
- spatial-block concentration metrics;
- effective-number-of-blocks diversity diagnostic;
- train/test spatial split visualization;
- grouped hyperparameter tuning with prevalidated two-class folds.

Why it matters:

Spatial autocorrelation can make a species-distribution model appear more accurate when nearby observations are divided randomly between training and testing. RangeShift makes that potential optimism visible and keeps complete spatial groups together when requested.

## v0.3 — Raster suitability prediction and mapping

**Goal:** project a trained habitat-suitability model across aligned environmental raster grids.

Implemented:

- one single-band raster per trained predictor;
- exact model/raster predictor-name matching;
- dimensions, CRS, affine-transform, and nodata checks;
- strict grid-alignment validation;
- complete-case cell masking;
- 0–1 suitability prediction;
- compressed `float32` GeoTIFF export;
- high-resolution suitability maps;
- CRS-aware map axes;
- optional vector-boundary overlay;
- synthetic raster demonstration;
- automated raster and rendering tests.

Extended in v0.6:

- bounded-memory windowed raster prediction;
- configurable raster window size;
- parity tests showing windowed and in-memory engines produce equivalent outputs.

Scientific design choice:

RangeShift does not silently crop, reproject, or resample mismatched environmental rasters. Those preprocessing decisions can alter ecological inference, so users must make them explicitly before model projection.

## v0.4 — Current-to-future range-shift analysis

**Goal:** quantify how threshold-defined suitable habitat changes between aligned current and future suitability surfaces.

Implemented:

- current/future suitability-raster alignment validation;
- explicit required suitability threshold;
- stable unsuitable, lost, gained, and stable suitable classes;
- continuous future-minus-current suitability raster;
- projected-CRS cell area using CRS linear units;
- geodesic cell-area calculation for geographic CRSs;
- current and future suitable-area estimates;
- gained, lost, stable, and net range-area change;
- percentage area change relative to current suitable habitat;
- area-weighted Jaccard overlap;
- area-weighted geographic centroids;
- geodesic centroid shift distance;
- centroid shift bearing;
- JSON summary output;
- discrete range-shift transition maps;
- synthetic end-to-end demonstration;
- automated calculation and rendering tests.

Scientific design choices:

1. Probability `0.5` is never silently treated as the habitat threshold.
2. The continuous suitability-change surface is retained alongside classified transitions.
3. Geographic raster cells are not assumed to have equal physical area.
4. Centroid movement describes movement in modeled suitable area, not organismal dispersal distance.

## v0.5 — Threshold selection and probability calibration

**Goal:** make suitability thresholds and probability interpretation defensible without contaminating final evaluation data.

Implemented:

- separate training / validation / final test partitions;
- Random Forest fitting on training data only;
- cross-validated probability calibration inside the training partition;
- sigmoid calibration;
- isotonic calibration;
- validation-only threshold selection;
- TSS / Youden J selection;
- F1 selection;
- balanced-accuracy selection;
- sensitivity, specificity, precision, recall, F1, accuracy, and positive-rate diagnostics;
- Brier score, log loss, and ROC-AUC probability evaluation;
- calibrated-vs-uncalibrated test probability comparison;
- final threshold-based metrics on an untouched test set;
- calibration diagnostic tables;
- threshold diagnostic tables;
- calibrated model persistence using the standard RangeShift prediction bundle;
- `calibrate` and `select-threshold` CLI workflows;
- synthetic calibration demonstration;
- automated threshold and calibration tests.

Scientific design choices:

1. Thresholds are never selected on the final test set.
2. Calibration is not assumed to improve every model.
3. Selected thresholds remain explicit when used for downstream range classification.
4. Validation and final test performance are reported separately.

## v0.6 — Roadmap consolidation, robust ML, and ecological transfer diagnostics

**Goal:** close the meaningful unfinished items from Phases 1–5 before expanding into productization.

Implemented:

### Real ecological data example

- reproducible GBIF occurrence retrieval;
- current GBIF species-name matching;
- default *Ambystoma maculatum* example;
- WorldClim 2.1 BIO1, BIO12, and BIO15 preparation;
- climate extraction at presence records;
- climate-valid random background sampling;
- provenance JSON output;
- generated downloads and large climate files excluded from Git history;
- explicit publication-use caveats recommending citable GBIF DOI downloads and justified sampling design.

### Sampling-bias diagnostics

- duplicate-coordinate fraction;
- nearest-neighbor distance distribution using haversine distance;
- spatial-block observation counts;
- maximum block concentration;
- block-count coefficient of variation;
- effective-number-of-blocks diagnostic;
- no arbitrary universal pass/fail cutoff.

### Spatial visualization

- train vs held-out observation map;
- optional geographic block grid overlay;
- explicit train/test overlap rejection;
- high-resolution figure export.

### Model tuning and comparison

- Random Forest hyperparameter search;
- Gradient Boosting hyperparameter search;
- common scoring criteria across model families;
- balanced sample weighting for both algorithms;
- ordinary stratified CV or spatially grouped CV;
- grouped-fold prevalidation so every train/test fold contains both classes;
- selected-model persistence through the standard model-bundle contract.

### Explainability

- one-dimensional partial-dependence response tables;
- high-resolution response-curve figures;
- optional Tree SHAP mean-absolute feature attribution;
- dedicated optional SHAP dependency and CI job.

### Environmental extrapolation

- predictor-wise training envelopes;
- row-level counts/fractions of novel predictor values;
- normalized distance beyond the training envelope;
- feature-level fractions below/above/outside the observed training range;
- transparent labeling as a univariate envelope diagnostic rather than a full multivariate MESS implementation.

### Raster scaling

- windowed/chunked raster prediction;
- fixed-size bounded-memory processing;
- strict metadata validation before any window is written;
- equivalent output contract to the original in-memory engine.

## Phase 6 — Ecological safeguards and productization

The next development phase should deepen ecological defensibility rather than merely add more algorithms.

Planned:

1. pseudo-absence/background generation strategies beyond the current demonstration sampler;
2. spatial thinning workflows;
3. environmental collinearity diagnostics and predictor-screening reports;
4. multivariate novel-climate / MESS-style diagnostics;
5. uncertainty summaries across resamples, models, or scenarios;
6. optional dispersal constraints on future suitable habitat;
7. reproducible configuration files and run manifests;
8. batch future-scenario comparisons;
9. interactive visualization interface;
10. packaged public release and release notes.

## Later scale and reproducibility goals

After Phase 6, useful engineering extensions include:

- tiled/cloud-optimized raster workflows;
- provenance metadata embedded in output rasters;
- scenario manifests;
- batch climate-model / SSP comparison;
- exportable report tables and figures;
- stable configuration schema;
- formal semantic-versioned releases.

## v1.0 target

A v1.0 release should let a user move from occurrence records and environmental layers to a documented range-shift report without hiding the underlying scientific assumptions.

Expected v1.0 outputs:

- validated occurrence/environmental inputs;
- sampling-bias and coordinate diagnostics;
- trained and spatially evaluated candidate models;
- documented model selection;
- calibrated probabilities where appropriate;
- documented threshold selection;
- current suitability raster;
- future suitability raster;
- environmental extrapolation diagnostics;
- gain/loss/stability raster;
- continuous suitability-change raster;
- range-area and overlap summary;
- centroid shift distance and direction;
- model interpretation;
- uncertainty and transferability warnings;
- reproducible metadata/configuration;
- exportable figures and tables.

## Scientific boundaries

RangeShift AI must never present habitat-suitability projections as guaranteed future distributions. Realized ranges may differ because of dispersal, biotic interactions, adaptation, demographic processes, sampling bias, barriers, land-use change, detectability, environmental novelty, and future-scenario uncertainty.

A model producing a probability is not evidence that its projection environment is familiar. High apparent predictive accuracy is not automatically evidence of geographic transferability. A classified suitability map is not equivalent to realized occupancy.

Those limitations are part of the model, not footnotes to hide.
