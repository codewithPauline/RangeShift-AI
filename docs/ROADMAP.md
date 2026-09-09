# RangeShift AI Development Roadmap

RangeShift AI is being developed as both a scientific software project and a practical machine-learning learning path. Each stage adds one clearly testable capability before the project becomes more complex.

## v0.1 — Baseline habitat suitability

**Goal:** learn the complete supervised-learning loop using tabular environmental predictors.

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
- automated tests and CI.

## v0.2 — Spatial intelligence

**Goal:** test whether performance survives geographic separation between training and evaluation data.

Implemented:

- coordinate validation;
- geographic spatial blocks;
- complete held-out block evaluation;
- random-vs-spatial comparison;
- repeated spatial cross-validation;
- GeoPandas conversion;
- local UTM estimation;
- projected kilometer-scale blocks;
- user-specified projected EPSG support;
- automated core and geospatial tests.

Why it matters:

Spatial autocorrelation can make a species-distribution model appear more accurate when nearby observations are split randomly between training and testing. Spatially separated evaluation makes that optimism visible.

## v0.3 — Raster suitability prediction and mapping

**Goal:** project a trained model across environmental raster grids and render the result clearly.

Implemented:

- one single-band raster per trained predictor;
- exact predictor-name matching;
- strict dimensions, CRS, and affine-transform checks;
- nodata and non-finite masking;
- cell-wise suitability probability prediction;
- compressed `float32` GeoTIFF export;
- fixed 0–1 suitability maps;
- CRS-aware axes;
- optional vector-boundary overlay;
- synthetic raster demonstration;
- automated raster and rendering tests.

Scientific design choice:

RangeShift does not silently crop, reproject, or resample mismatched environmental rasters. Those preprocessing choices remain explicit.

## v0.4 — Current-to-future range-shift analysis

**Goal:** quantify how threshold-defined suitable habitat changes between aligned current and future suitability surfaces.

Implemented:

- current/future raster validation;
- explicit required suitability threshold;
- stable unsuitable, lost, gained, and stable suitable classes;
- continuous future-minus-current suitability raster;
- physically meaningful area calculations for projected and geographic CRSs;
- current/future suitable area;
- gained, lost, stable, and net change;
- percent area change;
- Jaccard overlap;
- area-weighted geographic centroids;
- centroid shift distance and bearing;
- JSON summary output;
- discrete range-shift map;
- synthetic end-to-end demonstration;
- automated calculation and rendering tests.

Scientific design choices:

1. RangeShift does not silently interpret probability `0.5` as suitable habitat.
2. The continuous suitability-change surface is retained alongside threshold classes.
3. Geographic cells are not treated as equal-area.
4. Centroid movement describes modeled suitable-area movement, not organismal dispersal.

## v0.5 — Threshold selection and probability calibration

**Goal:** make habitat threshold choice and probability interpretation defensible without contaminating final evaluation data.

Implemented:

- three-way train / validation / test splitting;
- Random Forest fitting on training data only;
- cross-validated probability calibration within the training partition;
- sigmoid calibration;
- isotonic calibration;
- validation-only threshold selection;
- TSS threshold selection;
- Youden J threshold selection;
- F1 threshold selection;
- balanced-accuracy threshold selection;
- sensitivity, specificity, precision, recall, F1, accuracy, and positive-rate diagnostics across candidate thresholds;
- Brier score, log loss, and ROC-AUC for probability evaluation;
- calibrated vs. uncalibrated test probability comparison;
- final threshold-based metrics on an untouched test set;
- calibration diagnostic tables;
- threshold diagnostic tables;
- calibrated model persistence using the existing prediction bundle contract;
- `calibrate` and `select-threshold` CLI workflows;
- synthetic calibration demonstration;
- automated threshold and calibration tests.

Scientific design choices:

1. Thresholds are never selected on the final test set.
2. Calibration is not assumed to improve every model; before/after probability metrics are reported.
3. The selected threshold remains explicit when it is later used for range-shift classification.
4. Validation and final test performance are reported separately.

Next v0.5 improvements:

1. Hyperparameter tuning with nested or leakage-aware evaluation.
2. Gradient-boosted model comparison.
3. SHAP-based interpretation.
4. Partial dependence and response curves.
5. Calibration plots and threshold trade-off plots.

## v0.6 — Ecological safeguards

Planned:

- pseudo-absence/background sampling strategies;
- spatial thinning;
- environmental collinearity diagnostics;
- sampling-bias diagnostics;
- environmental extrapolation and novel-climate warnings;
- uncertainty summaries;
- optional dispersal constraints.

## v0.7 — Scale and reproducibility

Planned:

- windowed/chunked raster prediction;
- configuration-driven workflows;
- provenance metadata for scenarios and predictors;
- reproducible run manifests;
- batch scenario comparison;
- exportable report tables and figures.

## v1.0 — Reproducible RangeShift workflow

A v1.0 release should let a user move from occurrence records and environmental layers to a documented range-shift report without hiding the underlying scientific assumptions.

Expected v1.0 outputs:

- validated occurrence/environmental inputs;
- trained, calibrated, and spatially evaluated model;
- documented threshold selection;
- current suitability raster;
- future suitability raster;
- gain/loss/stability raster;
- continuous suitability-change raster;
- range-area summary;
- centroid shift distance and direction;
- model interpretation;
- uncertainty and extrapolation diagnostics;
- reproducible metadata and configuration;
- exportable figures and tables.

## Scientific boundaries

RangeShift AI should never present habitat-suitability projections as guaranteed future distributions. Realized ranges may differ because of dispersal, biotic interactions, adaptation, demographic processes, sampling bias, barriers, land-use change, detectability, and environmental novelty.

Those limitations are part of the model, not footnotes to hide.
