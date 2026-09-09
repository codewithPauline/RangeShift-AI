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
- prediction of continuous suitability probabilities;
- command-line interface;
- automated unit tests;
- GitHub Actions CI.

## v0.2 — Spatial intelligence

**Goal:** test whether model performance survives geographic separation between training and evaluation data.

Implemented:

- latitude/longitude input validation;
- geographic spatial blocks;
- complete held-out block evaluation;
- random-vs-spatial performance comparison;
- repeated spatial cross-validation;
- WGS84 GeoPandas conversion;
- local UTM estimation;
- projected kilometer-scale spatial blocks;
- user-specified projected EPSG support;
- automated core and geospatial tests.

Why it matters:

Spatial autocorrelation can make a species-distribution model appear more accurate when geographically nearby observations are split randomly between training and testing. Spatially separated evaluation makes that potential optimism visible.

Remaining spatial improvements:

1. Sampling-bias and spatial clustering diagnostics.
2. Maps showing train/test blocks and observations.
3. More general blocking strategies for continental and global studies.

## v0.3 — Raster suitability prediction and mapping

**Goal:** project a trained habitat-suitability model across an environmental raster grid and render the result clearly.

Implemented:

- one single-band raster per trained predictor;
- exact predictor-name matching;
- strict dimensions, CRS, and affine-transform checks;
- nodata and non-finite masking;
- cell-wise suitability probability prediction;
- compressed `float32` GeoTIFF export;
- `predict-raster` command;
- fixed 0–1 high-resolution suitability maps;
- CRS-aware map axes;
- optional vector-boundary overlay;
- `plot-raster` command;
- synthetic end-to-end raster demonstration;
- automated raster and map-rendering tests.

Scientific design choice:

RangeShift does not silently crop, reproject, or resample mismatched environmental rasters. Those preprocessing choices remain explicit.

Current limitation:

The raster engine currently loads full predictor stacks into memory. Windowed processing is still needed for very large continental or global grids.

## v0.4 — Current-to-future range-shift analysis

**Goal:** quantify how threshold-defined suitable habitat changes between aligned current and future suitability surfaces.

Implemented:

- aligned current/future suitability-raster validation;
- explicit required suitability threshold;
- stable unsuitable, lost, gained, and stable suitable habitat classes;
- compressed `uint8` transition GeoTIFF;
- continuous future-minus-current suitability GeoTIFF;
- projected-CRS cell area using linear-unit conversion;
- geodesic cell-area calculation for geographic CRSs;
- current and future suitable-area estimates;
- gained, lost, stable, and net range-area change;
- percent area change relative to current suitability;
- area-weighted Jaccard overlap;
- area-weighted geographic centroids;
- geodesic centroid shift distance;
- centroid shift bearing;
- JSON summary output;
- discrete four-class range-shift map;
- `range-shift` and `plot-range-shift` CLI workflows;
- fully synthetic current-to-future demonstration;
- automated calculation and rendering tests.

Scientific design choices:

1. RangeShift does not silently interpret probability `0.5` as suitable habitat. A threshold must be supplied explicitly.
2. The continuous future-minus-current suitability surface is retained alongside threshold-based transition classes.
3. Geographic raster cells are not treated as equal-area; their physical areas are calculated geodesically.
4. Centroid movement describes the center of modeled suitable area, not organismal dispersal distance.

## v0.5 — Better ML, thresholds, and explainability

**Goal:** make model selection and threshold choice more defensible and interpretable.

Planned:

- threshold-selection methods based on validation data;
- sensitivity/specificity trade-off reporting;
- probability calibration;
- hyperparameter tuning;
- gradient-boosted tree comparison;
- SHAP interpretation;
- partial dependence / response curves;
- model comparison reports.

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
- trained and spatially evaluated model;
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
