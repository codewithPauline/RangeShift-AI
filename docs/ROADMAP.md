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

Core concepts:

- features vs. target;
- training vs. test data;
- classification probability vs. class label;
- why accuracy alone is insufficient;
- ROC-AUC and class imbalance;
- Random Forest feature importance;
- overfitting.

## v0.2 — Spatial intelligence

**Goal:** test whether model performance survives geographic separation between training and evaluation data.

Implemented:

- latitude/longitude input validation;
- deterministic geographic grid-block assignment;
- group-aware holdout with complete blocks reserved for testing;
- explicit prevention of train/test block overlap;
- repeated attempts to obtain valid two-class spatial partitions;
- random-vs-spatial performance comparison;
- metric deltas showing how performance changes under spatial transfer;
- repeated spatial block cross-validation;
- optional conversion to WGS84 GeoPandas `GeoDataFrame` objects;
- local UTM estimation for regional data;
- projected spatial blocks measured in kilometers;
- user-specified projected EPSG support;
- command-line workflows for comparison, cross-validation, and projected blocks;
- automated core and geospatial tests.

Important interpretation:

Spatial autocorrelation can make a species-distribution model appear more accurate when geographically nearby observations are split randomly between training and testing. Spatially separated evaluation makes that potential optimism visible.

Remaining spatial improvements:

1. Sampling-bias and spatial clustering diagnostics.
2. Maps showing train/test blocks and observations.
3. Additional strategies for studies spanning multiple projection zones or global extents.

## v0.3 — Raster suitability prediction and mapping

**Goal:** project a trained habitat-suitability model across a geographic environmental grid and render the result clearly.

Implemented:

- one single-band environmental raster per trained model predictor;
- exact predictor-name matching against the saved model bundle;
- strict raster alignment checks for dimensions, CRS, and affine transform;
- nodata and non-finite predictor masking;
- conversion of complete raster cells to model predictor rows;
- suitability probability prediction for every valid cell;
- reconstruction of predictions into the original raster grid;
- compressed `float32` GeoTIFF export;
- output band description and metadata tags;
- `predict-raster` command-line workflow;
- high-resolution PNG map rendering with a fixed 0–1 suitability scale;
- CRS-aware coordinate-axis labels;
- optional vector study-area boundary overlay with automatic CRS reprojection;
- `plot-raster` command-line workflow;
- a fully synthetic end-to-end raster and map demonstration;
- Rasterio and Matplotlib included in the optional geospatial installation;
- automated tests for raster prediction and map rendering.

Scientific design choice:

RangeShift does not silently resample, crop, or reproject mismatched environmental layers. Those preprocessing choices can affect model projections and should be explicit and reproducible.

Current implementation limitation:

The v0.3 engine loads the full environmental predictor stack into memory. This is appropriate for moderate regional datasets but should be replaced or supplemented with windowed/chunked processing for very large rasters.

Next raster improvements:

1. Windowed prediction for large raster stacks.
2. Observation and training/testing overlays.
3. Additional publication-layout controls such as scale bars and boundary styling.
4. Raster-stack diagnostics that summarize valid coverage and predictor ranges.

## v0.4 — Current vs. future environments

**Goal:** turn habitat suitability into a range-shift analysis.

Planned:

- accept future environmental raster stacks;
- apply the same fitted model to current and future layers;
- calculate per-cell suitability change;
- classify stable, gained, and lost suitable habitat;
- estimate total area gained/lost;
- calculate current/future suitable-range centroids;
- report shift distance and bearing.

A major methodological issue in this phase will be **threshold selection**: continuous suitability must not be converted into suitable/unsuitable habitat using an arbitrary cutoff without making that decision visible.

## v0.5 — Better ML and explainability

Planned:

- hyperparameter tuning;
- gradient-boosted tree comparison;
- probability calibration;
- threshold-selection methods;
- SHAP interpretation;
- partial dependence/response curves;
- model comparison reports.

## v0.6 — Ecological safeguards

Planned:

- pseudo-absence/background sampling strategies;
- spatial thinning;
- environmental collinearity diagnostics;
- extrapolation/novel-climate warnings;
- multivariate environmental similarity analysis;
- uncertainty summaries;
- optional dispersal constraints.

## v1.0 — Reproducible RangeShift workflow

A v1.0 release should let a user move from occurrence records and environmental layers to a documented range-shift report without hiding the underlying scientific assumptions.

Expected v1.0 outputs:

- validated occurrence/environmental inputs;
- trained and spatially evaluated model;
- current suitability raster;
- future suitability raster;
- gain/loss/stability raster;
- range-area summary;
- centroid shift distance and direction;
- model interpretation;
- reproducible metadata and configuration;
- exportable figures and tables.

## Scientific boundaries

RangeShift AI should never present habitat-suitability projections as guaranteed future distributions. Realized ranges may differ because of dispersal, biotic interactions, adaptation, demographic processes, sampling bias, barriers, land-use change, detectability, and environmental novelty.

Those limitations are part of the model, not footnotes to hide.
