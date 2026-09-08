# RangeShift AI Development Roadmap

RangeShift AI is being developed as both a scientific software project and a practical machine-learning learning path. Each stage adds one clearly testable capability before the project becomes more complex.

## v0.1 — Baseline habitat suitability

**Goal:** learn the complete supervised-learning loop using tabular environmental predictors.

Implemented:

- CSV input and validation
- binary presence/background target
- reproducible train/test split
- class-balanced Random Forest
- ROC-AUC, accuracy, precision, recall, and F1
- feature importance
- model persistence with `joblib`
- prediction of continuous suitability probabilities
- command-line interface
- automated unit tests
- GitHub Actions CI

Core concepts:

- features vs. target
- training vs. test data
- classification probability vs. class label
- why accuracy alone is insufficient
- ROC-AUC and class imbalance
- Random Forest feature importance
- overfitting

## v0.2 — Spatial intelligence baseline

**Goal:** test whether model performance survives geographic separation between training and evaluation data.

Implemented:

- latitude/longitude input validation;
- deterministic spatial grid-block assignment;
- group-aware holdout with complete blocks reserved for testing;
- explicit prevention of train/test block overlap;
- repeated attempts to obtain a spatial split containing both binary target classes;
- random-vs-spatial performance comparison;
- metric deltas showing how performance changes under spatial transfer;
- `compare-spatial` command-line workflow;
- optional conversion to WGS84 GeoPandas `GeoDataFrame` objects;
- automated tests for spatial blocking and evaluation behavior.

Important interpretation:

Spatial autocorrelation can make a species-distribution model appear more accurate when geographically nearby observations are split randomly between training and testing. The v0.2 comparison is designed to make that potential optimism visible.

Current limitation:

The first blocker uses geographic-degree grid cells. These cells are transparent and useful for diagnostics, but they are not equal-area and do not correspond to a constant physical distance across latitudes.

Next spatial improvements:

1. CRS-aware projected blocks measured in kilometers.
2. Automatic or user-specified projected CRS handling.
3. Repeated spatial cross-validation rather than a single valid holdout.
4. Sampling-bias and spatial clustering diagnostics.
5. Maps showing train/test blocks and observations.

## v0.3 — Raster suitability prediction

**Goal:** create the first actual habitat-suitability map.

Planned:

- load aligned environmental raster layers;
- check resolution, extent, CRS, and nodata compatibility;
- convert raster cells to model predictor rows;
- predict suitability probability per cell;
- rebuild predictions as a raster;
- export GeoTIFF;
- create a publication-quality map.

Likely stack: `rasterio`, `numpy`, `geopandas`, `matplotlib`.

## v0.4 — Current vs. future environments

**Goal:** turn habitat suitability into a range-shift analysis.

Planned:

- accept a future environmental raster stack;
- apply the same fitted model to current and future layers;
- calculate per-cell suitability change;
- classify stable, gained, and lost suitable habitat;
- estimate total area gained/lost;
- calculate current/future suitable-range centroids;
- report shift distance and bearing.

## v0.5 — Better ML and explainability

Planned:

- repeated cross-validation;
- hyperparameter tuning;
- gradient-boosted tree comparison;
- probability calibration;
- threshold selection;
- SHAP interpretation;
- partial dependence/response curves;
- model comparison report.

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
- trained and evaluated model;
- current suitability raster;
- future suitability raster;
- gain/loss/stability raster;
- range area summary;
- centroid shift distance and direction;
- model interpretation;
- reproducible metadata and configuration;
- exportable figures and tables.

## Scientific boundaries

RangeShift AI should never present habitat-suitability projections as guaranteed future distributions. Realized ranges may differ because of dispersal, biotic interactions, adaptation, demographic processes, sampling bias, barriers, land-use change, and environmental novelty.

Those limitations are part of the model, not footnotes to hide.
