# RangeShift AI Development Roadmap

RangeShift AI is being developed as both a scientific software project and a practical machine-learning learning path. Each stage should add one clearly testable capability before the project becomes more complex.

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

What to understand before moving on:

- features vs. target
- training vs. test data
- classification probability vs. class label
- why accuracy alone is insufficient
- ROC-AUC and class imbalance
- Random Forest feature importance
- overfitting
- why random train/test splitting can be optimistic for spatial data

## v0.2 — Spatial data foundation

**Goal:** connect the ML model to geography.

Planned:

1. Accept longitude and latitude alongside environmental predictors.
2. Introduce GeoPandas data structures.
3. Validate coordinate reference systems.
4. Visualize presence/background observations.
5. Add spatial train/test splitting or spatial blocking.
6. Compare random-split and spatial-split performance.

The comparison in step 6 is important: spatial autocorrelation can make a species-distribution model appear more accurate than it really is when geographically nearby observations are split randomly between training and testing.

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

- cross-validation;
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
