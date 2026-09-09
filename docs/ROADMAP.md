# RangeShift AI Development Roadmap

RangeShift AI is being developed as scientific software, not as a collection of disconnected notebooks. Each milestone adds a testable capability while keeping ecological assumptions explicit and outputs reproducible.

## Core roadmap status

| Phase | Focus | Status |
| --- | --- | --- |
| Phase 1 | Current habitat suitability | Complete ✅ |
| Phase 2 | Spatial intelligence | Complete ✅ |
| Phase 3 | Environmental rasters and mapping | Complete ✅ |
| Phase 4 | Future range-shift projection | Complete ✅ |
| Phase 5 | Explainable and robust ML | Complete ✅ |
| Phase 6 | Ecological safeguards and productization | Complete ✅ |

The six-phase public roadmap is complete in v0.7.0. Later work is intentionally separated below rather than being retroactively treated as unfinished Phase 6 work.

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
- automated raster and rendering tests.

Extended in v0.6:

- bounded-memory windowed raster prediction;
- configurable raster window size;
- parity tests showing windowed and in-memory engines produce equivalent outputs.

Scientific design choice: RangeShift does not silently crop, reproject, or resample mismatched environmental rasters.

## v0.4 — Current-to-future range-shift analysis

**Goal:** quantify how threshold-defined suitable habitat changes between aligned current and future suitability surfaces.

Implemented:

- current/future suitability-raster alignment validation;
- explicit required suitability threshold;
- stable unsuitable, lost, gained, and stable suitable classes;
- continuous future-minus-current suitability raster;
- projected-CRS and geodesic geographic cell-area calculations;
- current/future suitable-area estimates;
- gained, lost, stable, and net range-area change;
- percentage area change relative to current suitable habitat;
- Jaccard overlap;
- area-weighted geographic centroids;
- centroid shift distance and bearing;
- JSON summary output;
- discrete transition maps.

Scientific design choices:

1. Probability `0.5` is never silently treated as the habitat threshold.
2. Continuous suitability change is retained alongside classified transitions.
3. Geographic raster cells are not assumed to have equal physical area.
4. Centroid movement describes modeled suitable-area movement, not organismal dispersal distance.

## v0.5 — Threshold selection and probability calibration

**Goal:** make suitability thresholds and probability interpretation defensible without contaminating final evaluation data.

Implemented:

- separate training / validation / final test partitions;
- Random Forest fitting on training data only;
- cross-validated calibration inside the training partition;
- sigmoid and isotonic calibration;
- validation-only threshold selection;
- TSS / Youden J, F1, and balanced-accuracy selection;
- sensitivity, specificity, precision, recall, F1, accuracy, and positive-rate diagnostics;
- Brier score, log loss, and ROC-AUC probability evaluation;
- calibrated-vs-uncalibrated test probability comparison;
- final threshold-based metrics on an untouched test set;
- calibrated model persistence;
- CLI workflows and automated tests.

## v0.6 — Roadmap consolidation, robust ML, and ecological transfer diagnostics

**Goal:** close the meaningful unfinished items from Phases 1–5 before expanding into ecological safeguards and productization.

Implemented:

### Real ecological data example

- reproducible GBIF occurrence retrieval;
- current GBIF species-name matching;
- default *Ambystoma maculatum* example;
- WorldClim 2.1 BIO1, BIO12, and BIO15 preparation;
- climate extraction at presence records;
- climate-valid background sampling;
- provenance JSON output;
- explicit publication-use caveats.

### Sampling-bias and spatial diagnostics

- duplicate-coordinate fraction;
- nearest-neighbor distance distribution;
- spatial-block observation counts and concentration metrics;
- effective-number-of-blocks diagnostic;
- train/test spatial split visualization.

### Model tuning and explainability

- Random Forest hyperparameter search;
- Gradient Boosting hyperparameter search;
- ordinary stratified CV or spatially grouped CV;
- grouped-fold prevalidation so every train/test fold contains both classes;
- partial-dependence response tables and figures;
- optional Tree SHAP mean-absolute feature attribution.

### Environmental transfer and raster scale

- predictor-wise training envelopes;
- row-level and feature-level extrapolation diagnostics;
- normalized distance beyond the observed envelope;
- bounded-memory windowed raster prediction.

## v0.7 — Ecological safeguards and productization

**Goal:** make the end-to-end workflow more ecologically defensible, reproducible, installable, and inspectable without hiding modeling assumptions.

Implemented:

### Background / pseudo-absence design

- equal-area random background generation;
- spatially stratified background generation;
- candidate-pool / target-group background sampling;
- explicit minimum-distance exclusion from presence locations;
- deterministic seeds and provenance metadata.

### Spatial thinning

- minimum geodesic-distance thinning;
- deterministic tie handling;
- optional priority column so higher-quality records can be retained first;
- retained/removed record accounting.

### Environmental collinearity

- Pearson correlation matrix;
- high-correlation-pair reporting;
- variance-inflation factors (VIF);
- user-controlled correlation and VIF warning thresholds;
- no silent predictor deletion.

### Novel-climate safeguards

- existing univariate training-envelope diagnostics retained;
- multivariate standardized environmental-distance diagnostics;
- training-derived novelty threshold;
- combined row-level `novel_climate_warning` output.

This is described as a transparent MESS-style warning layer, not as an exact implementation of every published MESS variant.

### Dispersal constraints

- explicit maximum-distance dispersal assumption;
- accessibility raster from currently suitable habitat;
- separation of accessible and beyond-distance future-suitable cells;
- dispersal-constrained future suitability raster;
- downstream range-shift comparison can use the constrained surface.

### Reproducible configuration

- JSON `RunConfig` schema;
- one-command calibrated current-to-future workflow;
- optional spatial validation;
- optional dispersal constraint;
- SHA-256 hash of the normalized configuration;
- machine-readable run manifest recording model, threshold, assumptions, diagnostics, and output paths;
- example configuration file.

### Product interface

- `rangeshift-eco` ecological-safeguard CLI;
- `rangeshift-run` configuration-driven workflow CLI;
- packaged `rangeshift-app` launcher;
- Streamlit explorer for suitability rasters, range-shift rasters, and summary/run-manifest JSON;
- explicit warning in the interface that suitability projections are not guaranteed future distributions.

### Packaging and release engineering

- Python wheel build;
- source distribution build;
- Twine metadata validation;
- clean wheel installation test;
- automated GitHub Release workflow;
- distribution artifacts attached to versioned releases;
- Python 3.10, 3.11, and 3.12 CI;
- dedicated geospatial, explainability, app, and package-build jobs.

## Post-v0.7 roadmap

These are **future extensions**, not unfinished Phase 6 boxes.

### v0.8 candidates — uncertainty and scenario ensembles

- uncertainty summaries across resamples and candidate models;
- batch climate-model / SSP scenario comparison;
- ensemble consensus and disagreement maps;
- threshold-sensitivity summaries;
- scenario-level uncertainty tables and figures.

### v0.9 candidates — scale and provenance

- tiled or cloud-optimized raster workflows;
- provenance metadata embedded directly in output rasters;
- richer scenario manifests;
- exportable analysis reports;
- configuration-schema versioning and migration support.

### v1.0 target

A v1.0 release should let a user move from occurrence records and environmental layers to a documented range-shift report without hiding the underlying scientific assumptions.

Expected v1.0 outputs include:

- validated occurrence/environmental inputs;
- background-generation and sampling-bias diagnostics;
- spatially evaluated candidate models;
- documented model selection and calibration;
- explicit threshold selection;
- current and future suitability rasters;
- extrapolation and novelty warnings;
- optional dispersal-constrained projection;
- gain/loss/stability raster;
- continuous suitability-change raster;
- range-area, overlap, and centroid-shift summaries;
- model interpretation;
- uncertainty and transferability warnings;
- reproducible configuration/run metadata;
- exportable figures and tables.

## Scientific boundaries

RangeShift AI must never present habitat-suitability projections as guaranteed future distributions. Realized ranges may differ because of dispersal, barriers, biotic interactions, adaptation, demographic processes, sampling bias, detectability, land-use change, environmental novelty, and future-scenario uncertainty.

A model producing a probability is not evidence that its projection environment is familiar. High apparent predictive accuracy is not automatically evidence of geographic transferability. A classified suitability map is not equivalent to realized occupancy.

Those limitations are part of the analysis, not footnotes to hide.
