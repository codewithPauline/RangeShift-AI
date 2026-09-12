# Case study: red-backed salamander (*Plethodon cinereus*)

This case study is the flagship real-data demonstration for RangeShift AI v0.8.

It uses **public occurrence and climate data** to demonstrate the complete RangeShift workflow without using unpublished dissertation data or the focal manuscript taxa *Ambystoma barbouri* and *A. texanum*.

## Why this species?

The eastern red-backed salamander is a strong demonstration species because it is widespread in eastern North America, strongly associated with forest environments, dispersal-limited at local scales, and has an established literature linking temperature, precipitation, range limits, and climate sensitivity.

The goal is not to claim a definitive forecast for the species. The case study is designed to show how RangeShift makes modeling assumptions, transfer risks, thresholds, spatial validation, and range-change summaries explicit.

## Public-data workflow

```text
GBIF occurrences
      ↓
Coordinate filtering and duplicate handling
      ↓
Spatial thinning + sampling-bias diagnostics
      ↓
Background generation
      ↓
WorldClim current predictors
      ↓
Collinearity diagnostics
      ↓
Spatial model validation
      ↓
Random Forest ↔ Gradient Boosting comparison
      ↓
Probability calibration
      ↓
Validation-only threshold selection
      ↓
Current suitability map
      ↓
Future climate projection
      ↓
Environmental novelty diagnostics
      ↓
Optional dispersal-distance constraint
      ↓
Stable / lost / gained / stable-suitable habitat
      ↓
Area change + overlap + centroid shift
      ↓
Run manifest + case-study figure
```

## Reproducibility target

A completed case-study run should record:

- scientific name and resolved GBIF taxon identifier;
- occurrence retrieval date and provenance;
- number of raw and retained occurrence records;
- spatial-thinning distance;
- background-generation method and seed;
- environmental predictors;
- spatial-block definition;
- candidate models and selected model;
- calibration method;
- selected suitability threshold;
- current climate source;
- future climate model, SSP, and time horizon;
- environmental-novelty diagnostics;
- dispersal assumption, if used;
- RangeShift software version;
- normalized configuration hash.

## Required flagship outputs

The final public example should contain one main figure with three panels:

1. **Current suitability**
2. **Future suitability**
3. **Range shift** — stable suitable, habitat gained, and habitat lost

A compact summary should report current and future suitable area, gained and lost area, net change, Jaccard overlap, centroid shift distance and bearing, selected threshold, and spatial test performance.

## Scientific caution

This is a reproducible software demonstration, not a substitute for a species-specific conservation assessment. Results depend on the occurrence sample, background design, predictor choice, validation scheme, climate scenario, threshold, transferability, environmental novelty, and dispersal assumptions.

For publication-grade use, occurrence data should be archived through a citable GBIF download DOI rather than relying only on live API retrieval.
