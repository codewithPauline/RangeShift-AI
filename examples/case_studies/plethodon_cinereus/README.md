# Case study: red-backed salamander (*Plethodon cinereus*)

This case study is the flagship real-data demonstration for RangeShift AI v0.8.

It uses **public occurrence and climate data** to demonstrate the complete RangeShift workflow without using unpublished dissertation data or the focal manuscript taxa *Ambystoma barbouri* and *A. texanum*.

## Why this species?

The eastern red-backed salamander is a strong demonstration species because it is widespread in eastern North America, strongly associated with forest environments, dispersal-limited at local scales, and has an established literature linking temperature, precipitation, range limits, and climate sensitivity.

The goal is not to claim a definitive forecast for the species. The case study is designed to show how RangeShift makes modeling assumptions, transfer risks, thresholds, spatial validation, scenario disagreement, and range-change summaries explicit.

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
Multiple future climate scenarios
      ↓
Per-scenario future suitability + range shift
      ↓
Environmental novelty diagnostics
      ↓
Optional dispersal-distance constraint
      ↓
Scenario uncertainty: mean + SD + suitability agreement
      ↓
Area change + overlap + centroid shift by scenario
      ↓
Run manifests + case-study figure
```

## Multi-scenario uncertainty

RangeShift v0.8 adds a batch scenario layer without changing the existing single-scenario workflow. A shared analysis configuration can be projected across multiple future climate-layer sets using `run_scenario_batch`.

```python
from rangeshift import RunConfig, run_scenario_batch

base = RunConfig(
    training_csv="data/training.csv",
    features=["bio1", "bio12", "bio15"],
    current_layers={
        "bio1": "climate/current/bio1.tif",
        "bio12": "climate/current/bio12.tif",
        "bio15": "climate/current/bio15.tif",
    },
    future_layers={
        "bio1": "climate/future/example/bio1.tif",
        "bio12": "climate/future/example/bio12.tif",
        "bio15": "climate/future/example/bio15.tif",
    },
    output_dir="single_run",
)

scenarios = {
    "gcm_a_ssp245_2061_2080": {
        "bio1": "climate/future/gcm_a_ssp245/bio1.tif",
        "bio12": "climate/future/gcm_a_ssp245/bio12.tif",
        "bio15": "climate/future/gcm_a_ssp245/bio15.tif",
    },
    "gcm_b_ssp245_2061_2080": {
        "bio1": "climate/future/gcm_b_ssp245/bio1.tif",
        "bio12": "climate/future/gcm_b_ssp245/bio12.tif",
        "bio15": "climate/future/gcm_b_ssp245/bio15.tif",
    },
}

result = run_scenario_batch(base, scenarios, "scenario_runs")
```

The uncertainty directory contains:

- `future_suitability_mean.tif` — cell-wise mean suitability across supplied scenarios;
- `future_suitability_sd.tif` — cell-wise between-scenario spread;
- `future_suitable_fraction.tif` — fraction of scenarios above the same validated suitability threshold;
- `scenario_summary.csv` — scenario-level outputs and scalar range-shift metrics;
- `scenario_manifest.json` — paths, configuration hashes, threshold, and interpretation metadata.

**Important:** `future_suitable_fraction.tif` is scenario agreement, not the probability that the species will occupy a cell. Its meaning depends entirely on the set of climate scenarios supplied.

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
- future climate model, SSP, and time horizon for every scenario;
- environmental-novelty diagnostics;
- dispersal assumption, if used;
- scenario-set composition and uncertainty outputs;
- RangeShift software version;
- normalized configuration hash.

## Required flagship outputs

The final public example should contain one main figure with four panels:

1. **Current suitability**
2. **Representative future suitability**
3. **Range shift** — stable suitable, habitat gained, and habitat lost
4. **Scenario agreement / uncertainty**

A compact summary should report current and future suitable area, gained and lost area, net change, Jaccard overlap, centroid shift distance and bearing, selected threshold, spatial test performance, and the range of outcomes across future scenarios.

## Scientific caution

This is a reproducible software demonstration, not a substitute for a species-specific conservation assessment. Results depend on the occurrence sample, background design, predictor choice, validation scheme, climate scenario set, threshold, transferability, environmental novelty, and dispersal assumptions.

For publication-grade use, occurrence data should be archived through a citable GBIF download DOI rather than relying only on live API retrieval.
