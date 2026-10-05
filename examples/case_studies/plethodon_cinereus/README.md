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

## Runnable climate workflow

The case study now includes an end-to-end WorldClim climate-preparation path. It uses WorldClim 2.1 current bioclimatic layers and the official downscaled CMIP6 archive at 10-minute resolution.

The default future set contains **six projections** for **2061–2080**:

- ACCESS-CM2 × SSP245
- ACCESS-CM2 × SSP585
- MIROC6 × SSP245
- MIROC6 × SSP585
- MRI-ESM2-0 × SSP245
- MRI-ESM2-0 × SSP585

This is intentionally a transparent software-demonstration ensemble, not a claim that three GCMs and two SSPs fully represent future climate uncertainty.

### 1. Prepare public GBIF occurrences and current WorldClim training data

```bash
python examples/real_ecology/prepare_gbif_worldclim.py \
  --species "Plethodon cinereus" \
  --output-dir real_ecology_output
```

### 2. Download and align current + future climate rasters

```bash
python examples/case_studies/plethodon_cinereus/prepare_climate_scenarios.py
```

The script derives a study extent from the retained presence coordinates, adds a configurable geographic buffer, crops the current WorldClim grid, downloads the requested CMIP6 bioclimatic projections, and resamples every future predictor onto the exact same current reference grid.

Generated files include:

```text
plethodon_cinereus_climate/
├── base_run_config.json
├── climate_provenance.json
├── scenario_layers.json
├── cache/
└── climate/
    ├── current/
    │   ├── bio1.tif
    │   ├── bio12.tif
    │   └── bio15.tif
    └── future/
        ├── ACCESS-CM2_ssp245_2061-2080/
        ├── ACCESS-CM2_ssp585_2061-2080/
        ├── MIROC6_ssp245_2061-2080/
        ├── MIROC6_ssp585_2061-2080/
        ├── MRI-ESM2-0_ssp245_2061-2080/
        └── MRI-ESM2-0_ssp585_2061-2080/
```

Large climate rasters are generated locally and are deliberately excluded from Git.

### 3. Run all scenarios through RangeShift

```bash
python examples/case_studies/plethodon_cinereus/run_scenario_batch.py
```

This produces per-scenario suitability and range-shift outputs plus the cross-scenario uncertainty products described below.

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

## Crossed model × climate uncertainty: verified public-data demonstration

The post-v0.8 development workflow now combines **12 spatial-block bootstrap
Random Forest fits** with the six CMIP6 future scenarios above, yielding **72
model–scenario score predictions per common valid grid cell**. The run uses
a 1° geographic block bootstrap, 100 trees per fit, fixed seed 42, and
*uncalibrated* scores; it does not replace the independently calibrated v0.8
case-study pipeline. The current example intentionally does **not** apply a
suitability threshold, because none was validated for these separate fits.

A successfully completed GitHub Actions run generated four GeoTIFFs and
a provenance manifest over **7,740 common valid cells**. Descriptive
cellwise summaries from that run are:

| Output | Mean across valid cells | Interpretation |
| --- | ---: | --- |
| Mean modeled score | 0.2613 | Mean of all model × climate combinations |
| Total score SD | 0.1325 | Entire crossed ensemble |
| Scenario component SD | 0.0848 | Differences between scenario means |
| Model-fit component SD | 0.0975 | Average within-scenario fit variance, square-rooted |

**These averages cannot be used as an additive variance decomposition.**
The underlying identity holds **within each cell, on the variance scale**:
total variance = between-scenario variance + mean within-scenario model
variance. Likewise, comparing the two spatially averaged SDs is not a
formal attribution of forecast uncertainty. The sampled GCM–SSP set is
equally weighted for demonstration, not weighted by its probability.

The geographically structured score variation visible in the results is
conditional on public occurrence filtering, coarse WorldClim 10-minute
predictors, selected models, spatial blocks, and six scenarios. It must not
be interpreted as a validated forecast of realized distribution, a
species-occupancy probability, or a conservation assessment.

### Fully automated reproduction

The
[Crossed Plethodon case-study workflow](../../../.github/workflows/crossed-case-study.yml)
prepares public GBIF/WorldClim inputs, fits all models, projects the
crossed ensemble, generates a four-panel PNG **from the computed rasters**,
writes a machine-readable numerical summary, and uploads the figure,
GeoTIFFs, and provenance as one GitHub Actions artifact.

The workflow runs automatically for relevant development-branch changes;
after merging, it can also be launched from GitHub Actions with
**Run workflow** if the source data need refreshing. Its artifacts expire
after 30 days; numerical claims in the repository reflect the verified
run documented above, not live source updates. Software-only CI checks
use small synthetic fixtures and do not substitute for this external-data
execution.

The four panels are: (A) mean score; (B) total SD; (C) scenario SD;
(D) model-fit SD, using matched color ranges across SD panels. The
reproducible renderer is
[`make_crossed_figure.py`](make_crossed_figure.py).
