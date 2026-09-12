# RangeShift AI v0.8.0

## Highlights

RangeShift AI v0.8.0 adds a verified, reproducible real-data flagship case study using the eastern red-backed salamander, *Plethodon cinereus*.

The release demonstrates the complete public workflow from GBIF occurrence records and WorldClim current climate through six CMIP6 future projections, multi-scenario range-shift analysis, uncertainty summaries, provenance capture, and a generated four-panel flagship figure.

## What is new

- real public *Plethodon cinereus* case study;
- six CMIP6 projections for 2061–2080 using ACCESS-CM2, MIROC6, and MRI-ESM2-0 under SSP245 and SSP585;
- multi-scenario suitability summaries and scenario agreement outputs;
- real range-shift classification and centroid-shift summaries;
- committed compact provenance and scenario results;
- README flagship figure generated from actual RangeShift outputs;
- executable GitHub Actions workflow for reproducing the case study;
- tighter case-study workflow triggers to avoid unnecessary redownloads for documentation-only changes.

## Verified demonstration snapshot

The public demonstration retained 382 filtered GBIF presence records and selected a suitability threshold of 0.53. Across the six supplied climate projections, current suitable area was approximately 393,955 km², projected suitable area ranged from approximately 256,576 to 415,408 km², range-area change ranged from -34.9% to +5.4%, and estimated suitable-range centroid shifts ranged from approximately 397 to 592 km.

These values are software-demonstration outputs under the stated occurrence sample, predictor set, threshold, GCM/SSP ensemble, and modeling assumptions. They are not presented as a species conservation forecast.

## Reproducibility

The case-study workflow records occurrence and climate provenance, scenario composition, configuration hashes, selected threshold, scenario-level summaries, and uncertainty rasters. Large downloaded climate rasters remain excluded from Git history.

## Installation

Download the wheel or source distribution attached to the GitHub release and install with pip. Geospatial support requires the `geo` optional dependency.
