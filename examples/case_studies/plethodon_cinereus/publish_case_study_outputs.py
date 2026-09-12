"""Publish compact verified case-study outputs and refresh the root README."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

MARKER = "<!-- PLETHODON_CASE_STUDY -->"
END_MARKER = "<!-- /PLETHODON_CASE_STUDY -->"


def _fmt(value: float, digits: int = 1) -> str:
    return f"{float(value):,.{digits}f}"


def publish(artifact_dir: Path, repository_root: Path) -> None:
    figure_source = artifact_dir / "plethodon_cinereus_flagship.png"
    results_source = artifact_dir / "results"
    summary_source = results_source / "scenario_summary.csv"
    if not figure_source.exists() or not summary_source.exists():
        raise FileNotFoundError("Expected verified case-study artifacts were not found.")

    asset_dir = repository_root / "docs" / "assets"
    result_dir = (
        repository_root
        / "examples"
        / "case_studies"
        / "plethodon_cinereus"
        / "results"
    )
    asset_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    figure_target = asset_dir / "plethodon_cinereus_case_study.png"
    shutil.copy2(figure_source, figure_target)
    for filename in (
        "scenario_summary.csv",
        "scenario_manifest.json",
        "gbif_worldclim_provenance.json",
        "climate_provenance.json",
    ):
        shutil.copy2(results_source / filename, result_dir / filename)

    summary = pd.read_csv(summary_source)
    current_area = float(summary["range_current_suitable_area_km2"].iloc[0])
    future_min = float(summary["range_future_suitable_area_km2"].min())
    future_max = float(summary["range_future_suitable_area_km2"].max())
    percent_min = float(summary["range_percent_change_from_current"].min())
    percent_max = float(summary["range_percent_change_from_current"].max())
    centroid_min = float(summary["range_centroid_shift_km"].min())
    centroid_max = float(summary["range_centroid_shift_km"].max())
    threshold = float(summary["selected_threshold"].iloc[0])

    provenance = json.loads((results_source / "gbif_worldclim_provenance.json").read_text())
    retained = int(provenance["presence_records_after_filtering"])

    block = f"""{MARKER}
## Real public case study: *Plethodon cinereus*

RangeShift is exercised end to end on a reproducible public-data demonstration using
**{retained} filtered GBIF presence records**, WorldClim 2.1 current climate, and six
CMIP6 projections (ACCESS-CM2, MIROC6, and MRI-ESM2-0 × SSP245/SSP585; 2061–2080).
The figure below is generated from the actual RangeShift outputs, not synthetic maps.

![Real case-study outputs](docs/assets/plethodon_cinereus_case_study.png)

Across the six supplied climate projections, the validated threshold was **{threshold:.2f}**.
Current suitable area was approximately **{_fmt(current_area, 0)} km²**; projected suitable
area ranged from **{_fmt(future_min, 0)} to {_fmt(future_max, 0)} km²**, corresponding to
**{percent_min:.1f}% to +{percent_max:.1f}%** change from current suitability. Estimated
suitable-range centroid shifts ranged from **{_fmt(centroid_min, 0)} to
{_fmt(centroid_max, 0)} km**.

These numbers describe this software demonstration under its stated occurrence sample,
predictors, threshold, GCM/SSP set, and modeling assumptions. They are **not** presented as
a species conservation forecast. See
[`examples/case_studies/plethodon_cinereus/`](examples/case_studies/plethodon_cinereus/)
for the workflow and compact provenance/results.
{END_MARKER}
"""

    readme_path = repository_root / "README.md"
    readme = readme_path.read_text()
    readme = readme.replace(
        "By default it uses the spotted salamander, *Ambystoma maculatum*, and:",
        "By default it uses the eastern red-backed salamander, *Plethodon cinereus*, and:",
    )
    readme = readme.replace(
        '--species "Ambystoma maculatum"',
        '--species "Plethodon cinereus"',
    )

    if MARKER in readme and END_MARKER in readme:
        before = readme.split(MARKER, 1)[0]
        after = readme.split(END_MARKER, 1)[1]
        readme = before + block + after
    else:
        insertion = "## Expected tabular input"
        if insertion not in readme:
            raise ValueError("README insertion point was not found.")
        readme = readme.replace(insertion, block + "\n" + insertion, 1)

    readme_path.write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("case_study_artifacts"))
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args()
    publish(args.artifact_dir, args.repository_root)


if __name__ == "__main__":
    main()
