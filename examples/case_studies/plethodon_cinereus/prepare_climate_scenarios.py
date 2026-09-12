"""Prepare aligned WorldClim current and CMIP6 future layers for the case study.

Run the GBIF + current-climate training-data preparation first, then run this
script. Large downloaded rasters stay in the generated output directory and are
not intended to be committed to Git.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rangeshift.climate import (
    align_raster_band_to_reference,
    crop_raster_band_to_bounds,
    extract_bioclim_bands_to_reference,
    validate_aligned_rasters,
    worldclim_cmip6_bioc_url,
)

CURRENT_WORLDCLIM_URL = (
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip"
)
BIOCLIM_FEATURES = {"bio1": 1, "bio12": 12, "bio15": 15}
DEFAULT_GCMS = ["ACCESS-CM2", "MIROC6", "MRI-ESM2-0"]
DEFAULT_SSPS = ["ssp245", "ssp585"]
DEFAULT_PERIOD = "2061-2080"


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "RangeShift-AI Plethodon-cinereus case study"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        destination.write_bytes(response.read())
    return destination


def _study_bounds(training_csv: Path, buffer_degrees: float) -> tuple[float, float, float, float]:
    frame = pd.read_csv(training_csv)
    required = {"presence", "latitude", "longitude"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Training table is missing required columns: {', '.join(missing)}")
    presences = frame.loc[frame["presence"] == 1]
    if presences.empty:
        raise ValueError("Training table contains no presence rows.")
    if buffer_degrees < 0:
        raise ValueError("buffer_degrees cannot be negative.")
    return (
        max(-180.0, float(presences["longitude"].min()) - buffer_degrees),
        max(-90.0, float(presences["latitude"].min()) - buffer_degrees),
        min(180.0, float(presences["longitude"].max()) + buffer_degrees),
        min(90.0, float(presences["latitude"].max()) + buffer_degrees),
    )


def _extract_current_sources(archive: Path, cache_dir: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    extract_dir = cache_dir / "current_worldclim"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        for feature, band in BIOCLIM_FEATURES.items():
            filename = f"wc2.1_10m_bio_{band}.tif"
            if filename not in names:
                raise RuntimeError(f"WorldClim archive is missing {filename}")
            destination = extract_dir / filename
            if not destination.exists():
                with zipped.open(filename) as source, destination.open("wb") as target:
                    target.write(source.read())
            output[feature] = destination
    return output


def prepare_current_layers(
    cache_dir: Path,
    climate_dir: Path,
    bounds: tuple[float, float, float, float],
) -> dict[str, Path]:
    archive = _download(CURRENT_WORLDCLIM_URL, cache_dir / "wc2.1_10m_bio.zip")
    sources = _extract_current_sources(archive, cache_dir)
    current_dir = climate_dir / "current"
    reference = crop_raster_band_to_bounds(
        sources["bio1"],
        current_dir / "bio1.tif",
        bounds=bounds,
    )
    outputs = {"bio1": reference}
    for feature in ("bio12", "bio15"):
        outputs[feature] = align_raster_band_to_reference(
            sources[feature],
            reference,
            current_dir / f"{feature}.tif",
        )
    validate_aligned_rasters(outputs)
    return outputs


def prepare_future_scenarios(
    cache_dir: Path,
    climate_dir: Path,
    reference_path: Path,
    *,
    gcms: list[str],
    ssps: list[str],
    period: str,
) -> tuple[dict[str, dict[str, Path]], dict[str, dict[str, str]]]:
    scenarios: dict[str, dict[str, Path]] = {}
    provenance: dict[str, dict[str, str]] = {}
    for gcm in gcms:
        for ssp in ssps:
            scenario = f"{gcm}_{ssp}_{period}"
            url = worldclim_cmip6_bioc_url(gcm, ssp, period)
            source = _download(url, cache_dir / "future" / Path(url).name)
            layers = extract_bioclim_bands_to_reference(
                source,
                reference_path,
                climate_dir / "future" / scenario,
                features=BIOCLIM_FEATURES,
            )
            validate_aligned_rasters({"reference": reference_path, **layers})
            scenarios[scenario] = layers
            provenance[scenario] = {
                "gcm": gcm,
                "ssp": ssp,
                "period": period,
                "source_url": url,
            }
    return scenarios, provenance


def _stringify_layers(layers: dict[str, Path]) -> dict[str, str]:
    return {feature: str(path) for feature, path in layers.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=Path("real_ecology_output/gbif_worldclim_training.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plethodon_cinereus_climate"),
    )
    parser.add_argument("--buffer-degrees", type=float, default=2.0)
    parser.add_argument("--gcms", nargs="+", default=DEFAULT_GCMS)
    parser.add_argument(
        "--ssps",
        nargs="+",
        choices=["ssp126", "ssp245", "ssp370", "ssp585"],
        default=DEFAULT_SSPS,
    )
    parser.add_argument(
        "--period",
        choices=["2021-2040", "2041-2060", "2061-2080", "2081-2100"],
        default=DEFAULT_PERIOD,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.training_csv.exists():
        raise FileNotFoundError(
            f"Training table not found: {args.training_csv}. Run "
            "examples/real_ecology/prepare_gbif_worldclim.py first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    climate_dir = args.output_dir / "climate"
    bounds = _study_bounds(args.training_csv, args.buffer_degrees)

    current_layers = prepare_current_layers(cache_dir, climate_dir, bounds)
    scenarios, scenario_provenance = prepare_future_scenarios(
        cache_dir,
        climate_dir,
        current_layers["bio1"],
        gcms=list(args.gcms),
        ssps=list(args.ssps),
        period=args.period,
    )

    scenario_layers_path = args.output_dir / "scenario_layers.json"
    scenario_layers_path.write_text(
        json.dumps(
            {name: _stringify_layers(layers) for name, layers in scenarios.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    first_scenario = next(iter(scenarios))
    base_config = {
        "training_csv": str(args.training_csv),
        "features": list(BIOCLIM_FEATURES),
        "current_layers": _stringify_layers(current_layers),
        "future_layers": _stringify_layers(scenarios[first_scenario]),
        "output_dir": str(args.output_dir / "single_scenario_template"),
        "target": "presence",
        "latitude_column": "latitude",
        "longitude_column": "longitude",
        "seed": 42,
        "calibration_method": "sigmoid",
        "threshold_method": "tss",
        "calibration_cv": 5,
        "spatial_validation": True,
        "spatial_block_size_degrees": 1.0,
        "spatial_test_size": 0.25,
        "raster_window_size": 512,
        "dispersal_max_distance_km": None,
    }
    base_config_path = args.output_dir / "base_run_config.json"
    base_config_path.write_text(json.dumps(base_config, indent=2, sort_keys=True) + "\n")

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "species": "Plethodon cinereus",
        "training_csv": str(args.training_csv),
        "worldclim_version": "2.1",
        "resolution": "10 minutes",
        "current_source_url": CURRENT_WORLDCLIM_URL,
        "future_archive": "WorldClim CMIP6 downscaled and bias-corrected projections",
        "features": BIOCLIM_FEATURES,
        "study_bounds": {
            "left": bounds[0],
            "bottom": bounds[1],
            "right": bounds[2],
            "top": bounds[3],
        },
        "scenario_count": len(scenarios),
        "scenarios": scenario_provenance,
        "alignment": (
            "All current and future predictor rasters are resampled to the cropped "
            "current bio1 reference grid using bilinear interpolation for continuous "
            "bioclimatic variables."
        ),
        "interpretation": (
            "The default six projections are a transparent software demonstration, "
            "not a claim that three GCMs or two SSPs fully characterize climate uncertainty."
        ),
    }
    provenance_path = args.output_dir / "climate_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    print(f"Prepared current grid: {current_layers['bio1']}")
    print(f"Prepared future scenarios: {len(scenarios)}")
    print(f"Scenario layers: {scenario_layers_path}")
    print(f"Base config: {base_config_path}")
    print(f"Climate provenance: {provenance_path}")


if __name__ == "__main__":
    main()
