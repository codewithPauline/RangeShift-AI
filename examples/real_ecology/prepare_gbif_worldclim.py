"""Prepare a real GBIF + WorldClim dataset for a RangeShift demonstration.

The script downloads public occurrence records from GBIF, downloads WorldClim
2.1 bioclimatic rasters, extracts climate values at presence locations, samples
background climate cells within the observed study extent, and writes a compact
CSV suitable for RangeShift training.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

GBIF_BASE = "https://api.gbif.org/v1"
WORLDCLIM_BIO_URL = (
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip"
)
SELECTED_BIOCLIM = {"bio1": 1, "bio12": 12, "bio15": 15}


def _json_get(url: str, params: dict[str, object]) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "RangeShift-AI real-ecology example"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "RangeShift-AI real-ecology example"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        destination.write_bytes(response.read())
    return destination


def resolve_gbif_taxon(species: str) -> tuple[int, str]:
    """Resolve a scientific name against the GBIF backbone."""
    result = _json_get(f"{GBIF_BASE}/species/match", {"name": species})
    usage_key = result.get("usageKey")
    if usage_key is None:
        raise RuntimeError(f"GBIF could not resolve the species name: {species}")
    return int(usage_key), str(result.get("scientificName", species))


def fetch_occurrences(
    taxon_key: int,
    *,
    country: str,
    max_records: int,
) -> pd.DataFrame:
    """Retrieve quality-filtered georeferenced occurrence records from GBIF."""
    if max_records < 20:
        raise ValueError("max_records must be at least 20 for the example workflow.")

    records = []
    offset = 0
    while len(records) < max_records:
        limit = min(300, max_records - len(records))
        payload = _json_get(
            f"{GBIF_BASE}/occurrence/search",
            {
                "taxon_key": taxon_key,
                "country": country,
                "has_coordinate": "true",
                "has_geospatial_issue": "false",
                "occurrence_status": "PRESENT",
                "limit": limit,
                "offset": offset,
            },
        )
        page = payload.get("results", [])
        if not page:
            break
        for item in page:
            latitude = item.get("decimalLatitude")
            longitude = item.get("decimalLongitude")
            if latitude is None or longitude is None:
                continue
            records.append(
                {
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "gbif_key": item.get("key"),
                    "basis_of_record": item.get("basisOfRecord"),
                    "event_date": item.get("eventDate"),
                }
            )
            if len(records) >= max_records:
                break
        offset += len(page)
        if payload.get("endOfRecords", False):
            break

    if len(records) < 20:
        raise RuntimeError(
            "Too few GBIF records were returned. Try a more widespread species, "
            "a different country, or a larger geographic scope."
        )

    frame = pd.DataFrame(records)
    frame["coordinate_key"] = (
        frame["latitude"].round(4).astype(str)
        + "_"
        + frame["longitude"].round(4).astype(str)
    )
    return frame.drop_duplicates("coordinate_key").drop(columns="coordinate_key")


def prepare_worldclim(cache_dir: Path) -> dict[str, Path]:
    """Download WorldClim 2.1 10-minute bioclimatic rasters and select predictors."""
    archive = _download(WORLDCLIM_BIO_URL, cache_dir / "wc2.1_10m_bio.zip")
    extract_dir = cache_dir / "worldclim_bio"
    extract_dir.mkdir(parents=True, exist_ok=True)

    selected = {}
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        for feature, number in SELECTED_BIOCLIM.items():
            filename = f"wc2.1_10m_bio_{number}.tif"
            if filename not in names:
                raise RuntimeError(f"WorldClim archive is missing expected file: {filename}")
            target = extract_dir / filename
            if not target.exists():
                with zipped.open(filename) as source, target.open("wb") as destination:
                    destination.write(source.read())
            selected[feature] = target
    return selected


def extract_presence_climate(
    occurrences: pd.DataFrame,
    raster_paths: dict[str, Path],
) -> pd.DataFrame:
    """Sample selected WorldClim predictors at GBIF presence coordinates."""
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("Install RangeShift with the 'geo' extra to run this example.") from exc

    coordinates = list(zip(occurrences["longitude"], occurrences["latitude"], strict=True))
    output = occurrences.copy()
    valid = np.ones(len(output), dtype=bool)
    for feature, path in raster_paths.items():
        with rasterio.open(path) as dataset:
            values = np.array([sample[0] for sample in dataset.sample(coordinates)], dtype=float)
            nodata = dataset.nodata
            feature_valid = np.isfinite(values)
            if nodata is not None:
                feature_valid &= values != nodata
            valid &= feature_valid
            output[feature] = values
    output = output.loc[valid].copy()
    output["presence"] = 1
    output["source"] = "GBIF presence"
    return output


def sample_background(
    presences: pd.DataFrame,
    raster_paths: dict[str, Path],
    *,
    count: int,
    seed: int,
    buffer_degrees: float,
) -> pd.DataFrame:
    """Sample random climate-valid background cells inside the observed extent."""
    try:
        import rasterio
        from rasterio.transform import xy
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise ImportError("Install RangeShift with the 'geo' extra to run this example.") from exc

    if count < 1:
        raise ValueError("Background count must be positive.")
    rng = random.Random(seed)
    first_path = next(iter(raster_paths.values()))
    datasets = {feature: rasterio.open(path) for feature, path in raster_paths.items()}
    try:
        reference = datasets[next(iter(datasets))]
        left = max(-180.0, float(presences["longitude"].min()) - buffer_degrees)
        right = min(180.0, float(presences["longitude"].max()) + buffer_degrees)
        bottom = max(-90.0, float(presences["latitude"].min()) - buffer_degrees)
        top = min(90.0, float(presences["latitude"].max()) + buffer_degrees)
        window = from_bounds(left, bottom, right, top, transform=reference.transform)
        row_min = max(0, int(np.floor(window.row_off)))
        row_max = min(reference.height - 1, int(np.ceil(window.row_off + window.height)))
        col_min = max(0, int(np.floor(window.col_off)))
        col_max = min(reference.width - 1, int(np.ceil(window.col_off + window.width)))

        presence_keys = set(
            zip(
                presences["latitude"].round(3),
                presences["longitude"].round(3),
                strict=True,
            )
        )
        selected = []
        selected_keys = set()
        attempts = 0
        max_attempts = max(10000, count * 100)
        while len(selected) < count and attempts < max_attempts:
            attempts += 1
            row = rng.randint(row_min, row_max)
            col = rng.randint(col_min, col_max)
            longitude, latitude = xy(reference.transform, row, col, offset="center")
            key = (round(float(latitude), 3), round(float(longitude), 3))
            if key in presence_keys or key in selected_keys:
                continue

            record = {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "gbif_key": None,
                "basis_of_record": None,
                "event_date": None,
                "presence": 0,
                "source": "Random climate-valid background",
            }
            usable = True
            for feature, dataset in datasets.items():
                value = float(next(dataset.sample([(longitude, latitude)]))[0])
                if not np.isfinite(value) or (
                    dataset.nodata is not None and value == dataset.nodata
                ):
                    usable = False
                    break
                record[feature] = value
            if usable:
                selected.append(record)
                selected_keys.add(key)

        if len(selected) < count:
            raise RuntimeError(
                f"Only {len(selected)} valid background cells were found after "
                f"{attempts} attempts. Reduce --background or increase --buffer-degrees."
            )
        return pd.DataFrame(selected)
    finally:
        for dataset in datasets.values():
            dataset.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", default="Ambystoma maculatum")
    parser.add_argument("--country", default="US")
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--background", type=int, default=500)
    parser.add_argument("--buffer-degrees", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("real_ecology_output"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"

    taxon_key, matched_name = resolve_gbif_taxon(args.species)
    occurrences = fetch_occurrences(
        taxon_key,
        country=args.country,
        max_records=args.max_records,
    )
    raster_paths = prepare_worldclim(cache_dir)
    presences = extract_presence_climate(occurrences, raster_paths)
    backgrounds = sample_background(
        presences,
        raster_paths,
        count=args.background,
        seed=args.seed,
        buffer_degrees=args.buffer_degrees,
    )

    columns = [
        "presence",
        "bio1",
        "bio12",
        "bio15",
        "latitude",
        "longitude",
        "source",
        "gbif_key",
        "basis_of_record",
        "event_date",
    ]
    training = pd.concat([presences[columns], backgrounds[columns]], ignore_index=True)
    training = training.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    training_path = args.output_dir / "gbif_worldclim_training.csv"
    training.to_csv(training_path, index=False)

    metadata = {
        "requested_species": args.species,
        "gbif_matched_name": matched_name,
        "gbif_taxon_key": taxon_key,
        "country": args.country,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "presence_records_after_filtering": int(len(presences)),
        "background_records": int(len(backgrounds)),
        "worldclim_version": "2.1",
        "worldclim_resolution": "10 minutes",
        "worldclim_predictors": list(SELECTED_BIOCLIM),
        "gbif_api": "https://api.gbif.org/v1/occurrence/search",
        "worldclim_url": WORLDCLIM_BIO_URL,
        "note": (
            "This is a reproducible demonstration dataset, not a publication-ready occurrence "
            "download. For research publication, create a citable GBIF download DOI and document "
            "species-specific sampling and background design choices."
        ),
    }
    metadata_path = args.output_dir / "provenance.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Matched species: {matched_name} (GBIF key {taxon_key})")
    print(f"Presence rows: {len(presences)}")
    print(f"Background rows: {len(backgrounds)}")
    print(f"Training table: {training_path}")
    print(f"Provenance: {metadata_path}")


if __name__ == "__main__":
    main()
