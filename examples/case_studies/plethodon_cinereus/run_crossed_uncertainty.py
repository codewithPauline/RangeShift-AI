"""Run a public-data crossed model x climate demonstration for Plethodon cinereus.

Requires outputs of prepare_gbif_worldclim.py and prepare_climate_scenarios.py.
No dissertation or private study data are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from rangeshift import summarize_crossed_uncertainty
from rangeshift.data import validate_training_frame


def run_case_study(
    training_csv: Path,
    scenario_json: Path,
    output_dir: Path,
    *,
    n_models: int = 12,
    n_estimators: int = 100,
    seed: int = 42,
    block_degrees: float = 1.0,
) -> Path:
    """Resample coarse geographic blocks, fit forests, project a crossed ensemble."""
    if n_models < 2 or n_estimators < 1 or block_degrees <= 0:
        raise ValueError("Require >=2 models, >=1 trees, and positive block_degrees.")
    frame = pd.read_csv(training_csv)
    features = ["bio1", "bio12", "bio15"]
    validate_training_frame(frame, features, "presence")
    if not {"longitude", "latitude"} <= set(frame):
        raise ValueError("Training CSV needs longitude and latitude columns.")
    if not np.isfinite(frame[["longitude", "latitude"]].to_numpy()).all():
        raise ValueError("Training coordinates must be finite.")
    raw = json.loads(scenario_json.read_text())
    if len(raw) < 2:
        raise ValueError("At least two climate scenarios are required.")

    # Resampling whole geographic blocks preserves records within each block.
    lon = np.floor((frame["longitude"].to_numpy() + 180) / block_degrees).astype(int)
    lat = np.floor((frame["latitude"].to_numpy() + 90) / block_degrees).astype(int)
    labels = list(zip(lon.tolist(), lat.tolist()))
    unique = sorted(set(labels))
    if len(unique) < 2:
        raise ValueError("Need two or more occupied spatial blocks.")
    blocks = [np.flatnonzero([label == item for label in labels]) for item in unique]
    y = frame["presence"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    bundles = []
    draw_sizes = []
    for replicate in range(n_models):
        for _attempt in range(100):
            selected = np.concatenate(
                [blocks[int(j)] for j in rng.integers(len(blocks), size=len(blocks))]
            )
            if len(np.unique(y[selected])) == 2:
                break
        else:
            raise ValueError("No two-class bootstrap after 100 draws; review block sampling.")
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=seed + replicate,
            n_jobs=-1,
        )
        estimator.fit(frame.iloc[selected][features], y[selected])
        bundles.append({"model": estimator, "feature_columns": features})
        draw_sizes.append(int(len(selected)))

    output_dir.mkdir(parents=True, exist_ok=True)
    result = summarize_crossed_uncertainty(bundles, raw, output_dir / "rasters")
    manifest = {
        "species": "Plethodon cinereus",
        "training_csv": str(training_csv),
        "scenario_layers_json": str(scenario_json),
        "scenario_names": list(raw),
        "n_models": n_models,
        "n_scenarios": result.scenario_count,
        "n_estimators": n_estimators,
        "random_state": seed,
        "spatial_block_degrees": block_degrees,
        "n_occupied_blocks": len(blocks),
        "bootstrap_draw_row_counts": draw_sizes,
        "common_valid_cells": result.valid_cells,
        "threshold": None,
        "mean_raster": str(result.mean_path),
        "total_sd_raster": str(result.total_sd_path),
        "scenario_sd_raster": str(result.scenario_sd_path),
        "model_sd_raster": str(result.model_sd_path),
        "interpretation": (
            "Software demonstration of equal-weighted scenario/model score variability. "
            "Uncalibrated forests, not validated predictive probabilities, occupancy "
            "likelihoods, formal confidence intervals, or conservation forecasts."
        ),
    }
    path = output_dir / "crossed_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-csv", type=Path,
        default=Path("real_ecology_output/gbif_worldclim_training.csv"),
    )
    parser.add_argument(
        "--scenario-json", type=Path,
        default=Path("plethodon_cinereus_climate/scenario_layers.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("plethodon_crossed_output"),
    )
    parser.add_argument("--models", type=int, default=12)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        run_case_study(
            args.training_csv, args.scenario_json, args.output_dir,
            n_models=args.models, n_estimators=args.trees, seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
