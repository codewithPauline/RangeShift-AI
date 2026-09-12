"""Create the public RangeShift flagship figure from completed case-study outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


def _read_raster(path: Path):
    import rasterio

    with rasterio.open(path) as dataset:
        values = dataset.read(1).astype(float)
        if dataset.nodata is not None:
            values[values == dataset.nodata] = np.nan
        bounds = dataset.bounds
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return values, extent


def _first_scenario(manifest: dict) -> tuple[str, dict]:
    scenarios = manifest.get("scenarios", {})
    if not scenarios:
        raise ValueError("Scenario manifest contains no scenarios.")
    preferred = next((name for name in scenarios if "MIROC6_ssp245" in name), None)
    name = preferred or sorted(scenarios)[0]
    return name, scenarios[name]


def build_figure(batch_dir: Path, output_path: Path) -> Path:
    manifest_path = batch_dir / "scenario_manifest.json"
    summary_path = batch_dir / "scenario_summary.csv"
    if not manifest_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Completed scenario-batch outputs were not found.")

    manifest = json.loads(manifest_path.read_text())
    scenario_name, scenario = _first_scenario(manifest)
    run_manifest = json.loads(Path(scenario["manifest"]).read_text())

    current_path = Path(run_manifest["outputs"]["current_suitability"])
    future_path = Path(run_manifest["outputs"]["future_suitability"])
    classes_path = Path(run_manifest["outputs"]["range_shift_classes"])
    agreement_path = Path(manifest["uncertainty"]["suitable_fraction"])

    current, extent = _read_raster(current_path)
    future, _ = _read_raster(future_path)
    classes, _ = _read_raster(classes_path)
    agreement, _ = _read_raster(agreement_path)

    summary = pd.read_csv(summary_path)
    selected = summary.loc[summary["scenario"] == scenario_name].iloc[0]
    threshold = float(selected["selected_threshold"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    im0 = axes[0, 0].imshow(current, origin="upper", extent=extent, vmin=0, vmax=1)
    axes[0, 0].set_title("A. Current habitat suitability")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, label="Suitability")

    im1 = axes[0, 1].imshow(future, origin="upper", extent=extent, vmin=0, vmax=1)
    axes[0, 1].set_title(f"B. Future suitability — {scenario_name}")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, label="Suitability")

    class_cmap = ListedColormap(["#d9d9d9", "#d95f02", "#1f78b4", "#238b45"])
    im2 = axes[1, 0].imshow(classes, origin="upper", extent=extent, vmin=0, vmax=3, cmap=class_cmap)
    axes[1, 0].set_title("C. Range shift")
    cbar = fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["Stable unsuitable", "Lost", "Gained", "Stable suitable"])

    im3 = axes[1, 1].imshow(agreement, origin="upper", extent=extent, vmin=0, vmax=1)
    axes[1, 1].set_title("D. Scenario agreement")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, label="Fraction of scenarios suitable")

    for ax in axes.flat:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    net = selected.get("range_percent_change_from_current", np.nan)
    jaccard = selected.get("range_jaccard_overlap", np.nan)
    shift = selected.get("range_centroid_shift_km", np.nan)
    fig.suptitle(
        "RangeShift AI — Plethodon cinereus public case study\n"
        f"Threshold={threshold:.3f} · net suitable-area change={net:.1f}% · "
        f"Jaccard={jaccard:.2f} · centroid shift={shift:.1f} km",
        fontsize=16,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=Path("plethodon_cinereus_scenario_runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("case_study_artifacts/plethodon_cinereus_flagship.png"),
    )
    args = parser.parse_args()
    print(build_figure(args.batch_dir, args.output))


if __name__ == "__main__":
    main()
