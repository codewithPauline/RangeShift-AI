"""Render the validated four-panel crossed-uncertainty GeoTIFF figure and summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.ticker import FormatStrFormatter

PANELS = (
    ("crossed_mean.tif", "A   Mean modeled suitability", "viridis", 0.85),
    ("crossed_total_sd.tif", "B   Total score variability (SD)", "magma", 0.35),
    ("crossed_scenario_sd.tif", "C   Between-scenario component (SD)", "magma", 0.35),
    ("crossed_model_sd.tif", "D   Within-scenario model component (SD)", "magma", 0.35),
)


def render(raster_dir: Path, figure_path: Path, summary_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    statistics = {}
    for ax, (filename, heading, cmap, vmax) in zip(axes.flat, PANELS, strict=True):
        with rasterio.open(raster_dir / filename) as ds:
            values = ds.read(1, masked=True)
            bounds = ds.bounds
            if not values.count():
                raise ValueError(f"No valid cells in {filename}")
            valid = values.compressed()
            statistics[filename] = {
                "valid_cells": int(valid.size),
                "mean": float(np.mean(valid)),
                "median": float(np.median(valid)),
                "minimum": float(np.min(valid)),
                "maximum": float(np.max(valid)),
            }
            image = ax.imshow(
                values,
                origin="upper",
                extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                interpolation="nearest",
            )
        ax.set_title(heading, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        fig.colorbar(
            image,
            ax=ax,
            shrink=0.76,
            label="Model score" if filename == "crossed_mean.tif" else "Across-ensemble SD",
        )
    counts = {item["valid_cells"] for item in statistics.values()}
    if len(counts) != 1:
        raise ValueError("Crossed output rasters have inconsistent valid-cell counts.")
    fig.suptitle(
        "RangeShift AI  |  Plethodon cinereus  |  2061–2080",
        fontsize=16, weight="bold",
    )
    fig.text(
        0.5, -0.01,
        "Public-data software demonstration · 12 block-bootstrap models × 6 CMIP6 "
        "scenarios · equal weights · uncalibrated scores · not a conservation forecast",
        ha="center", fontsize=8,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary_path.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raster-dir", type=Path,
        default=Path("plethodon_crossed_output/rasters"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("plethodon_crossed_output/crossed_uncertainty.png"),
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("plethodon_crossed_output/figure_statistics.json"),
    )
    args = parser.parse_args()
    render(args.raster_dir, args.figure, args.summary)
    print(f"Figure: {args.figure}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
