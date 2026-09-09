"""Packaged Streamlit explorer for RangeShift outputs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.plot import plotting_extent

from .visualization import RANGE_SHIFT_LABELS

RANGE_SHIFT_COLORS = ["#e5e7eb", "#d55e00", "#0072b2", "#009e73"]


def _temporary_upload(uploaded, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(uploaded.getvalue())
        return Path(handle.name)


def _render_suitability(uploaded) -> None:
    path = _temporary_upload(uploaded, ".tif")
    try:
        with rasterio.open(path) as dataset:
            values = dataset.read(1, masked=True)
            extent = plotting_extent(dataset)
            crs = dataset.crs
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        image = axis.imshow(
            values,
            extent=extent,
            origin="upper",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            interpolation="nearest",
        )
        figure.colorbar(image, ax=axis, label="Predicted suitability")
        axis.set_title("Habitat suitability")
        axis.set_xlabel("Longitude" if crs and crs.is_geographic else "Easting")
        axis.set_ylabel("Latitude" if crs and crs.is_geographic else "Northing")
        axis.set_aspect("equal")
        figure.tight_layout()
        st.pyplot(figure, clear_figure=True)
    finally:
        path.unlink(missing_ok=True)


def _render_range_shift(uploaded) -> None:
    path = _temporary_upload(uploaded, ".tif")
    try:
        with rasterio.open(path) as dataset:
            classes = dataset.read(1, masked=True)
            extent = plotting_extent(dataset)
            crs = dataset.crs
        valid_values = {int(value) for value in np.unique(classes.compressed())}
        unexpected = sorted(valid_values.difference(RANGE_SHIFT_LABELS))
        if unexpected:
            st.error(f"Unexpected RangeShift class codes: {unexpected}")
            return

        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        axis.imshow(
            classes,
            extent=extent,
            origin="upper",
            vmin=-0.5,
            vmax=3.5,
            cmap=ListedColormap(RANGE_SHIFT_COLORS),
            interpolation="nearest",
        )
        handles = [
            Patch(facecolor=RANGE_SHIFT_COLORS[code], label=label)
            for code, label in RANGE_SHIFT_LABELS.items()
        ]
        axis.legend(handles=handles, loc="lower left", title="Range transition")
        axis.set_title("Projected range shift")
        axis.set_xlabel("Longitude" if crs and crs.is_geographic else "Easting")
        axis.set_ylabel("Latitude" if crs and crs.is_geographic else "Northing")
        axis.set_aspect("equal")
        figure.tight_layout()
        st.pyplot(figure, clear_figure=True)
    finally:
        path.unlink(missing_ok=True)


def _summary_metrics(payload: dict) -> None:
    metrics = [
        ("Current suitable km²", "current_suitable_area_km2"),
        ("Future suitable km²", "future_suitable_area_km2"),
        ("Gained km²", "gained_area_km2"),
        ("Lost km²", "lost_area_km2"),
        ("Centroid shift km", "centroid_shift_km"),
        ("Jaccard overlap", "jaccard_overlap"),
    ]
    columns = st.columns(3)
    for position, (label, key) in enumerate(metrics):
        value = payload.get(key)
        display = "—" if value is None else f"{float(value):,.3f}"
        columns[position % 3].metric(label, display)
    st.json(payload)


def main() -> None:
    """Render the RangeShift interactive output explorer."""
    st.set_page_config(page_title="RangeShift AI Explorer", layout="wide")
    st.title("RangeShift AI Explorer")
    st.caption(
        "Inspect suitability surfaces, range-shift classifications, and summary outputs "
        "without hiding the model's ecological assumptions."
    )
    st.warning(
        "RangeShift outputs are scenario-conditioned habitat-suitability projections, "
        "not guaranteed future species distributions."
    )

    suitability_tab, shift_tab, summary_tab = st.tabs(
        ["Suitability raster", "Range-shift raster", "Summary JSON"]
    )

    with suitability_tab:
        suitability = st.file_uploader(
            "Upload a RangeShift suitability GeoTIFF",
            type=["tif", "tiff"],
            key="suitability",
        )
        if suitability is not None:
            _render_suitability(suitability)

    with shift_tab:
        transition = st.file_uploader(
            "Upload a RangeShift transition GeoTIFF",
            type=["tif", "tiff"],
            key="transition",
        )
        if transition is not None:
            _render_range_shift(transition)

    with summary_tab:
        summary = st.file_uploader(
            "Upload range_shift_summary.json or run_manifest.json",
            type=["json"],
            key="summary",
        )
        if summary is not None:
            try:
                payload = json.loads(summary.getvalue().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                st.error(f"Could not parse JSON: {exc}")
            else:
                if "outputs" in payload and "selected_threshold" in payload:
                    st.subheader("Run manifest")
                    st.metric("Selected threshold", f"{payload['selected_threshold']:.4f}")
                    st.json(payload)
                else:
                    st.subheader("Range-shift summary")
                    _summary_metrics(payload)


if __name__ == "__main__":
    main()
