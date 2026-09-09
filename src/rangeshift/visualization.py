"""Visualization utilities for RangeShift habitat-suitability rasters."""

from __future__ import annotations

from pathlib import Path


RANGE_SHIFT_LABELS = {
    0: "Stable unsuitable",
    1: "Lost suitable habitat",
    2: "Gained suitable habitat",
    3: "Stable suitable habitat",
}


def _require_plotting():
    """Import optional plotting dependencies with an actionable error."""
    try:
        import matplotlib.pyplot as plt
        import rasterio
        from rasterio.plot import plotting_extent
    except ImportError as exc:
        raise ImportError(
            "Raster map rendering requires Matplotlib and Rasterio. "
            "Install RangeShift with the 'geo' optional dependency."
        ) from exc
    return plt, rasterio, plotting_extent


def _overlay_boundary(axis, boundary_path: str | Path | None, raster_crs) -> None:
    if boundary_path is None:
        return

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "GeoPandas is required to overlay a vector study-area boundary."
        ) from exc

    boundary_path = Path(boundary_path)
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file does not exist: {boundary_path}")
    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        raise ValueError("Boundary data must define a coordinate reference system.")
    boundary.to_crs(raster_crs).boundary.plot(ax=axis, linewidth=0.9)


def _set_map_axes(axis, raster_crs) -> None:
    if raster_crs.is_geographic:
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
    else:
        axis.set_xlabel("Easting")
        axis.set_ylabel("Northing")
    axis.set_aspect("equal")


def plot_suitability_map(
    raster_path: str | Path,
    output_path: str | Path,
    *,
    title: str = "Predicted habitat suitability",
    boundary_path: str | Path | None = None,
    dpi: int = 300,
    cmap: str = "viridis",
) -> Path:
    """Render a high-resolution map from a RangeShift suitability GeoTIFF.

    The suitability scale is fixed from 0 to 1 so maps from different scenarios
    remain visually comparable. An optional vector study-area boundary is
    reprojected to the raster CRS before plotting.
    """
    plt, rasterio, plotting_extent = _require_plotting()

    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Suitability raster does not exist: {raster_path}")
    if dpi < 72:
        raise ValueError("dpi must be at least 72.")

    with rasterio.open(raster_path) as dataset:
        if dataset.count != 1:
            raise ValueError("Suitability map rendering expects a single-band raster.")
        if dataset.crs is None:
            raise ValueError("Suitability raster must define a coordinate reference system.")

        suitability = dataset.read(1, masked=True)
        extent = plotting_extent(dataset)
        raster_crs = dataset.crs

    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    image = axis.imshow(
        suitability,
        extent=extent,
        origin="upper",
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        interpolation="nearest",
    )

    _overlay_boundary(axis, boundary_path, raster_crs)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Predicted suitability")
    axis.set_title(title)
    _set_map_axes(axis, raster_crs)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_range_shift_map(
    raster_path: str | Path,
    output_path: str | Path,
    *,
    title: str = "Projected habitat range shift",
    boundary_path: str | Path | None = None,
    dpi: int = 300,
) -> Path:
    """Render the four-class RangeShift transition GeoTIFF."""
    try:
        import numpy as np
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError(
            "Range-shift map rendering requires NumPy and Matplotlib. "
            "Install RangeShift with the 'geo' optional dependency."
        ) from exc

    plt, rasterio, plotting_extent = _require_plotting()
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Range-shift raster does not exist: {raster_path}")
    if dpi < 72:
        raise ValueError("dpi must be at least 72.")

    with rasterio.open(raster_path) as dataset:
        if dataset.count != 1:
            raise ValueError("Range-shift map rendering expects a single-band raster.")
        if dataset.crs is None:
            raise ValueError("Range-shift raster must define a coordinate reference system.")

        classes = dataset.read(1, masked=True)
        valid_values = np.unique(classes.compressed())
        unexpected = [int(value) for value in valid_values if int(value) not in RANGE_SHIFT_LABELS]
        if unexpected:
            raise ValueError(f"Unexpected range-shift class codes: {unexpected}")
        extent = plotting_extent(dataset)
        raster_crs = dataset.crs

    colors = ["#e5e7eb", "#d55e00", "#0072b2", "#009e73"]
    colormap = ListedColormap(colors)

    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    axis.imshow(
        classes,
        extent=extent,
        origin="upper",
        vmin=-0.5,
        vmax=3.5,
        cmap=colormap,
        interpolation="nearest",
    )

    _overlay_boundary(axis, boundary_path, raster_crs)

    legend_handles = [
        Patch(facecolor=colors[code], label=label)
        for code, label in RANGE_SHIFT_LABELS.items()
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        title="Range transition",
    )
    axis.set_title(title)
    _set_map_axes(axis, raster_crs)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path
