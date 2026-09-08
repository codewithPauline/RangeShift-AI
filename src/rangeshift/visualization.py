"""Visualization utilities for RangeShift habitat-suitability rasters."""

from __future__ import annotations

from pathlib import Path


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

    if boundary_path is not None:
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

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Predicted suitability")
    axis.set_title(title)

    if raster_crs.is_geographic:
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
    else:
        axis.set_xlabel("Easting")
        axis.set_ylabel("Northing")

    axis.set_aspect("equal")
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path
