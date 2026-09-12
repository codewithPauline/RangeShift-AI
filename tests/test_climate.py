from pathlib import Path

import numpy as np
import pytest

from rangeshift.climate import (
    align_raster_band_to_reference,
    crop_raster_band_to_bounds,
    extract_bioclim_bands_to_reference,
    validate_aligned_rasters,
    worldclim_cmip6_bioc_url,
)


def test_worldclim_cmip6_bioc_url_matches_official_10m_pattern() -> None:
    assert worldclim_cmip6_bioc_url("MIROC6", "ssp245", "2021-2040") == (
        "https://geodata.ucdavis.edu/cmip6/10m/MIROC6/ssp245/"
        "wc2.1_10m_bioc_MIROC6_ssp245_2021-2040.tif"
    )

    with pytest.raises(ValueError, match="ssp must be one of"):
        worldclim_cmip6_bioc_url("MIROC6", "ssp999", "2021-2040")
    with pytest.raises(ValueError, match="period must be one of"):
        worldclim_cmip6_bioc_url("MIROC6", "ssp245", "2050-2070")
    with pytest.raises(ValueError, match="resolution='10m'"):
        worldclim_cmip6_bioc_url(
            "MIROC6", "ssp245", "2021-2040", resolution="2.5m"
        )


def test_crop_and_alignment_create_exact_shared_grid(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    source_path = tmp_path / "current_global.tif"
    future_path = tmp_path / "future_multiband.tif"
    transform = from_origin(-90.0, 45.0, 1.0, 1.0)

    current = np.arange(36, dtype=np.float32).reshape(6, 6)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        height=6,
        width=6,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(current, 1)

    future = np.stack(
        [
            current + 1,
            current + 10,
            current + 20,
        ]
    )
    with rasterio.open(
        future_path,
        "w",
        driver="GTiff",
        height=6,
        width=6,
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(future)

    reference = crop_raster_band_to_bounds(
        source_path,
        tmp_path / "reference.tif",
        bounds=(-89.0, 40.0, -85.0, 44.0),
    )
    aligned = align_raster_band_to_reference(
        future_path,
        reference,
        tmp_path / "future_band2.tif",
        band=2,
    )
    outputs = extract_bioclim_bands_to_reference(
        future_path,
        reference,
        tmp_path / "selected",
        features={"bio1": 1, "bio12": 3},
    )

    validate_aligned_rasters(
        {
            "reference": reference,
            "aligned": aligned,
            **outputs,
        }
    )

    with rasterio.open(reference) as ref, rasterio.open(aligned) as aligned_ds:
        assert aligned_ds.shape == ref.shape
        assert aligned_ds.transform == ref.transform
        assert aligned_ds.crs == ref.crs
        assert aligned_ds.count == 1


def test_validate_aligned_rasters_rejects_mismatch(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    paths = []
    for index, pixel_size in enumerate((1.0, 2.0)):
        path = tmp_path / f"raster_{index}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(-90.0, 45.0, pixel_size, pixel_size),
        ) as dataset:
            dataset.write(np.ones((2, 2), dtype=np.float32), 1)
        paths.append(path)

    with pytest.raises(ValueError, match="not aligned"):
        validate_aligned_rasters({"a": paths[0], "b": paths[1]})
