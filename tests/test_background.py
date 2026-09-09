import numpy as np
import pandas as pd
import pytest

from rangeshift.background import generate_background_points


def _presences() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latitude": [39.0, 39.5, 40.0],
            "longitude": [-84.8, -84.3, -83.9],
        }
    )


def test_random_background_is_reproducible_and_labeled() -> None:
    first = generate_background_points(
        _presences(),
        12,
        method="random",
        min_distance_km=5.0,
        buffer_degrees=1.0,
        seed=7,
    )
    second = generate_background_points(
        _presences(),
        12,
        method="random",
        min_distance_km=5.0,
        buffer_degrees=1.0,
        seed=7,
    )
    pd.testing.assert_frame_equal(first.points, second.points)
    assert len(first.points) == 12
    assert set(first.points["presence"]) == {0}
    assert set(first.points["background_method"]) == {"random"}


def test_spatial_stratified_background_spans_multiple_cells() -> None:
    result = generate_background_points(
        _presences(),
        20,
        method="spatial_stratified",
        strata_size_degrees=0.5,
        buffer_degrees=1.5,
        seed=11,
    )
    lat_cells = np.floor(result.points["latitude"] / 0.5)
    lon_cells = np.floor(result.points["longitude"] / 0.5)
    assert pd.DataFrame({"lat": lat_cells, "lon": lon_cells}).drop_duplicates().shape[0] >= 8


def test_candidate_pool_background_samples_without_replacement() -> None:
    pool = pd.DataFrame(
        {
            "latitude": np.linspace(37.0, 42.0, 50),
            "longitude": np.linspace(-87.0, -82.0, 50),
        }
    )
    result = generate_background_points(
        _presences(),
        10,
        method="candidate_pool",
        candidate_pool=pool,
        bounds=(-87.0, 37.0, -82.0, 42.0),
        seed=5,
    )
    assert len(result.points) == 10
    assert not result.points.duplicated(["latitude", "longitude"]).any()


def test_candidate_pool_method_requires_pool() -> None:
    with pytest.raises(ValueError, match="candidate_pool"):
        generate_background_points(_presences(), 5, method="candidate_pool")
