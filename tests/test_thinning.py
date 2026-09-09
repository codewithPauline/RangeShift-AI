import pandas as pd

from rangeshift.thinning import thin_spatial_points


def test_spatial_thinning_respects_priority_and_distance() -> None:
    frame = pd.DataFrame(
        {
            "latitude": [39.0000, 39.0010, 40.0000],
            "longitude": [-84.0000, -84.0010, -85.0000],
            "quality": [10, 1, 5],
        },
        index=["best", "near_duplicate", "far"],
    )
    result = thin_spatial_points(
        frame,
        1.0,
        priority_column="quality",
        seed=3,
    )
    assert result.kept_indices == ["best", "far"]
    assert result.removed_indices == ["near_duplicate"]
    assert result.retained_count == 2
    assert result.original_count == 3
    assert result.retention_fraction == 2 / 3


def test_spatial_thinning_is_reproducible_without_priority() -> None:
    frame = pd.DataFrame(
        {
            "latitude": [39.0, 39.001, 39.002, 40.0],
            "longitude": [-84.0, -84.001, -84.002, -85.0],
        }
    )
    first = thin_spatial_points(frame, 1.0, seed=99)
    second = thin_spatial_points(frame, 1.0, seed=99)
    assert first.kept_indices == second.kept_indices
    pd.testing.assert_frame_equal(first.thinned_frame, second.thinned_frame)
