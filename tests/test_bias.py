from __future__ import annotations

import pandas as pd
import pytest

from rangeshift.bias import diagnose_sampling_bias


def test_sampling_bias_reports_clustering_and_duplicates():
    frame = pd.DataFrame(
        {
            "latitude": [39.0, 39.0, 39.1, 39.2, 41.0, 43.0],
            "longitude": [-84.0, -84.0, -84.1, -84.2, -82.0, -80.0],
        }
    )

    result = diagnose_sampling_bias(frame, block_size_degrees=1.0)

    assert result.metrics["observations"] == 6.0
    assert result.metrics["occupied_blocks"] >= 3.0
    assert result.metrics["duplicate_coordinate_fraction"] == pytest.approx(1 / 6)
    assert 0.0 < result.metrics["max_block_fraction"] <= 1.0
    assert result.metrics["median_nearest_neighbor_km"] >= 0.0
    assert result.block_counts["fraction"].sum() == pytest.approx(1.0)
    assert len(result.nearest_neighbor_km) == len(frame)


def test_sampling_bias_requires_multiple_observations():
    frame = pd.DataFrame({"latitude": [39.0], "longitude": [-84.0]})

    with pytest.raises(ValueError, match="at least two observations"):
        diagnose_sampling_bias(frame)
