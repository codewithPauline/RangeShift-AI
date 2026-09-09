from __future__ import annotations

import pandas as pd
import pytest

from rangeshift.visualization import plot_spatial_split


def test_plot_spatial_split_writes_image(tmp_path):
    pytest.importorskip("matplotlib")
    frame = pd.DataFrame(
        {
            "latitude": [38.0, 38.5, 39.0, 39.5, 40.0, 40.5],
            "longitude": [-86.0, -85.5, -85.0, -84.5, -84.0, -83.5],
        }
    )
    output = tmp_path / "spatial_split.png"

    result = plot_spatial_split(
        frame,
        train_indices=[0, 1, 2, 3],
        test_indices=[4, 5],
        output_path=output,
        block_size_degrees=1.0,
        dpi=100,
    )

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_spatial_split_rejects_overlapping_indices(tmp_path):
    pytest.importorskip("matplotlib")
    frame = pd.DataFrame(
        {
            "latitude": [38.0, 39.0, 40.0],
            "longitude": [-86.0, -85.0, -84.0],
        }
    )

    with pytest.raises(ValueError, match="must not overlap"):
        plot_spatial_split(
            frame,
            train_indices=[0, 1],
            test_indices=[1, 2],
            output_path=tmp_path / "bad.png",
        )
