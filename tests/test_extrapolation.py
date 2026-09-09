from __future__ import annotations

import pandas as pd
import pytest

from rangeshift.extrapolation import diagnose_extrapolation


def test_extrapolation_flags_values_outside_training_envelope():
    training = pd.DataFrame(
        {
            "bio1": [10.0, 12.0, 14.0, 16.0],
            "bio12": [700.0, 800.0, 900.0, 1000.0],
        }
    )
    projection = pd.DataFrame(
        {
            "bio1": [11.0, 18.0, 9.0],
            "bio12": [850.0, 1050.0, 600.0],
        }
    )

    result = diagnose_extrapolation(training, projection, ["bio1", "bio12"])

    assert result.row_diagnostics.loc[0, "novel_feature_count"] == 0
    assert bool(result.row_diagnostics.loc[0, "requires_extrapolation"]) is False
    assert result.row_diagnostics.loc[1, "novel_feature_count"] == 2
    assert bool(result.row_diagnostics.loc[1, "requires_extrapolation"]) is True
    assert result.row_diagnostics.loc[2, "novel_feature_count"] == 2
    assert result.feature_summary["fraction_outside_training"].max() > 0.0


def test_extrapolation_rejects_missing_predictors():
    training = pd.DataFrame({"bio1": [10.0, 12.0]})
    projection = pd.DataFrame({"bio1": [11.0]})

    with pytest.raises(ValueError, match="missing"):
        diagnose_extrapolation(training, projection, ["bio1", "bio12"])
