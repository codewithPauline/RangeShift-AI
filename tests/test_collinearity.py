import numpy as np
import pandas as pd

from rangeshift.collinearity import diagnose_collinearity


def test_collinearity_flags_correlated_predictors_and_vif() -> None:
    rng = np.random.default_rng(4)
    x = np.linspace(-2.0, 2.0, 120)
    y = 2.0 * x + rng.normal(0.0, 0.02, len(x))
    z = rng.normal(0.0, 1.0, len(x))
    frame = pd.DataFrame({"bio1": x, "bio2": y, "bio3": z})

    result = diagnose_collinearity(
        frame,
        ["bio1", "bio2", "bio3"],
        correlation_threshold=0.8,
        vif_threshold=5.0,
    )

    pairs = {
        frozenset((row.feature_a, row.feature_b))
        for row in result.high_correlation_pairs.itertuples()
    }
    assert frozenset(("bio1", "bio2")) in pairs
    assert {"bio1", "bio2"}.issubset(result.flagged_features)
    vif = result.vif_table.set_index("feature")["vif"]
    assert vif["bio1"] > 5.0
    assert vif["bio2"] > 5.0


def test_collinearity_allows_uncorrelated_predictors_without_pair_flags() -> None:
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=200),
            "b": rng.normal(size=200),
            "c": rng.normal(size=200),
        }
    )
    result = diagnose_collinearity(frame, ["a", "b", "c"], correlation_threshold=0.95)
    assert result.high_correlation_pairs.empty
