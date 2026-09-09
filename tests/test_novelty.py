import numpy as np
import pandas as pd

from rangeshift.novelty import diagnose_novel_climate


def test_novel_climate_combines_envelope_and_multivariate_distance() -> None:
    diagonal = np.linspace(0.0, 1.0, 30)
    training = pd.DataFrame({"bio1": diagonal, "bio12": diagonal})
    projection = pd.DataFrame(
        {
            "bio1": [0.5, 0.9, 1.5],
            "bio12": [0.5, 0.1, 1.5],
        },
        index=["supported", "multivariate_novel", "outside_envelope"],
    )

    result = diagnose_novel_climate(
        training,
        projection,
        ["bio1", "bio12"],
        distance_quantile=0.95,
    )
    diagnostics = result.row_diagnostics

    assert not bool(diagnostics.loc["supported", "novel_climate_warning"])
    assert bool(diagnostics.loc["multivariate_novel", "multivariate_novel"])
    assert not bool(diagnostics.loc["multivariate_novel", "requires_extrapolation"])
    assert bool(diagnostics.loc["outside_envelope", "requires_extrapolation"])
    assert bool(diagnostics.loc["outside_envelope", "novel_climate_warning"])
    assert result.multivariate_distance_threshold > 0.0
