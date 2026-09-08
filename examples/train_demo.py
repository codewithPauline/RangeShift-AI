"""Run a small synthetic RangeShift AI training demonstration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rangeshift.model import train_habitat_model


def make_demo_data(n: int = 300, seed: int = 13) -> pd.DataFrame:
    """Create synthetic environmental data with a simple habitat signal."""
    rng = np.random.default_rng(seed)

    temperature = rng.normal(15.0, 4.5, n)
    precipitation = rng.normal(900.0, 200.0, n)
    elevation = rng.uniform(25.0, 900.0, n)

    signal = (
        -0.4 * np.abs(temperature - 13.5)
        + 0.0035 * (precipitation - 800.0)
        + 0.0012 * elevation
        + rng.normal(0.0, 0.8, n)
    )
    presence = (signal > np.median(signal)).astype(int)

    return pd.DataFrame(
        {
            "presence": presence,
            "bio1": temperature,
            "bio12": precipitation,
            "elevation": elevation,
        }
    )


def main() -> None:
    frame = make_demo_data()
    result = train_habitat_model(
        frame,
        feature_columns=["bio1", "bio12", "elevation"],
    )

    print("RangeShift AI synthetic demo")
    print("\nMetrics")
    for metric, value in result.metrics.items():
        print(f"{metric:>10}: {value:.3f}")

    print("\nFeature importance")
    print(result.feature_importance.to_string(index=False))


if __name__ == "__main__":
    main()
