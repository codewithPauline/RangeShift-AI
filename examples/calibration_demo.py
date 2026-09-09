"""Demonstrate leakage-aware calibration and threshold selection in RangeShift AI."""

import numpy as np
import pandas as pd

from rangeshift.calibration import train_calibrated_habitat_model


def main() -> None:
    """Train a calibrated model and report validation-selected threshold performance."""
    rng = np.random.default_rng(105)
    n = 500
    bio1 = rng.normal(14.0, 3.0, n)
    bio12 = rng.normal(900.0, 180.0, n)
    elevation = rng.normal(280.0, 130.0, n)

    signal = (
        -((bio1 - 13.5) ** 2) / 8.0
        + (bio12 - 850.0) / 250.0
        + (elevation - 250.0) / 500.0
        + rng.normal(0.0, 0.8, n)
    )
    presence = (signal > np.median(signal)).astype(int)
    frame = pd.DataFrame(
        {
            "presence": presence,
            "bio1": bio1,
            "bio12": bio12,
            "elevation": elevation,
        }
    )

    result = train_calibrated_habitat_model(
        frame,
        feature_columns=["bio1", "bio12", "elevation"],
        calibration_method="sigmoid",
        threshold_method="tss",
        random_state=105,
    )

    print("Split sizes:", result.split_sizes)
    print("Selected validation threshold:", result.selected_threshold)
    print("Validation threshold metrics:", result.validation_threshold_metrics)
    print("Uncalibrated test probability metrics:")
    print(result.uncalibrated_test_probability_metrics)
    print("Calibrated test probability metrics:")
    print(result.test_probability_metrics)
    print("Final test classification metrics:")
    print(result.test_classification_metrics)


if __name__ == "__main__":
    main()
