"""Demonstrate random versus spatial holdout evaluation with synthetic data."""

import numpy as np
import pandas as pd

from rangeshift.spatial import compare_random_and_spatial

rng = np.random.default_rng(7)
rows: list[dict[str, float | int]] = []

centers = [
    (32.0, -101.0),
    (32.0, -97.0),
    (36.0, -101.0),
    (36.0, -97.0),
    (40.0, -101.0),
    (40.0, -97.0),
    (44.0, -101.0),
    (44.0, -97.0),
]

for latitude, longitude in centers:
    local_effect = rng.normal(0, 0.7)
    for _ in range(30):
        bio1 = rng.normal(14 + local_effect, 2.0)
        bio12 = rng.normal(850 - 20 * local_effect, 120)
        elevation = rng.normal(250 + 30 * local_effect, 70)
        score = -0.7 * bio1 + 0.006 * bio12 + 0.004 * elevation + rng.normal(0, 1.2)
        presence = int(score > -4.0)
        rows.append(
            {
                "presence": presence,
                "bio1": bio1,
                "bio12": bio12,
                "elevation": elevation,
                "latitude": latitude + rng.normal(0, 0.15),
                "longitude": longitude + rng.normal(0, 0.15),
            }
        )

frame = pd.DataFrame(rows)
result = compare_random_and_spatial(
    frame,
    feature_columns=["bio1", "bio12", "elevation"],
    block_size_degrees=2.0,
    random_state=42,
)

print("Random holdout")
print(result.random.metrics)
print("\nSpatial holdout")
print(result.spatial.metrics)
print("\nSpatial minus random")
print(result.spatial_minus_random)
print(f"\nTrain blocks: {len(result.spatial.train_blocks)}")
print(f"Test blocks: {len(result.spatial.test_blocks)}")
