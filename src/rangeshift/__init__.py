"""RangeShift AI: machine-learning tools for habitat suitability and range-shift analysis."""

from .model import TrainingResult, train_habitat_model
from .spatial import (
    SpatialComparisonResult,
    SpatialEvaluationResult,
    assign_spatial_blocks,
    compare_random_and_spatial,
    evaluate_spatial_holdout,
)

__all__ = [
    "SpatialComparisonResult",
    "SpatialEvaluationResult",
    "TrainingResult",
    "assign_spatial_blocks",
    "compare_random_and_spatial",
    "evaluate_spatial_holdout",
    "train_habitat_model",
]
__version__ = "0.2.0"
