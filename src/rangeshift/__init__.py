"""RangeShift AI: machine-learning tools for habitat suitability and range-shift analysis."""

from .geospatial import ProjectedBlockResult, assign_projected_blocks, estimate_local_utm_epsg
from .model import TrainingResult, train_habitat_model
from .spatial import (
    SpatialComparisonResult,
    SpatialEvaluationResult,
    assign_spatial_blocks,
    compare_random_and_spatial,
    evaluate_spatial_holdout,
)
from .spatial_cv import SpatialCVResult, cross_validate_spatial_blocks, spatial_cross_validate

__all__ = [
    "ProjectedBlockResult",
    "SpatialCVResult",
    "SpatialComparisonResult",
    "SpatialEvaluationResult",
    "TrainingResult",
    "assign_projected_blocks",
    "assign_spatial_blocks",
    "compare_random_and_spatial",
    "cross_validate_spatial_blocks",
    "estimate_local_utm_epsg",
    "evaluate_spatial_holdout",
    "spatial_cross_validate",
    "train_habitat_model",
]
__version__ = "0.2.0"
