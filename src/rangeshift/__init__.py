"""RangeShift AI: machine-learning tools for habitat suitability and range-shift analysis."""

from .geospatial import ProjectedBlockResult, assign_projected_blocks, estimate_local_utm_epsg
from .model import TrainingResult, train_habitat_model
from .raster import (
    RasterPredictionResult,
    RasterStack,
    parse_layer_specs,
    predict_suitability_raster,
)
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
    "RasterPredictionResult",
    "RasterStack",
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
    "parse_layer_specs",
    "predict_suitability_raster",
    "spatial_cross_validate",
    "train_habitat_model",
]
__version__ = "0.3.0"
