"""RangeShift AI: machine-learning tools for habitat suitability and range-shift analysis."""

from .calibration import (
    CalibrationResult,
    save_calibrated_model_bundle,
    train_calibrated_habitat_model,
)
from .geospatial import ProjectedBlockResult, assign_projected_blocks, estimate_local_utm_epsg
from .model import TrainingResult, train_habitat_model
from .range_shift import RangeShiftResult, compare_suitability_rasters
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
from .threshold import ThresholdSelectionResult, evaluate_thresholds, select_threshold
from .visualization import plot_range_shift_map, plot_suitability_map

__all__ = [
    "CalibrationResult",
    "ProjectedBlockResult",
    "RangeShiftResult",
    "RasterPredictionResult",
    "RasterStack",
    "SpatialCVResult",
    "SpatialComparisonResult",
    "SpatialEvaluationResult",
    "ThresholdSelectionResult",
    "TrainingResult",
    "assign_projected_blocks",
    "assign_spatial_blocks",
    "compare_random_and_spatial",
    "compare_suitability_rasters",
    "cross_validate_spatial_blocks",
    "estimate_local_utm_epsg",
    "evaluate_spatial_holdout",
    "evaluate_thresholds",
    "parse_layer_specs",
    "plot_range_shift_map",
    "plot_suitability_map",
    "predict_suitability_raster",
    "save_calibrated_model_bundle",
    "select_threshold",
    "spatial_cross_validate",
    "train_calibrated_habitat_model",
    "train_habitat_model",
]
__version__ = "0.5.0"
