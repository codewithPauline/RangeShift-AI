"""RangeShift AI: machine-learning tools for habitat suitability and range-shift analysis."""

from .bias import SamplingBiasResult, diagnose_sampling_bias
from .calibration import (
    CalibrationResult,
    save_calibrated_model_bundle,
    train_calibrated_habitat_model,
)
from .explainability import (
    partial_dependence_table,
    plot_response_curves,
    shap_importance_table,
)
from .extrapolation import ExtrapolationResult, diagnose_extrapolation
from .geospatial import ProjectedBlockResult, assign_projected_blocks, estimate_local_utm_epsg
from .model import TrainingResult, train_habitat_model
from .model_selection import (
    ModelSelectionResult,
    save_selected_model_bundle,
    tune_and_compare_models,
)
from .range_shift import RangeShiftResult, compare_suitability_rasters
from .raster import (
    RasterPredictionResult,
    RasterStack,
    parse_layer_specs,
    predict_suitability_raster,
    predict_suitability_raster_windowed,
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
from .visualization import plot_range_shift_map, plot_spatial_split, plot_suitability_map

__all__ = [
    "CalibrationResult",
    "ExtrapolationResult",
    "ModelSelectionResult",
    "ProjectedBlockResult",
    "RangeShiftResult",
    "RasterPredictionResult",
    "RasterStack",
    "SamplingBiasResult",
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
    "diagnose_extrapolation",
    "diagnose_sampling_bias",
    "estimate_local_utm_epsg",
    "evaluate_spatial_holdout",
    "evaluate_thresholds",
    "parse_layer_specs",
    "partial_dependence_table",
    "plot_range_shift_map",
    "plot_response_curves",
    "plot_spatial_split",
    "plot_suitability_map",
    "predict_suitability_raster",
    "predict_suitability_raster_windowed",
    "save_calibrated_model_bundle",
    "save_selected_model_bundle",
    "select_threshold",
    "shap_importance_table",
    "spatial_cross_validate",
    "train_calibrated_habitat_model",
    "train_habitat_model",
    "tune_and_compare_models",
]
__version__ = "0.5.0"
