"""Command-line interface for RangeShift AI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .bias import diagnose_sampling_bias
from .calibration import save_calibrated_model_bundle, train_calibrated_habitat_model
from .data import load_occurrence_table
from .explainability import (
    partial_dependence_table,
    plot_response_curves,
    shap_importance_table,
)
from .extrapolation import diagnose_extrapolation
from .geospatial import assign_projected_blocks
from .model import save_model_bundle, train_habitat_model
from .model_selection import save_selected_model_bundle, tune_and_compare_models
from .prediction import load_model_bundle, predict_suitability
from .range_shift import compare_suitability_rasters
from .raster import (
    parse_layer_specs,
    predict_suitability_raster,
    predict_suitability_raster_windowed,
)
from .spatial import assign_spatial_blocks, compare_random_and_spatial
from .spatial_cv import spatial_cross_validate
from .threshold import select_threshold
from .visualization import (
    plot_range_shift_map,
    plot_spatial_split,
    plot_suitability_map,
)


def _add_feature_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", default="presence")
    parser.add_argument("--features", nargs="+", required=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the RangeShift command-line parser."""
    parser = argparse.ArgumentParser(
        prog="rangeshift",
        description="Machine-learning tools for habitat suitability and range-shift analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a baseline habitat-suitability model.")
    train.add_argument("input_csv", type=Path)
    _add_feature_arguments(train)
    train.add_argument("--output", type=Path, default=Path("rangeshift_model.joblib"))
    train.add_argument("--test-size", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--trees", type=int, default=300)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Train a calibrated model and select a threshold on validation data.",
    )
    calibrate.add_argument("input_csv", type=Path)
    _add_feature_arguments(calibrate)
    calibrate.add_argument("--validation-size", type=float, default=0.20)
    calibrate.add_argument("--test-size", type=float, default=0.20)
    calibrate.add_argument("--seed", type=int, default=42)
    calibrate.add_argument("--trees", type=int, default=300)
    calibrate.add_argument(
        "--calibration-method",
        choices=("sigmoid", "isotonic"),
        default="sigmoid",
    )
    calibrate.add_argument("--calibration-cv", type=int, default=5)
    calibrate.add_argument(
        "--threshold-method",
        choices=("tss", "youden_j", "f1", "balanced_accuracy"),
        default="tss",
    )
    calibrate.add_argument("--calibration-bins", type=int, default=10)
    calibrate.add_argument(
        "--output",
        type=Path,
        default=Path("rangeshift_calibrated.joblib"),
    )
    calibrate.add_argument(
        "--threshold-output",
        type=Path,
        default=Path("threshold_diagnostics.csv"),
    )
    calibrate.add_argument(
        "--calibration-output",
        type=Path,
        default=Path("calibration_diagnostics.csv"),
    )
    calibrate.add_argument(
        "--summary-output",
        type=Path,
        default=Path("calibration_summary.json"),
    )

    threshold = subparsers.add_parser(
        "select-threshold",
        help="Select a suitability threshold from validation labels and probabilities.",
    )
    threshold.add_argument("input_csv", type=Path)
    threshold.add_argument("--target", default="presence")
    threshold.add_argument("--probability", default="suitability")
    threshold.add_argument(
        "--method",
        choices=("tss", "youden_j", "f1", "balanced_accuracy"),
        default="tss",
    )
    threshold.add_argument(
        "--output",
        type=Path,
        default=Path("threshold_diagnostics.csv"),
    )

    tuning = subparsers.add_parser(
        "tune-models",
        help="Tune and compare Random Forest and Gradient Boosting models.",
    )
    tuning.add_argument("input_csv", type=Path)
    _add_feature_arguments(tuning)
    tuning.add_argument("--splits", type=int, default=5)
    tuning.add_argument("--seed", type=int, default=42)
    tuning.add_argument(
        "--scoring",
        choices=("roc_auc", "balanced_accuracy", "f1"),
        default="roc_auc",
    )
    tuning.add_argument("--spatial-groups", action="store_true")
    tuning.add_argument("--latitude", default="latitude")
    tuning.add_argument("--longitude", default="longitude")
    tuning.add_argument("--block-size", type=float, default=1.0)
    tuning.add_argument(
        "--output",
        type=Path,
        default=Path("rangeshift_selected_model.joblib"),
    )
    tuning.add_argument(
        "--results-output",
        type=Path,
        default=Path("model_tuning_results.csv"),
    )
    tuning.add_argument(
        "--summary-output",
        type=Path,
        default=Path("model_tuning_summary.json"),
    )

    bias = subparsers.add_parser(
        "diagnose-bias",
        help="Quantify spatial clustering and duplicate-coordinate sampling bias.",
    )
    bias.add_argument("input_csv", type=Path)
    bias.add_argument("--latitude", default="latitude")
    bias.add_argument("--longitude", default="longitude")
    bias.add_argument("--block-size", type=float, default=1.0)
    bias.add_argument(
        "--summary-output",
        type=Path,
        default=Path("sampling_bias_summary.json"),
    )
    bias.add_argument(
        "--blocks-output",
        type=Path,
        default=Path("sampling_bias_blocks.csv"),
    )
    bias.add_argument(
        "--nearest-output",
        type=Path,
        default=Path("nearest_neighbor_distances.csv"),
    )

    extrapolation = subparsers.add_parser(
        "diagnose-extrapolation",
        help="Flag projection environments outside the training predictor envelope.",
    )
    extrapolation.add_argument("training_csv", type=Path)
    extrapolation.add_argument("projection_csv", type=Path)
    extrapolation.add_argument("--features", nargs="+", required=True)
    extrapolation.add_argument(
        "--row-output",
        type=Path,
        default=Path("extrapolation_rows.csv"),
    )
    extrapolation.add_argument(
        "--feature-output",
        type=Path,
        default=Path("extrapolation_features.csv"),
    )

    response = subparsers.add_parser(
        "response-curves",
        help="Compute and render partial-dependence response curves.",
    )
    response.add_argument("model", type=Path)
    response.add_argument("input_csv", type=Path)
    response.add_argument("--grid-resolution", type=int, default=25)
    response.add_argument(
        "--table-output",
        type=Path,
        default=Path("partial_dependence.csv"),
    )
    response.add_argument(
        "--plot-output",
        type=Path,
        default=Path("response_curves.png"),
    )
    response.add_argument("--dpi", type=int, default=300)

    shap_parser = subparsers.add_parser(
        "shap-importance",
        help="Calculate mean absolute Tree SHAP feature attributions.",
    )
    shap_parser.add_argument("model", type=Path)
    shap_parser.add_argument("input_csv", type=Path)
    shap_parser.add_argument("--max-samples", type=int, default=500)
    shap_parser.add_argument(
        "--output",
        type=Path,
        default=Path("shap_importance.csv"),
    )

    predict = subparsers.add_parser(
        "predict",
        help="Predict suitability from a saved RangeShift model.",
    )
    predict.add_argument("model", type=Path)
    predict.add_argument("input_csv", type=Path)
    predict.add_argument(
        "--output",
        type=Path,
        default=Path("suitability_predictions.csv"),
    )

    raster = subparsers.add_parser(
        "predict-raster",
        help="Predict a habitat-suitability GeoTIFF from aligned environmental rasters.",
    )
    raster.add_argument("model", type=Path)
    raster.add_argument(
        "--layer",
        action="append",
        required=True,
        metavar="FEATURE=PATH",
        help="Predictor raster mapping; repeat once for each trained model feature.",
    )
    raster.add_argument(
        "--output",
        type=Path,
        default=Path("habitat_suitability.tif"),
    )
    raster.add_argument("--nodata", type=float, default=-9999.0)
    raster.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="Use bounded-memory windows of this pixel size; 0 uses in-memory prediction.",
    )

    plot_raster = subparsers.add_parser(
        "plot-raster",
        help="Render a high-resolution map from a suitability GeoTIFF.",
    )
    plot_raster.add_argument("input_raster", type=Path)
    plot_raster.add_argument(
        "--output",
        type=Path,
        default=Path("habitat_suitability.png"),
    )
    plot_raster.add_argument("--title", default="Predicted habitat suitability")
    plot_raster.add_argument("--boundary", type=Path, default=None)
    plot_raster.add_argument("--dpi", type=int, default=300)
    plot_raster.add_argument("--cmap", default="viridis")

    range_shift = subparsers.add_parser(
        "range-shift",
        help="Compare aligned current and future suitability rasters.",
    )
    range_shift.add_argument("current_raster", type=Path)
    range_shift.add_argument("future_raster", type=Path)
    range_shift.add_argument("--threshold", type=float, required=True)
    range_shift.add_argument(
        "--classes-output",
        type=Path,
        default=Path("range_shift_classes.tif"),
    )
    range_shift.add_argument("--difference-output", type=Path, default=None)
    range_shift.add_argument(
        "--summary-output",
        type=Path,
        default=Path("range_shift_summary.json"),
    )

    plot_shift = subparsers.add_parser(
        "plot-range-shift",
        help="Render the four-class range-shift transition GeoTIFF.",
    )
    plot_shift.add_argument("input_raster", type=Path)
    plot_shift.add_argument(
        "--output",
        type=Path,
        default=Path("range_shift_map.png"),
    )
    plot_shift.add_argument("--title", default="Projected habitat range shift")
    plot_shift.add_argument("--boundary", type=Path, default=None)
    plot_shift.add_argument("--dpi", type=int, default=300)

    spatial = subparsers.add_parser(
        "compare-spatial",
        help="Compare random holdout with spatial block holdout performance.",
    )
    spatial.add_argument("input_csv", type=Path)
    _add_feature_arguments(spatial)
    spatial.add_argument("--latitude", default="latitude")
    spatial.add_argument("--longitude", default="longitude")
    spatial.add_argument("--block-size", type=float, default=1.0)
    spatial.add_argument("--test-size", type=float, default=0.25)
    spatial.add_argument("--seed", type=int, default=42)
    spatial.add_argument("--trees", type=int, default=300)
    spatial.add_argument("--output", type=Path, default=None)
    spatial.add_argument("--plot-output", type=Path, default=None)

    spatial_cv = subparsers.add_parser(
        "spatial-cv",
        help="Run repeated spatial block holdouts and summarize model performance.",
    )
    spatial_cv.add_argument("input_csv", type=Path)
    _add_feature_arguments(spatial_cv)
    spatial_cv.add_argument("--latitude", default="latitude")
    spatial_cv.add_argument("--longitude", default="longitude")
    spatial_cv.add_argument("--block-size", type=float, default=1.0)
    spatial_cv.add_argument("--splits", type=int, default=5)
    spatial_cv.add_argument("--test-size", type=float, default=0.25)
    spatial_cv.add_argument("--seed", type=int, default=42)
    spatial_cv.add_argument("--trees", type=int, default=300)
    spatial_cv.add_argument("--output", type=Path, default=None)

    projected = subparsers.add_parser(
        "project-blocks",
        help="Create kilometer-scale projected spatial blocks for coordinate data.",
    )
    projected.add_argument("input_csv", type=Path)
    projected.add_argument("--latitude", default="latitude")
    projected.add_argument("--longitude", default="longitude")
    projected.add_argument("--block-km", type=float, default=100.0)
    projected.add_argument("--epsg", type=int, default=None)
    projected.add_argument(
        "--output",
        type=Path,
        default=Path("projected_blocks.csv"),
    )
    return parser


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Run the RangeShift command-line interface."""
    args = build_parser().parse_args()

    if args.command == "train":
        frame = load_occurrence_table(args.input_csv)
        result = train_habitat_model(
            frame,
            feature_columns=args.features,
            target_column=args.target,
            test_size=args.test_size,
            random_state=args.seed,
            n_estimators=args.trees,
        )
        saved_path = save_model_bundle(result, args.output)
        print(json.dumps(result.metrics, indent=2, sort_keys=True))
        print("\nFeature importance")
        print(result.feature_importance.to_string(index=False))
        print(f"\nSaved model: {saved_path}")
        return

    if args.command == "calibrate":
        frame = load_occurrence_table(args.input_csv)
        result = train_calibrated_habitat_model(
            frame,
            feature_columns=args.features,
            target_column=args.target,
            validation_size=args.validation_size,
            test_size=args.test_size,
            random_state=args.seed,
            n_estimators=args.trees,
            calibration_method=args.calibration_method,
            calibration_cv=args.calibration_cv,
            threshold_method=args.threshold_method,
            calibration_bins=args.calibration_bins,
        )
        saved_path = save_calibrated_model_bundle(result, args.output)
        args.threshold_output.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
        result.threshold_table.to_csv(args.threshold_output, index=False)
        result.calibration_table.to_csv(args.calibration_output, index=False)
        payload = {
            "selected_threshold": result.selected_threshold,
            "threshold_method": result.threshold_method,
            "calibration_method": result.calibration_method,
            "split_sizes": result.split_sizes,
            "validation_threshold_metrics": result.validation_threshold_metrics,
            "test_probability_metrics": result.test_probability_metrics,
            "uncalibrated_test_probability_metrics": (
                result.uncalibrated_test_probability_metrics
            ),
            "test_classification_metrics": result.test_classification_metrics,
            "model": str(saved_path),
        }
        _write_json(args.summary_output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "select-threshold":
        frame = pd.read_csv(args.input_csv)
        missing = [column for column in (args.target, args.probability) if column not in frame]
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
        result = select_threshold(
            frame[args.target],
            frame[args.probability],
            method=args.method,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.table.to_csv(args.output, index=False)
        print(
            json.dumps(
                {
                    "selected_threshold": result.threshold,
                    "method": result.method,
                    "score": result.score,
                    "metrics": result.metrics,
                    "diagnostics": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "tune-models":
        frame = load_occurrence_table(args.input_csv)
        groups = None
        if args.spatial_groups:
            groups = assign_spatial_blocks(
                frame,
                latitude_column=args.latitude,
                longitude_column=args.longitude,
                block_size_degrees=args.block_size,
            )
        result = tune_and_compare_models(
            frame,
            feature_columns=args.features,
            target_column=args.target,
            groups=groups,
            n_splits=args.splits,
            random_state=args.seed,
            scoring=args.scoring,
        )
        saved_path = save_selected_model_bundle(result, args.output)
        args.results_output.parent.mkdir(parents=True, exist_ok=True)
        result.cv_results.to_csv(args.results_output, index=False)
        payload = {
            "best_model": result.best_model_name,
            "best_score": result.best_score,
            "best_params": result.best_params,
            "scoring": result.scoring,
            "spatial_groups_used": result.spatial_groups_used,
            "model": str(saved_path),
            "results": str(args.results_output),
        }
        _write_json(args.summary_output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "diagnose-bias":
        frame = pd.read_csv(args.input_csv)
        result = diagnose_sampling_bias(
            frame,
            latitude_column=args.latitude,
            longitude_column=args.longitude,
            block_size_degrees=args.block_size,
        )
        args.blocks_output.parent.mkdir(parents=True, exist_ok=True)
        args.nearest_output.parent.mkdir(parents=True, exist_ok=True)
        result.block_counts.to_csv(args.blocks_output, index=False)
        result.nearest_neighbor_km.to_frame().to_csv(args.nearest_output, index=True)
        _write_json(args.summary_output, result.metrics)
        print(json.dumps(result.metrics, indent=2, sort_keys=True))
        return

    if args.command == "diagnose-extrapolation":
        training = pd.read_csv(args.training_csv)
        projection = pd.read_csv(args.projection_csv)
        result = diagnose_extrapolation(training, projection, args.features)
        args.row_output.parent.mkdir(parents=True, exist_ok=True)
        args.feature_output.parent.mkdir(parents=True, exist_ok=True)
        result.row_diagnostics.to_csv(args.row_output, index=True)
        result.feature_summary.to_csv(args.feature_output, index=False)
        fraction = float(result.row_diagnostics["requires_extrapolation"].mean())
        print(
            json.dumps(
                {
                    "projection_rows": int(len(projection)),
                    "rows_requiring_extrapolation": int(
                        result.row_diagnostics["requires_extrapolation"].sum()
                    ),
                    "fraction_requiring_extrapolation": fraction,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "response-curves":
        frame = pd.read_csv(args.input_csv)
        bundle = load_model_bundle(args.model)
        table = partial_dependence_table(
            bundle["model"],
            frame,
            bundle["feature_columns"],
            grid_resolution=args.grid_resolution,
        )
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.table_output, index=False)
        output = plot_response_curves(table, args.plot_output, dpi=args.dpi)
        print(f"Saved response table: {args.table_output}")
        print(f"Saved response curves: {output}")
        return

    if args.command == "shap-importance":
        frame = pd.read_csv(args.input_csv)
        bundle = load_model_bundle(args.model)
        table = shap_importance_table(
            bundle["model"],
            frame,
            bundle["feature_columns"],
            max_samples=args.max_samples,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(table.to_string(index=False))
        print(f"\nSaved SHAP importance: {args.output}")
        return

    if args.command == "predict":
        frame = pd.read_csv(args.input_csv)
        bundle = load_model_bundle(args.model)
        output = frame.copy()
        output["suitability"] = predict_suitability(frame, bundle)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print(f"Saved predictions: {args.output}")
        return

    if args.command == "predict-raster":
        bundle = load_model_bundle(args.model)
        layers = parse_layer_specs(args.layer)
        if args.window_size > 0:
            result = predict_suitability_raster_windowed(
                bundle,
                layers,
                args.output,
                nodata=args.nodata,
                window_size=args.window_size,
            )
            engine = "windowed"
        else:
            result = predict_suitability_raster(
                bundle,
                layers,
                args.output,
                nodata=args.nodata,
            )
            engine = "in_memory"
        print(
            json.dumps(
                {
                    "output": str(result.output_path),
                    "valid_cells": result.valid_cells,
                    "total_cells": result.total_cells,
                    "features": result.feature_columns,
                    "crs": result.crs,
                    "engine": engine,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "plot-raster":
        output = plot_suitability_map(
            args.input_raster,
            args.output,
            title=args.title,
            boundary_path=args.boundary,
            dpi=args.dpi,
            cmap=args.cmap,
        )
        print(f"Saved suitability map: {output}")
        return

    if args.command == "range-shift":
        result = compare_suitability_rasters(
            args.current_raster,
            args.future_raster,
            args.classes_output,
            threshold=args.threshold,
            difference_output_path=args.difference_output,
        )
        payload = result.to_dict()
        _write_json(args.summary_output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "plot-range-shift":
        output = plot_range_shift_map(
            args.input_raster,
            args.output,
            title=args.title,
            boundary_path=args.boundary,
            dpi=args.dpi,
        )
        print(f"Saved range-shift map: {output}")
        return

    if args.command == "compare-spatial":
        frame = load_occurrence_table(args.input_csv)
        result = compare_random_and_spatial(
            frame,
            feature_columns=args.features,
            target_column=args.target,
            latitude_column=args.latitude,
            longitude_column=args.longitude,
            block_size_degrees=args.block_size,
            test_size=args.test_size,
            random_state=args.seed,
            n_estimators=args.trees,
        )
        payload = {
            "random": result.random.metrics,
            "spatial": result.spatial.metrics,
            "spatial_minus_random": result.spatial_minus_random,
            "spatial_train_blocks": len(result.spatial.train_blocks),
            "spatial_test_blocks": len(result.spatial.test_blocks),
            "block_size_degrees": result.spatial.block_size_degrees,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.output is not None:
            _write_json(args.output, payload)
        if args.plot_output is not None:
            plot_spatial_split(
                frame,
                result.spatial.train_indices,
                result.spatial.test_indices,
                args.plot_output,
                latitude_column=args.latitude,
                longitude_column=args.longitude,
                block_size_degrees=args.block_size,
            )
            print(f"Saved spatial split map: {args.plot_output}")
        return

    if args.command == "spatial-cv":
        frame = load_occurrence_table(args.input_csv)
        result = spatial_cross_validate(
            frame,
            feature_columns=args.features,
            target_column=args.target,
            latitude_column=args.latitude,
            longitude_column=args.longitude,
            block_size_degrees=args.block_size,
            n_splits=args.splits,
            test_size=args.test_size,
            random_state=args.seed,
            n_estimators=args.trees,
        )
        payload = {
            "mean": result.mean_metrics,
            "std": result.std_metrics,
            "requested_splits": result.requested_splits,
            "valid_splits": result.valid_splits,
            "occupied_blocks": result.occupied_blocks,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.fold_metrics.to_csv(args.output, index=False)
        return

    if args.command == "project-blocks":
        frame = pd.read_csv(args.input_csv)
        result = assign_projected_blocks(
            frame,
            latitude_column=args.latitude,
            longitude_column=args.longitude,
            block_size_km=args.block_km,
            crs_epsg=args.epsg,
        )
        output = frame.copy()
        output["spatial_block"] = result.blocks
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print(f"Projected CRS: EPSG:{result.crs_epsg}")
        print(f"Block size: {result.block_size_km} km")
        print(f"Saved projected blocks: {args.output}")
        return


if __name__ == "__main__":
    main()
