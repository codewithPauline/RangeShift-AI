"""Command-line interface for RangeShift AI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import load_occurrence_table
from .geospatial import assign_projected_blocks
from .model import save_model_bundle, train_habitat_model
from .prediction import load_model_bundle, predict_suitability
from .spatial import compare_random_and_spatial
from .spatial_cv import spatial_cross_validate


def build_parser() -> argparse.ArgumentParser:
    """Build the RangeShift command-line parser."""
    parser = argparse.ArgumentParser(
        prog="rangeshift",
        description="Machine-learning tools for habitat suitability and range-shift analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a baseline habitat-suitability model.")
    train.add_argument("input_csv", type=Path, help="CSV containing target and predictors.")
    train.add_argument("--target", default="presence")
    train.add_argument("--features", nargs="+", required=True)
    train.add_argument(
        "--output",
        type=Path,
        default=Path("rangeshift_model.joblib"),
    )
    train.add_argument("--test-size", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--trees", type=int, default=300)

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

    spatial = subparsers.add_parser(
        "compare-spatial",
        help="Compare random holdout with spatial block holdout performance.",
    )
    spatial.add_argument("input_csv", type=Path)
    spatial.add_argument("--target", default="presence")
    spatial.add_argument("--features", nargs="+", required=True)
    spatial.add_argument("--latitude", default="latitude")
    spatial.add_argument("--longitude", default="longitude")
    spatial.add_argument("--block-size", type=float, default=1.0)
    spatial.add_argument("--test-size", type=float, default=0.25)
    spatial.add_argument("--seed", type=int, default=42)
    spatial.add_argument("--trees", type=int, default=300)
    spatial.add_argument("--output", type=Path, default=None)

    spatial_cv = subparsers.add_parser(
        "spatial-cv",
        help="Run repeated spatial block holdouts and summarize model performance.",
    )
    spatial_cv.add_argument("input_csv", type=Path)
    spatial_cv.add_argument("--target", default="presence")
    spatial_cv.add_argument("--features", nargs="+", required=True)
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


def main() -> None:
    """Run the RangeShift command-line interface."""
    parser = build_parser()
    args = parser.parse_args()

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
        print("Model metrics")
        print(json.dumps(result.metrics, indent=2, sort_keys=True))
        print("\nFeature importance")
        print(result.feature_importance.to_string(index=False))
        print(f"\nSaved model: {saved_path}")
        return

    if args.command == "predict":
        frame = pd.read_csv(args.input_csv)
        bundle = load_model_bundle(args.model)
        suitability = predict_suitability(frame, bundle)
        output = frame.copy()
        output["suitability"] = suitability
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print(f"Saved predictions: {args.output}")
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
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(f"Saved comparison: {args.output}")
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
        print("\nFold metrics")
        print(result.fold_metrics.to_string(index=False))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.fold_metrics.to_csv(args.output, index=False)
            print(f"Saved fold metrics: {args.output}")
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


if __name__ == "__main__":
    main()
