"""Command-line interface for RangeShift AI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_occurrence_table
from .model import save_model_bundle, train_habitat_model


def build_parser() -> argparse.ArgumentParser:
    """Build the RangeShift command-line parser."""
    parser = argparse.ArgumentParser(
        prog="rangeshift",
        description="Machine-learning tools for habitat suitability and range-shift analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a baseline habitat-suitability model.")
    train.add_argument("input_csv", type=Path, help="CSV containing target and predictors.")
    train.add_argument(
        "--target",
        default="presence",
        help="Binary response column (default: presence).",
    )
    train.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Environmental predictor column names.",
    )
    train.add_argument(
        "--output",
        type=Path,
        default=Path("rangeshift_model.joblib"),
        help="Path for the trained model bundle.",
    )
    train.add_argument("--test-size", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--trees", type=int, default=300)
    return parser


def main() -> None:
    """Run the RangeShift command line interface."""
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


if __name__ == "__main__":
    main()
