"""Command-line interface for RangeShift ecological safeguards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .background import SUPPORTED_BACKGROUND_METHODS, generate_background_points
from .collinearity import diagnose_collinearity
from .dispersal import apply_dispersal_constraint
from .novelty import diagnose_novel_climate
from .thinning import thin_spatial_points


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rangeshift-eco",
        description="Ecological safeguards for RangeShift AI workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    background = subparsers.add_parser(
        "background",
        help="Generate pseudo-absence/background coordinates.",
    )
    background.add_argument("presence_csv", type=Path)
    background.add_argument("--n", type=int, required=True)
    background.add_argument("--method", choices=sorted(SUPPORTED_BACKGROUND_METHODS), default="random")
    background.add_argument("--latitude", default="latitude")
    background.add_argument("--longitude", default="longitude")
    background.add_argument("--min-distance-km", type=float, default=0.0)
    background.add_argument("--buffer-degrees", type=float, default=1.0)
    background.add_argument("--strata-size-degrees", type=float, default=1.0)
    background.add_argument("--candidate-pool", type=Path, default=None)
    background.add_argument("--seed", type=int, default=42)
    background.add_argument("--output", type=Path, default=Path("background_points.csv"))
    background.add_argument("--summary-output", type=Path, default=Path("background_summary.json"))

    thinning = subparsers.add_parser(
        "thin",
        help="Thin clustered occurrence records by minimum geodesic distance.",
    )
    thinning.add_argument("input_csv", type=Path)
    thinning.add_argument("--min-distance-km", type=float, required=True)
    thinning.add_argument("--latitude", default="latitude")
    thinning.add_argument("--longitude", default="longitude")
    thinning.add_argument("--priority", default=None)
    thinning.add_argument("--seed", type=int, default=42)
    thinning.add_argument("--output", type=Path, default=Path("thinned_occurrences.csv"))
    thinning.add_argument("--summary-output", type=Path, default=Path("thinning_summary.json"))

    collinearity = subparsers.add_parser(
        "collinearity",
        help="Diagnose correlated environmental predictors and VIF.",
    )
    collinearity.add_argument("input_csv", type=Path)
    collinearity.add_argument("--features", nargs="+", required=True)
    collinearity.add_argument("--correlation-threshold", type=float, default=0.7)
    collinearity.add_argument("--vif-threshold", type=float, default=5.0)
    collinearity.add_argument("--correlation-output", type=Path, default=Path("correlation_matrix.csv"))
    collinearity.add_argument("--pairs-output", type=Path, default=Path("high_correlation_pairs.csv"))
    collinearity.add_argument("--vif-output", type=Path, default=Path("vif.csv"))
    collinearity.add_argument("--summary-output", type=Path, default=Path("collinearity_summary.json"))

    novelty = subparsers.add_parser(
        "novel-climate",
        help="Flag univariate and multivariate environmental novelty.",
    )
    novelty.add_argument("training_csv", type=Path)
    novelty.add_argument("projection_csv", type=Path)
    novelty.add_argument("--features", nargs="+", required=True)
    novelty.add_argument("--distance-quantile", type=float, default=0.99)
    novelty.add_argument("--row-output", type=Path, default=Path("novel_climate_rows.csv"))
    novelty.add_argument("--feature-output", type=Path, default=Path("novel_climate_features.csv"))
    novelty.add_argument("--summary-output", type=Path, default=Path("novel_climate_summary.json"))

    dispersal = subparsers.add_parser(
        "dispersal",
        help="Apply an explicit maximum-distance dispersal constraint.",
    )
    dispersal.add_argument("current_raster", type=Path)
    dispersal.add_argument("future_raster", type=Path)
    dispersal.add_argument("--threshold", type=float, required=True)
    dispersal.add_argument("--max-distance-km", type=float, required=True)
    dispersal.add_argument(
        "--accessibility-output",
        type=Path,
        default=Path("dispersal_accessibility.tif"),
    )
    dispersal.add_argument(
        "--constrained-output",
        type=Path,
        default=Path("future_suitability_constrained.tif"),
    )
    dispersal.add_argument("--summary-output", type=Path, default=Path("dispersal_summary.json"))

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "background":
        presences = pd.read_csv(args.presence_csv)
        candidate_pool = pd.read_csv(args.candidate_pool) if args.candidate_pool else None
        result = generate_background_points(
            presences,
            args.n,
            method=args.method,
            latitude_column=args.latitude,
            longitude_column=args.longitude,
            min_distance_km=args.min_distance_km,
            buffer_degrees=args.buffer_degrees,
            strata_size_degrees=args.strata_size_degrees,
            candidate_pool=candidate_pool,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.points.to_csv(args.output, index=False)
        payload = {
            "method": result.method,
            "seed": result.seed,
            "requested_points": result.requested_points,
            "generated_points": int(len(result.points)),
            "min_distance_km": result.min_distance_km,
            "bounds": result.bounds,
            "output": str(args.output),
        }
        _write_json(args.summary_output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "thin":
        frame = pd.read_csv(args.input_csv)
        result = thin_spatial_points(
            frame,
            args.min_distance_km,
            latitude_column=args.latitude,
            longitude_column=args.longitude,
            priority_column=args.priority,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.thinned_frame.to_csv(args.output, index=False)
        payload = {
            "original_count": result.original_count,
            "retained_count": result.retained_count,
            "removed_count": len(result.removed_indices),
            "retention_fraction": result.retention_fraction,
            "min_distance_km": result.min_distance_km,
            "seed": result.seed,
            "priority_column": result.priority_column,
            "output": str(args.output),
        }
        _write_json(args.summary_output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "collinearity":
        frame = pd.read_csv(args.input_csv)
        result = diagnose_collinearity(
            frame,
            args.features,
            correlation_threshold=args.correlation_threshold,
            vif_threshold=args.vif_threshold,
        )
        for path in (args.correlation_output, args.pairs_output, args.vif_output):
            path.parent.mkdir(parents=True, exist_ok=True)
        result.correlation_matrix.to_csv(args.correlation_output)
        result.high_correlation_pairs.to_csv(args.pairs_output, index=False)
        result.vif_table.to_csv(args.vif_output, index=False)
        payload = {
            "correlation_threshold": result.correlation_threshold,
            "vif_threshold": result.vif_threshold,
            "high_correlation_pair_count": int(len(result.high_correlation_pairs)),
            "flagged_features": result.flagged_features,
            "correlation_matrix": str(args.correlation_output),
            "high_correlation_pairs": str(args.pairs_output),
            "vif": str(args.vif_output),
        }
        _write_json(args.summary_output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "novel-climate":
        training = pd.read_csv(args.training_csv)
        projection = pd.read_csv(args.projection_csv)
        result = diagnose_novel_climate(
            training,
            projection,
            args.features,
            distance_quantile=args.distance_quantile,
        )
        args.row_output.parent.mkdir(parents=True, exist_ok=True)
        args.feature_output.parent.mkdir(parents=True, exist_ok=True)
        result.row_diagnostics.to_csv(args.row_output, index=True)
        result.feature_summary.to_csv(args.feature_output, index=False)
        warnings = int(result.row_diagnostics["novel_climate_warning"].sum())
        payload = {
            "projection_rows": int(len(projection)),
            "warning_rows": warnings,
            "warning_fraction": float(warnings / len(projection)) if len(projection) else 0.0,
            "multivariate_distance_threshold": result.multivariate_distance_threshold,
            "distance_quantile": result.distance_quantile,
            "row_diagnostics": str(args.row_output),
            "feature_diagnostics": str(args.feature_output),
        }
        _write_json(args.summary_output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "dispersal":
        result = apply_dispersal_constraint(
            args.current_raster,
            args.future_raster,
            args.accessibility_output,
            args.constrained_output,
            threshold=args.threshold,
            max_distance_km=args.max_distance_km,
        )
        payload = result.to_dict()
        _write_json(args.summary_output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return


if __name__ == "__main__":
    main()
