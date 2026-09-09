"""Run a reproducible RangeShift workflow from one JSON configuration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_run_config, run_configured_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rangeshift-run",
        description="Execute a reproducible RangeShift analysis from JSON configuration.",
    )
    parser.add_argument("config", type=Path, help="Path to a validated RangeShift JSON config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_run_config(args.config)
    result = run_configured_analysis(config)
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "model": str(result.model_path),
                "current_suitability": str(result.current_suitability_path),
                "future_suitability": str(result.future_suitability_path),
                "range_shift_classes": str(result.range_shift_classes_path),
                "range_shift_summary": str(result.range_shift_summary_path),
                "selected_threshold": result.selected_threshold,
                "config_sha256": result.config_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
