"""Run the prepared Plethodon cinereus climate scenarios through RangeShift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rangeshift import load_run_config, run_scenario_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--climate-dir",
        type=Path,
        default=Path("plethodon_cinereus_climate"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plethodon_cinereus_scenario_runs"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.climate_dir / "base_run_config.json"
    scenarios_path = args.climate_dir / "scenario_layers.json"
    if not config_path.exists() or not scenarios_path.exists():
        raise FileNotFoundError(
            "Prepared climate inputs were not found. Run prepare_climate_scenarios.py first."
        )

    config = load_run_config(config_path)
    scenarios = json.loads(scenarios_path.read_text())
    if not isinstance(scenarios, dict):
        raise ValueError("scenario_layers.json must contain a JSON object.")

    result = run_scenario_batch(config, scenarios, args.output_dir)
    print(f"Scenario summary: {result.scenario_summary_path}")
    print(f"Scenario manifest: {result.manifest_path}")
    print(f"Mean suitability: {result.uncertainty.mean_suitability_path}")
    print(f"Suitability SD: {result.uncertainty.suitability_sd_path}")
    print(f"Scenario agreement: {result.uncertainty.suitable_fraction_path}")


if __name__ == "__main__":
    main()
