"""Batch future-scenario analysis and projection uncertainty summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ConfigRunResult, RunConfig, run_configured_analysis


@dataclass
class ScenarioUncertaintyResult:
    """Raster outputs summarizing agreement across future scenarios."""

    mean_suitability_path: Path
    suitability_sd_path: Path
    suitable_fraction_path: Path
    scenario_count: int
    threshold: float

    def to_dict(self) -> dict:
        return {
            "mean_suitability": str(self.mean_suitability_path),
            "suitability_sd": str(self.suitability_sd_path),
            "suitable_fraction": str(self.suitable_fraction_path),
            "scenario_count": self.scenario_count,
            "threshold": self.threshold,
        }


@dataclass
class ScenarioBatchResult:
    """Outputs from running and summarizing multiple future scenarios."""

    manifest_path: Path
    scenario_summary_path: Path
    uncertainty: ScenarioUncertaintyResult
    scenario_results: dict[str, ConfigRunResult]


def _validate_scenario_name(name: str) -> None:
    if not name or name in {".", ".."}:
        raise ValueError("Scenario names must be non-empty directory-safe labels.")
    if any(character in name for character in ("/", "\\")):
        raise ValueError(f"Scenario name cannot contain path separators: {name!r}")


def _validate_scenario_layers(
    scenarios: dict[str, dict[str, Path]],
    features: list[str],
) -> None:
    if len(scenarios) < 2:
        raise ValueError("At least two future scenarios are required for uncertainty analysis.")
    expected = set(features)
    for name, layers in scenarios.items():
        _validate_scenario_name(name)
        observed = set(layers)
        if observed != expected:
            missing = sorted(expected.difference(observed))
            extras = sorted(observed.difference(expected))
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if extras:
                details.append(f"unexpected: {', '.join(extras)}")
            raise ValueError(
                f"Scenario {name!r} layer mapping does not match features; "
                + "; ".join(details)
            )


def summarize_suitability_scenarios(
    scenario_rasters: dict[str, str | Path],
    output_dir: str | Path,
    *,
    threshold: float,
) -> ScenarioUncertaintyResult:
    """Summarize future-suitability uncertainty across aligned scenario rasters.

    The outputs are the cell-wise mean suitability, standard deviation in suitability,
    and fraction of scenarios classified as suitable at the supplied threshold. Cells
    that are nodata in any scenario are nodata in all three summary rasters.
    """
    if len(scenario_rasters) < 2:
        raise ValueError("At least two scenario rasters are required.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("Install RangeShift with the 'geo' extra for scenario rasters.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(scenario_rasters)
    arrays: list[np.ndarray] = []
    combined_valid: np.ndarray | None = None
    reference_profile = None
    reference_shape = None
    reference_transform = None
    reference_crs = None

    for name in names:
        path = Path(scenario_rasters[name])
        with rasterio.open(path) as dataset:
            if reference_profile is None:
                reference_profile = dataset.profile.copy()
                reference_shape = dataset.shape
                reference_transform = dataset.transform
                reference_crs = dataset.crs
            elif (
                dataset.shape != reference_shape
                or dataset.transform != reference_transform
                or dataset.crs != reference_crs
            ):
                raise ValueError(
                    "Scenario rasters must have identical shape, transform, and CRS."
                )

            values = dataset.read(1).astype(float)
            valid = np.isfinite(values)
            if dataset.nodata is not None:
                valid &= values != dataset.nodata
            arrays.append(values)
            combined_valid = valid if combined_valid is None else combined_valid & valid

    stack = np.stack(arrays, axis=0)
    assert combined_valid is not None
    mean = np.mean(stack, axis=0)
    sd = np.std(stack, axis=0, ddof=0)
    suitable_fraction = np.mean(stack >= threshold, axis=0)

    nodata = -9999.0
    mean = np.where(combined_valid, mean, nodata).astype(np.float32)
    sd = np.where(combined_valid, sd, nodata).astype(np.float32)
    suitable_fraction = np.where(combined_valid, suitable_fraction, nodata).astype(np.float32)

    assert reference_profile is not None
    profile = reference_profile.copy()
    profile.update(dtype="float32", count=1, nodata=nodata, compress="deflate")

    outputs = {
        "mean": output_dir / "future_suitability_mean.tif",
        "sd": output_dir / "future_suitability_sd.tif",
        "fraction": output_dir / "future_suitable_fraction.tif",
    }
    for path, values in (
        (outputs["mean"], mean),
        (outputs["sd"], sd),
        (outputs["fraction"], suitable_fraction),
    ):
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.write(values, 1)

    return ScenarioUncertaintyResult(
        mean_suitability_path=outputs["mean"],
        suitability_sd_path=outputs["sd"],
        suitable_fraction_path=outputs["fraction"],
        scenario_count=len(scenario_rasters),
        threshold=float(threshold),
    )


def run_scenario_batch(
    base_config: RunConfig,
    scenarios: dict[str, dict[str, str | Path]],
    output_dir: str | Path,
) -> ScenarioBatchResult:
    """Run the same calibrated analysis assumptions across multiple future scenarios.

    Each scenario receives its own complete RangeShift run directory. The resulting
    future-suitability rasters are then summarized into ensemble mean, standard
    deviation, and threshold-agreement rasters. Identical seeds and training inputs
    make the per-scenario fitted model deterministic under the current workflow.
    """
    normalized = {
        name: {feature: Path(path) for feature, path in layers.items()}
        for name, layers in scenarios.items()
    }
    _validate_scenario_layers(normalized, base_config.features)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, ConfigRunResult] = {}
    summary_rows = []

    for name, future_layers in normalized.items():
        scenario_config = replace(
            base_config,
            future_layers=future_layers,
            output_dir=output_dir / name,
        )
        result = run_configured_analysis(scenario_config)
        results[name] = result
        range_summary = json.loads(result.range_shift_summary_path.read_text())
        row = {
            "scenario": name,
            "selected_threshold": result.selected_threshold,
            "future_suitability": str(result.future_suitability_path),
            "range_shift_summary": str(result.range_shift_summary_path),
        }
        for key, value in range_summary.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"range_{key}"] = value
        summary_rows.append(row)

    thresholds = np.array([result.selected_threshold for result in results.values()])
    if not np.allclose(thresholds, thresholds[0], rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "Scenario runs selected different thresholds; ensemble agreement would not "
            "be directly comparable."
        )

    uncertainty = summarize_suitability_scenarios(
        {name: result.future_suitability_path for name, result in results.items()},
        output_dir / "uncertainty",
        threshold=float(thresholds[0]),
    )

    summary_path = output_dir / "scenario_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    manifest = {
        "scenario_count": len(results),
        "scenarios": {
            name: {
                "manifest": str(result.manifest_path),
                "future_suitability": str(result.future_suitability_path),
                "range_shift_classes": str(result.range_shift_classes_path),
                "range_shift_summary": str(result.range_shift_summary_path),
                "config_sha256": result.config_sha256,
            }
            for name, result in results.items()
        },
        "uncertainty": uncertainty.to_dict(),
        "scenario_summary": str(summary_path),
        "interpretation": (
            "Agreement is the fraction of supplied scenarios above the same validated "
            "suitability threshold. It is scenario spread, not a probability that the "
            "species will occupy a cell."
        ),
    }
    manifest_path = output_dir / "scenario_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return ScenarioBatchResult(
        manifest_path=manifest_path,
        scenario_summary_path=summary_path,
        uncertainty=uncertainty,
        scenario_results=results,
    )
