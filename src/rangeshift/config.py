"""Configuration-driven reproducible RangeShift workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .calibration import save_calibrated_model_bundle, train_calibrated_habitat_model
from .data import load_occurrence_table
from .dispersal import apply_dispersal_constraint
from .range_shift import compare_suitability_rasters
from .raster import predict_suitability_raster, predict_suitability_raster_windowed
from .spatial import compare_random_and_spatial


@dataclass
class RunConfig:
    """Validated configuration for one reproducible RangeShift analysis."""

    training_csv: Path
    features: list[str]
    current_layers: dict[str, Path]
    future_layers: dict[str, Path]
    output_dir: Path
    target: str = "presence"
    latitude_column: str = "latitude"
    longitude_column: str = "longitude"
    seed: int = 42
    calibration_method: str = "sigmoid"
    threshold_method: str = "tss"
    calibration_cv: int = 5
    spatial_validation: bool = True
    spatial_block_size_degrees: float = 1.0
    spatial_test_size: float = 0.25
    raster_window_size: int = 512
    dispersal_max_distance_km: float | None = None

    def to_serializable_dict(self) -> dict:
        payload = asdict(self)
        payload["training_csv"] = str(self.training_csv)
        payload["output_dir"] = str(self.output_dir)
        payload["current_layers"] = {
            key: str(value) for key, value in self.current_layers.items()
        }
        payload["future_layers"] = {
            key: str(value) for key, value in self.future_layers.items()
        }
        return payload


@dataclass
class ConfigRunResult:
    """Outputs from a configuration-driven run."""

    manifest_path: Path
    model_path: Path
    current_suitability_path: Path
    future_suitability_path: Path
    range_shift_classes_path: Path
    range_shift_summary_path: Path
    selected_threshold: float
    config_sha256: str


def _validate_layer_mapping(layers: dict[str, Path], features: list[str], label: str) -> None:
    if set(layers) != set(features):
        missing = sorted(set(features).difference(layers))
        extras = sorted(set(layers).difference(features))
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extras:
            details.append(f"unexpected: {', '.join(extras)}")
        raise ValueError(f"{label} layer mapping does not match features; {'; '.join(details)}")


def _parse_config(payload: dict) -> RunConfig:
    required = {"training_csv", "features", "current_layers", "future_layers", "output_dir"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Configuration is missing required keys: {', '.join(missing)}")

    allowed = required | {
        "target",
        "latitude_column",
        "longitude_column",
        "seed",
        "calibration_method",
        "threshold_method",
        "calibration_cv",
        "spatial_validation",
        "spatial_block_size_degrees",
        "spatial_test_size",
        "raster_window_size",
        "dispersal_max_distance_km",
    }
    extras = sorted(set(payload).difference(allowed))
    if extras:
        raise ValueError(f"Unknown configuration keys: {', '.join(extras)}")

    features = list(payload["features"])
    if not features or any(not isinstance(feature, str) or not feature for feature in features):
        raise ValueError("features must be a non-empty list of predictor names.")
    if len(set(features)) != len(features):
        raise ValueError("features cannot contain duplicates.")

    config = RunConfig(
        training_csv=Path(payload["training_csv"]),
        features=features,
        current_layers={key: Path(value) for key, value in dict(payload["current_layers"]).items()},
        future_layers={key: Path(value) for key, value in dict(payload["future_layers"]).items()},
        output_dir=Path(payload["output_dir"]),
        target=str(payload.get("target", "presence")),
        latitude_column=str(payload.get("latitude_column", "latitude")),
        longitude_column=str(payload.get("longitude_column", "longitude")),
        seed=int(payload.get("seed", 42)),
        calibration_method=str(payload.get("calibration_method", "sigmoid")),
        threshold_method=str(payload.get("threshold_method", "tss")),
        calibration_cv=int(payload.get("calibration_cv", 5)),
        spatial_validation=bool(payload.get("spatial_validation", True)),
        spatial_block_size_degrees=float(payload.get("spatial_block_size_degrees", 1.0)),
        spatial_test_size=float(payload.get("spatial_test_size", 0.25)),
        raster_window_size=int(payload.get("raster_window_size", 512)),
        dispersal_max_distance_km=(
            None
            if payload.get("dispersal_max_distance_km") is None
            else float(payload["dispersal_max_distance_km"])
        ),
    )

    _validate_layer_mapping(config.current_layers, config.features, "Current")
    _validate_layer_mapping(config.future_layers, config.features, "Future")
    if config.calibration_method not in {"sigmoid", "isotonic"}:
        raise ValueError("calibration_method must be 'sigmoid' or 'isotonic'.")
    if config.threshold_method not in {"tss", "youden_j", "f1", "balanced_accuracy"}:
        raise ValueError(
            "threshold_method must be one of: tss, youden_j, f1, balanced_accuracy."
        )
    if config.calibration_cv < 2:
        raise ValueError("calibration_cv must be at least 2.")
    if config.raster_window_size < 0:
        raise ValueError("raster_window_size cannot be negative.")
    if config.dispersal_max_distance_km is not None and config.dispersal_max_distance_km <= 0:
        raise ValueError("dispersal_max_distance_km must be greater than 0 when provided.")
    return config


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a JSON RangeShift configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a JSON object.")
    return _parse_config(payload)


def _config_hash(config: RunConfig) -> str:
    canonical = json.dumps(
        config.to_serializable_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _predict_raster(config: RunConfig, bundle: dict, layers: dict[str, Path], output: Path):
    if config.raster_window_size > 0:
        return predict_suitability_raster_windowed(
            bundle,
            layers,
            output,
            window_size=config.raster_window_size,
        )
    return predict_suitability_raster(bundle, layers, output)


def run_configured_analysis(config: RunConfig) -> ConfigRunResult:
    """Execute a calibrated current-to-future analysis from one validated config."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_occurrence_table(config.training_csv)

    calibrated = train_calibrated_habitat_model(
        frame,
        feature_columns=config.features,
        target_column=config.target,
        random_state=config.seed,
        calibration_method=config.calibration_method,
        calibration_cv=config.calibration_cv,
        threshold_method=config.threshold_method,
    )
    model_path = save_calibrated_model_bundle(calibrated, output_dir / "model.joblib")
    bundle = {
        "model": calibrated.model,
        "feature_columns": calibrated.feature_columns,
        "target_column": calibrated.target_column,
        "metrics": calibrated.test_probability_metrics,
    }

    threshold_path = output_dir / "threshold_diagnostics.csv"
    calibration_path = output_dir / "calibration_diagnostics.csv"
    calibrated.threshold_table.to_csv(threshold_path, index=False)
    calibrated.calibration_table.to_csv(calibration_path, index=False)

    spatial_payload = None
    if config.spatial_validation:
        spatial = compare_random_and_spatial(
            frame,
            feature_columns=config.features,
            target_column=config.target,
            latitude_column=config.latitude_column,
            longitude_column=config.longitude_column,
            block_size_degrees=config.spatial_block_size_degrees,
            test_size=config.spatial_test_size,
            random_state=config.seed,
        )
        spatial_payload = {
            "random": spatial.random.metrics,
            "spatial": spatial.spatial.metrics,
            "spatial_minus_random": spatial.spatial_minus_random,
        }

    current_path = output_dir / "current_suitability.tif"
    future_raw_path = output_dir / "future_suitability_raw.tif"
    _predict_raster(config, bundle, config.current_layers, current_path)
    _predict_raster(config, bundle, config.future_layers, future_raw_path)

    future_path = future_raw_path
    dispersal_payload = None
    if config.dispersal_max_distance_km is not None:
        accessibility_path = output_dir / "dispersal_accessibility.tif"
        constrained_path = output_dir / "future_suitability_constrained.tif"
        dispersal = apply_dispersal_constraint(
            current_path,
            future_raw_path,
            accessibility_path,
            constrained_path,
            threshold=calibrated.selected_threshold,
            max_distance_km=config.dispersal_max_distance_km,
        )
        future_path = constrained_path
        dispersal_payload = dispersal.to_dict()

    classes_path = output_dir / "range_shift_classes.tif"
    difference_path = output_dir / "suitability_change.tif"
    range_shift = compare_suitability_rasters(
        current_path,
        future_path,
        classes_path,
        threshold=calibrated.selected_threshold,
        difference_output_path=difference_path,
    )
    range_summary_path = output_dir / "range_shift_summary.json"
    range_summary_path.write_text(json.dumps(range_shift.to_dict(), indent=2, sort_keys=True) + "\n")

    config_hash = _config_hash(config)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash,
        "config": config.to_serializable_dict(),
        "selected_threshold": calibrated.selected_threshold,
        "threshold_method": calibrated.threshold_method,
        "calibration_method": calibrated.calibration_method,
        "test_probability_metrics": calibrated.test_probability_metrics,
        "test_classification_metrics": calibrated.test_classification_metrics,
        "spatial_validation": spatial_payload,
        "dispersal_constraint": dispersal_payload,
        "outputs": {
            "model": str(model_path),
            "threshold_diagnostics": str(threshold_path),
            "calibration_diagnostics": str(calibration_path),
            "current_suitability": str(current_path),
            "future_suitability": str(future_path),
            "range_shift_classes": str(classes_path),
            "suitability_change": str(difference_path),
            "range_shift_summary": str(range_summary_path),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return ConfigRunResult(
        manifest_path=manifest_path,
        model_path=model_path,
        current_suitability_path=current_path,
        future_suitability_path=future_path,
        range_shift_classes_path=classes_path,
        range_shift_summary_path=range_summary_path,
        selected_threshold=float(calibrated.selected_threshold),
        config_sha256=config_hash,
    )
