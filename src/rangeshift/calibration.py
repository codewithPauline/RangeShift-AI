"""Probability calibration and leakage-aware threshold selection for RangeShift AI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import validate_training_frame
from .threshold import ThresholdSelectionResult, evaluate_thresholds, select_threshold

SUPPORTED_CALIBRATION_METHODS = {"sigmoid", "isotonic"}


@dataclass
class CalibrationResult:
    """Artifacts from calibrated training, threshold selection, and final testing."""

    model: CalibratedClassifierCV
    selected_threshold: float
    threshold_method: str
    calibration_method: str
    validation_threshold_metrics: dict[str, float]
    test_probability_metrics: dict[str, float]
    uncalibrated_test_probability_metrics: dict[str, float]
    test_classification_metrics: dict[str, float]
    calibration_table: pd.DataFrame
    threshold_table: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    split_sizes: dict[str, int]


def _probability_metrics(y_true: pd.Series, probabilities) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
    }


def _calibration_table(y_true: pd.Series, probabilities, *, n_bins: int) -> pd.DataFrame:
    observed, predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=n_bins,
        strategy="quantile",
    )
    return pd.DataFrame(
        {
            "mean_predicted_probability": predicted,
            "observed_positive_fraction": observed,
        }
    )


def train_calibrated_habitat_model(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str = "presence",
    *,
    validation_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
    n_estimators: int = 300,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 5,
    threshold_method: str = "tss",
    calibration_bins: int = 10,
) -> CalibrationResult:
    """Train a calibrated Random Forest with separate validation and test sets.

    The model is fitted only on the training partition. Probability calibration
    uses cross-validation within that training partition. The habitat threshold
    is selected from validation predictions, and the final metrics are reported
    once on a separate untouched test partition.
    """
    feature_columns = list(feature_columns)
    validate_training_frame(frame, feature_columns, target_column)

    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")
    if calibration_method not in SUPPORTED_CALIBRATION_METHODS:
        supported = ", ".join(sorted(SUPPORTED_CALIBRATION_METHODS))
        raise ValueError(
            f"Unsupported calibration method '{calibration_method}'. Choose from: {supported}."
        )
    if calibration_cv < 2:
        raise ValueError("calibration_cv must be at least 2.")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2.")

    X = frame[feature_columns]
    y = frame[target_column].astype(int)

    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        relative_validation_size = validation_size / (1.0 - test_size)
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_validation_size,
            random_state=random_state + 1,
            stratify=y_train_val,
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to create stratified train/validation/test partitions. "
            "Provide more observations per class or adjust the split fractions."
        ) from exc

    class_counts = y_train.value_counts()
    if int(class_counts.min()) < calibration_cv:
        raise ValueError(
            "The training partition has fewer observations in one class than calibration_cv. "
            "Reduce calibration_cv or provide more data."
        )

    baseline = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    baseline.fit(X_train, y_train)
    baseline_test_probabilities = baseline.predict_proba(X_test)[:, 1]

    calibrated_base = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    calibrated = CalibratedClassifierCV(
        estimator=calibrated_base,
        method=calibration_method,
        cv=calibration_cv,
    )
    calibrated.fit(X_train, y_train)

    validation_probabilities = calibrated.predict_proba(X_validation)[:, 1]
    threshold_result: ThresholdSelectionResult = select_threshold(
        y_validation,
        validation_probabilities,
        method=threshold_method,
    )

    test_probabilities = calibrated.predict_proba(X_test)[:, 1]
    test_threshold_row = evaluate_thresholds(
        y_test,
        test_probabilities,
        thresholds=[threshold_result.threshold],
    ).iloc[0]
    test_classification_metrics = {
        key: float(test_threshold_row[key])
        for key in (
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "balanced_accuracy",
            "tss",
            "youden_j",
            "predicted_positive_rate",
        )
    }
    test_classification_metrics["threshold"] = threshold_result.threshold

    return CalibrationResult(
        model=calibrated,
        selected_threshold=threshold_result.threshold,
        threshold_method=threshold_method,
        calibration_method=calibration_method,
        validation_threshold_metrics=threshold_result.metrics,
        test_probability_metrics=_probability_metrics(y_test, test_probabilities),
        uncalibrated_test_probability_metrics=_probability_metrics(
            y_test,
            baseline_test_probabilities,
        ),
        test_classification_metrics=test_classification_metrics,
        calibration_table=_calibration_table(
            y_test,
            test_probabilities,
            n_bins=calibration_bins,
        ),
        threshold_table=threshold_result.table,
        feature_columns=feature_columns,
        target_column=target_column,
        split_sizes={
            "train": int(len(X_train)),
            "validation": int(len(X_validation)),
            "test": int(len(X_test)),
        },
    )


def save_calibrated_model_bundle(result: CalibrationResult, path: str | Path) -> Path:
    """Save a calibrated model using the same prediction bundle contract as v0.4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        **{
            f"probability_{key}": value
            for key, value in result.test_probability_metrics.items()
        },
        **{
            f"classification_{key}": value
            for key, value in result.test_classification_metrics.items()
        },
    }
    bundle = {
        "model": result.model,
        "feature_columns": result.feature_columns,
        "target_column": result.target_column,
        "metrics": metrics,
        "selected_threshold": result.selected_threshold,
        "threshold_method": result.threshold_method,
        "calibration_method": result.calibration_method,
    }
    joblib.dump(bundle, path)
    return path
