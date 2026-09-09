"""Hyperparameter tuning and model comparison for RangeShift AI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from .data import validate_training_frame

SUPPORTED_SCORING = {"roc_auc", "balanced_accuracy", "f1"}
DEFAULT_PARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 12],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt"],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
        "min_samples_leaf": [1, 3],
    },
}


@dataclass
class ModelSelectionResult:
    """Best tuned estimator and comparable cross-validation results."""

    best_model_name: str
    best_estimator: object
    best_score: float
    best_params: dict[str, object]
    cv_results: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    scoring: str
    spatial_groups_used: bool


def _fold_has_both_classes(y: pd.Series, indices) -> bool:
    return set(y.iloc[indices].unique().tolist()) == {0, 1}


def _validation_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups,
    *,
    n_splits: int,
    random_state: int,
    max_group_split_attempts: int = 100,
):
    """Materialize reproducible CV folds with both classes in train and test data."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if groups is None:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(splitter.split(X, y))
        if all(
            _fold_has_both_classes(y, train_indices)
            and _fold_has_both_classes(y, test_indices)
            for train_indices, test_indices in splits
        ):
            return splits
        raise ValueError(
            "Cross-validation could not place both classes in every fold. "
            "Reduce n_splits or provide more observations per class."
        )

    group_series = pd.Series(groups, index=X.index)
    if group_series.isna().any():
        raise ValueError("Spatial groups cannot contain missing values.")
    if group_series.nunique() < n_splits:
        raise ValueError("Spatial model tuning requires at least n_splits unique groups.")
    if max_group_split_attempts < 1:
        raise ValueError("max_group_split_attempts must be at least 1.")

    for attempt in range(max_group_split_attempts):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + attempt,
        )
        splits = list(splitter.split(X, y, groups=group_series))
        if all(
            _fold_has_both_classes(y, train_indices)
            and _fold_has_both_classes(y, test_indices)
            for train_indices, test_indices in splits
        ):
            return splits

    raise ValueError(
        "Could not create grouped cross-validation folds containing both target classes "
        "in every training and test partition. Reduce n_splits, use smaller spatial "
        "blocks, or provide broader class coverage across groups."
    )


def tune_and_compare_models(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str = "presence",
    *,
    groups=None,
    n_splits: int = 5,
    random_state: int = 42,
    scoring: str = "roc_auc",
    param_grids: Mapping[str, Mapping[str, Sequence[object]]] | None = None,
    n_jobs: int = -1,
) -> ModelSelectionResult:
    """Tune Random Forest and Gradient Boosting under one validation design.

    When ``groups`` are supplied, complete groups remain together inside a
    validated ``StratifiedGroupKFold`` design. RangeShift materializes the folds
    before tuning and requires both classes in every train/test partition so
    classification metrics such as ROC-AUC remain defined. Balanced sample
    weights are supplied to both algorithms for a fair comparison.
    """
    feature_columns = list(feature_columns)
    validate_training_frame(frame, feature_columns, target_column)
    if scoring not in SUPPORTED_SCORING:
        supported = ", ".join(sorted(SUPPORTED_SCORING))
        raise ValueError(f"Unsupported scoring '{scoring}'. Choose from: {supported}.")
    if groups is not None and len(groups) != len(frame):
        raise ValueError("groups must contain one value per observation.")

    grids = dict(DEFAULT_PARAM_GRIDS if param_grids is None else param_grids)
    required = {"random_forest", "gradient_boosting"}
    if set(grids) != required:
        raise ValueError(
            "param_grids must contain exactly 'random_forest' and 'gradient_boosting'."
        )

    X = frame[feature_columns]
    y = frame[target_column].astype(int)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    cv_splits = _validation_splits(
        X,
        y,
        groups,
        n_splits=n_splits,
        random_state=random_state,
    )

    estimators = {
        "random_forest": RandomForestClassifier(
            random_state=random_state,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }

    rows: list[pd.DataFrame] = []
    searches = {}
    for name, estimator in estimators.items():
        search = GridSearchCV(
            estimator,
            param_grid=grids[name],
            scoring=scoring,
            cv=cv_splits,
            refit=True,
            n_jobs=n_jobs,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X, y, sample_weight=sample_weight)
        searches[name] = search

        result_frame = pd.DataFrame(search.cv_results_)
        compact = result_frame[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].copy()
        compact.insert(0, "model", name)
        rows.append(compact)

    best_model_name = max(searches, key=lambda name: searches[name].best_score_)
    best_search = searches[best_model_name]
    combined = pd.concat(rows, ignore_index=True).sort_values(
        ["rank_test_score", "mean_test_score"],
        ascending=[True, False],
        ignore_index=True,
    )

    return ModelSelectionResult(
        best_model_name=best_model_name,
        best_estimator=best_search.best_estimator_,
        best_score=float(best_search.best_score_),
        best_params=dict(best_search.best_params_),
        cv_results=combined,
        feature_columns=feature_columns,
        target_column=target_column,
        scoring=scoring,
        spatial_groups_used=groups is not None,
    )


def save_selected_model_bundle(result: ModelSelectionResult, path: str | Path) -> Path:
    """Persist the selected estimator using the standard RangeShift bundle contract."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": result.best_estimator,
        "feature_columns": result.feature_columns,
        "target_column": result.target_column,
        "metrics": {f"cv_{result.scoring}": result.best_score},
        "model_name": result.best_model_name,
        "best_params": result.best_params,
        "spatial_groups_used": result.spatial_groups_used,
    }
    joblib.dump(bundle, path)
    return path
