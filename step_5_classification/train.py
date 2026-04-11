"""Training pipeline for Step 5 — Classification.

Trains three models on the temporal train split:
1. Logistic Regression   (linear, ElasticNet penalty, class_weight="balanced")
2. Random Forest         (ensemble, class_weight="balanced")
3. XGBoost Classifier    (gradient boosting, sample_weight for imbalance)

Each model is tuned via CV using group-aware TimeSeriesSplit splits built by
``shared.utils.make_temporal_cv_splits``.

Usage (from project root)::

    python -m step_5_classification.train
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from shared.utils import (
    build_preprocessor,
    impute_missing,
    make_temporal_cv_splits,
    temporal_train_test_split,
)
from step_5_classification.config import (
    ARTIFACTS_DIR,
    CV_SCORING,
    DROP_COLS,
    LABEL_MAP,
    LOGISTIC_PARAM_GRID,
    N_CV_SPLITS,
    RANDOM_FOREST_PARAM_DIST,
    RANDOM_SEARCH_N_ITER,
    RANDOM_SEARCH_N_JOBS,
    RANDOM_STATE,
    TARGET_COL,
    XGBOOST_PARAM_DIST,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

def build_logistic_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Build Logistic Regression pipeline with StandardScaler.

    Structure::

        Pipeline([
            ("pre",        ColumnTransformer(StandardScaler on numerics, ...)),
            ("classifier", LogisticRegression(penalty="elasticnet", ...)),
        ])

    Parameter-grid keys::

        "classifier__C"    # Pipeline -> LogisticRegression.C

    Args:
        X_train: Training feature DataFrame — used only to build the
            ``ColumnTransformer`` (infers which columns are present).

    Returns:
        Unfitted Pipeline ready for GridSearchCV.
    """
    preprocessor = build_preprocessor(X_train, model_type="linear")

    classifier = LogisticRegression(
        solver="saga",
        l1_ratio=0.5,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    logger.info(
        "build_logistic_pipeline: created Pipeline(ColumnTransformer[linear] "
        "+ LogisticRegression(elasticnet, balanced))"
    )

    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_random_forest_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Build Random Forest Classifier pipeline with tree-mode preprocessor.

    Structure::

        Pipeline([
            ("pre",        ColumnTransformer(passthrough on numerics, ...)),
            ("classifier", RandomForestClassifier(class_weight="balanced")),
        ])

    Parameter-grid keys::

        "classifier__n_estimators"
        "classifier__max_depth"
        "classifier__min_samples_leaf"

    Args:
        X_train: Training feature DataFrame.

    Returns:
        Unfitted Pipeline ready for RandomizedSearchCV.
    """
    preprocessor = build_preprocessor(X_train, model_type="tree")

    classifier = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,  # outer RandomizedSearchCV already uses n_jobs=-1
    )

    logger.info(
        "build_random_forest_pipeline: created Pipeline(ColumnTransformer[tree] "
        "+ RandomForestClassifier(balanced))"
    )

    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_xgboost_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Build XGBoost Classifier pipeline with tree-mode preprocessor.

    Structure::

        Pipeline([
            ("pre",        ColumnTransformer(passthrough on numerics, ...)),
            ("classifier", XGBClassifier(objective="multi:softprob", num_class=4)),
        ])

    Class imbalance is handled via ``compute_sample_weight("balanced")``
    passed as a fit parameter (not built into the estimator).

    Parameter-grid keys::

        "classifier__n_estimators"
        "classifier__max_depth"
        "classifier__learning_rate"
        "classifier__subsample"
        "classifier__min_child_weight"
        "classifier__colsample_bytree"

    Args:
        X_train: Training feature DataFrame.

    Returns:
        Unfitted Pipeline ready for RandomizedSearchCV.
    """
    preprocessor = build_preprocessor(X_train, model_type="tree")

    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,  # outer RandomizedSearchCV already uses n_jobs=-1
    )

    logger.info(
        "build_xgboost_pipeline: created Pipeline(ColumnTransformer[tree] "
        "+ XGBClassifier(multi:softprob, 4 classes))"
    )

    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("classifier", classifier),
        ]
    )


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> GridSearchCV:
    """Train Logistic Regression via exhaustive GridSearchCV.

    Grid: 4 values of C × 5 folds = 20 fits.

    Args:
        X_train: Training features (post-imputation, without ``data_giorno``).
        y_train: Training target — encoded alert class (0–3).
        cv_splits: Temporal CV splits.

    Returns:
        Fitted GridSearchCV.
    """
    pipeline = build_logistic_pipeline(X_train)

    n_c = len(LOGISTIC_PARAM_GRID["classifier__C"])
    logger.info(
        "Logistic GridSearchCV: %d C values × %d folds = %d fits",
        n_c, len(cv_splits), n_c * len(cv_splits),
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=LOGISTIC_PARAM_GRID,
        cv=cv_splits,
        scoring=CV_SCORING,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    logger.info("Logistic best params:    %s", search.best_params_)
    logger.info("Logistic best CV f1_macro: %.4f", search.best_score_)

    return search


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> RandomizedSearchCV:
    """Train Random Forest Classifier via RandomizedSearchCV.

    Args:
        X_train: Training features (post-imputation, without ``data_giorno``).
        y_train: Training target — encoded alert class (0–3).
        cv_splits: Temporal CV splits.

    Returns:
        Fitted RandomizedSearchCV.
    """
    pipeline = build_random_forest_pipeline(X_train)

    param_dist = {f"classifier__{k}": v for k, v in RANDOM_FOREST_PARAM_DIST.items()}

    full_grid_size = int(np.prod([len(v) for v in RANDOM_FOREST_PARAM_DIST.values()]))
    logger.info(
        "RandomForest RandomizedSearchCV: %d/%d combinations × %d folds = %d fits",
        RANDOM_SEARCH_N_ITER, full_grid_size, len(cv_splits),
        RANDOM_SEARCH_N_ITER * len(cv_splits),
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=RANDOM_SEARCH_N_ITER,
        cv=cv_splits,
        scoring=CV_SCORING,
        n_jobs=RANDOM_SEARCH_N_JOBS,
        refit=True,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train, y_train)

    logger.info("RandomForest best params:    %s", search.best_params_)
    logger.info("RandomForest best CV f1_macro: %.4f", search.best_score_)

    return search


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> RandomizedSearchCV:
    """Train XGBoost Classifier via RandomizedSearchCV with sample weights.

    Class imbalance is handled by passing ``compute_sample_weight("balanced")``
    as a fit parameter to the classifier step in the pipeline.

    Args:
        X_train: Training features (post-imputation, without ``data_giorno``).
        y_train: Training target — encoded alert class (0–3).
        cv_splits: Temporal CV splits.

    Returns:
        Fitted RandomizedSearchCV.
    """
    pipeline = build_xgboost_pipeline(X_train)

    param_dist = {f"classifier__{k}": v for k, v in XGBOOST_PARAM_DIST.items()}

    full_grid_size = int(np.prod([len(v) for v in XGBOOST_PARAM_DIST.values()]))
    logger.info(
        "XGBoost RandomizedSearchCV: %d/%d combinations × %d folds = %d fits",
        RANDOM_SEARCH_N_ITER, full_grid_size, len(cv_splits),
        RANDOM_SEARCH_N_ITER * len(cv_splits),
    )

    sample_weights = compute_sample_weight("balanced", y_train)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=RANDOM_SEARCH_N_ITER,
        cv=cv_splits,
        scoring=CV_SCORING,
        n_jobs=RANDOM_SEARCH_N_JOBS,
        refit=True,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    logger.info("XGBoost best params:    %s", search.best_params_)
    logger.info("XGBoost best CV f1_macro: %.4f", search.best_score_)

    return search


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def run_training(parquet_path: str) -> Dict[str, object]:
    """Train all classification models and persist the best pipelines.

    Data flow::

        parquet
          -> temporal_train_test_split (70% train / 30% test by day)
          -> .map(LABEL_MAP) on y_train and y_test
          -> impute_missing (fit on train, apply same medians to test)
          -> make_temporal_cv_splits (group-aware, no intra-day leakage)
          -> train_logistic       (GridSearchCV, 4 combinations)
          -> train_random_forest  (RandomizedSearchCV, 50 iters)
          -> train_xgboost        (RandomizedSearchCV, 50 iters)
          -> select best by best_score_ (f1_macro)

    Args:
        parquet_path: Path to ``daily_dataset_clean.parquet``.

    Returns:
        Dict mapping model name to its fitted search object.
    """
    # 1. Load dataset
    df = pd.read_parquet(parquet_path)
    logger.info(
        "Loaded dataset: %d rows, %d columns from '%s'",
        len(df), df.shape[1], parquet_path,
    )

    # 2. Temporal train/test split
    X_train, y_train, X_test, y_test, train_dates = temporal_train_test_split(
        df, target_col=TARGET_COL, drop_cols=DROP_COLS,
    )

    # 3. Encode labels: string -> ordinal int via LABEL_MAP
    y_train = y_train.map(LABEL_MAP)
    y_test = y_test.map(LABEL_MAP)

    n_unmapped_train = int(y_train.isna().sum())
    n_unmapped_test = int(y_test.isna().sum())
    if n_unmapped_train or n_unmapped_test:
        logger.warning(
            "Unmapped labels after LABEL_MAP — train: %d, test: %d. "
            "These rows will be dropped.",
            n_unmapped_train, n_unmapped_test,
        )
        valid_train = y_train.notna()
        X_train, y_train = X_train[valid_train].reset_index(drop=True), y_train[valid_train].reset_index(drop=True)
        train_dates = train_dates[valid_train].reset_index(drop=True)
        valid_test = y_test.notna()
        X_test, y_test = X_test[valid_test].reset_index(drop=True), y_test[valid_test].reset_index(drop=True)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    logger.info(
        "Label distribution (train): %s",
        y_train.value_counts().sort_index().to_dict(),
    )
    logger.info(
        "Label distribution (test):  %s",
        y_test.value_counts().sort_index().to_dict(),
    )

    # 4. Imputation: fit medians on train only, apply same medians to test
    X_train, train_medians = impute_missing(X_train)
    X_test, _ = impute_missing(X_test, medians=train_medians)

    # 5. Build group-aware CV splits
    X_train_with_dates = X_train.copy()
    X_train_with_dates.insert(0, "data_giorno", train_dates)
    cv_splits = make_temporal_cv_splits(
        X_train_with_dates, y_train, n_splits=N_CV_SPLITS,
    )

    # Ensure artifacts directory exists
    artifacts_dir = Path(ARTIFACTS_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    trained_models: Dict[str, object] = {}

    # 6. Logistic Regression
    logger.info("=== Logistic Regression ===")
    lr_search = train_logistic(X_train, y_train, cv_splits)
    trained_models["logistic_regression"] = lr_search

    lr_path = artifacts_dir / "logistic_regression_best.joblib"
    joblib.dump(lr_search.best_estimator_, lr_path)
    logger.info("Saved best Logistic Regression pipeline -> %s", lr_path)

    # 7. Random Forest Classifier
    logger.info("=== Random Forest Classifier ===")
    rf_search = train_random_forest(X_train, y_train, cv_splits)
    trained_models["random_forest"] = rf_search

    rf_path = artifacts_dir / "random_forest_best.joblib"
    joblib.dump(rf_search.best_estimator_, rf_path)
    logger.info("Saved best Random Forest pipeline -> %s", rf_path)

    # 8. XGBoost Classifier
    logger.info("=== XGBoost Classifier ===")
    xgb_search = train_xgboost(X_train, y_train, cv_splits)
    trained_models["xgboost"] = xgb_search

    xgb_path = artifacts_dir / "xgboost_best.joblib"
    joblib.dump(xgb_search.best_estimator_, xgb_path)
    logger.info("Saved best XGBoost pipeline -> %s", xgb_path)

    # 9. Select best model across all three by CV f1_macro
    best_name = max(
        trained_models,
        key=lambda name: trained_models[name].best_score_,
    )
    best_search = trained_models[best_name]

    logger.info(
        "Best model: %s (CV f1_macro = %.4f)",
        best_name, best_search.best_score_,
    )

    best_path = artifacts_dir / "best_model.joblib"
    joblib.dump(best_search.best_estimator_, best_path)
    logger.info("Saved overall best model -> %s", best_path)

    # Also persist X_test/y_test for evaluate.py
    test_path = artifacts_dir / "test_data.joblib"
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_path)
    logger.info("Saved test data -> %s", test_path)

    # Persist X_train/y_train/train_dates for calibrate.py (step 5.5)
    train_path = artifacts_dir / "train_data.joblib"
    joblib.dump(
        {"X_train": X_train, "y_train": y_train, "train_dates": train_dates},
        train_path,
    )
    logger.info("Saved train data -> %s", train_path)

    return trained_models


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    _parquet = "step_3_eda/daily_dataset_clean.parquet"
    run_training(_parquet)
