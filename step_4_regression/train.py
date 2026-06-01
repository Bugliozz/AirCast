"""Training pipeline for Step 4 — Regression.

Trains three models on the temporal train split:
1. ElasticNet          (linear, log-transformed target, StandardScaler)
2. XGBoost Regressor   (gradient boosting, no scaling)  [task 3.3.2]
3. Random Forest       (ensemble baseline, no scaling)  [task 3.3.3]

Each model is tuned via CV using group-aware TimeSeriesSplit splits built by
``shared.utils.make_temporal_cv_splits``.

Usage (from project root)::

    python -m step_4_regression.train
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from shared.utils import (
    build_preprocessor,
    impute_missing,
    make_temporal_cv_splits,
    temporal_train_test_split,
)
from step_4_regression.config import (
    ARTIFACTS_DIR,
    CV_SCORING,
    DROP_COLS,
    ELASTICNET_PARAM_GRID,
    N_CV_SPLITS,
    RANDOM_FOREST_FIT_PARAMS,
    RANDOM_FOREST_PARAM_DIST,
    RANDOM_SEARCH_N_ITER,
    RANDOM_SEARCH_N_JOBS,
    RANDOM_STATE,
    TARGET_COL,
    XGBOOST_FIT_PARAMS,
    XGBOOST_PARAM_DIST,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3.2 — Pipeline factories (target transformation)
# ---------------------------------------------------------------------------

def build_elasticnet_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Build ElasticNet pipeline with StandardScaler and log-transform on the target.

    Structure::

        Pipeline([
            ("pre",       ColumnTransformer(StandardScaler on numerics, ...)),
            ("regressor", TransformedTargetRegressor(
                              regressor=ElasticNet(),
                              func=np.log1p,
                              inverse_func=np.expm1,
                          )),
        ])

    The ``TransformedTargetRegressor`` applies ``np.log1p`` to *y* before
    fitting (stabilises variance on the right-skewed PM10 distribution) and
    ``np.expm1`` to convert predictions back to the original ug/m3 scale.

    Parameter-grid keys follow the nested path convention::

        "regressor__regressor__alpha"    # Pipeline -> TTR -> ElasticNet.alpha
        "regressor__regressor__l1_ratio" # Pipeline -> TTR -> ElasticNet.l1_ratio

    These paths match ``ELASTICNET_PARAM_GRID`` in ``config.py``.

    Args:
        X_train: Training feature DataFrame — used only to build the
            ``ColumnTransformer`` (infers which columns are present).

    Returns:
        Unfitted :class:`~sklearn.pipeline.Pipeline` ready to be passed to
        :class:`~sklearn.model_selection.GridSearchCV`.
    """
    preprocessor = build_preprocessor(X_train, model_type="linear")

    base_regressor = ElasticNet(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    log_target_regressor = TransformedTargetRegressor(
        regressor=base_regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    logger.info(
        "build_elasticnet_pipeline: created Pipeline(ColumnTransformer[linear] "
        "+ TransformedTargetRegressor(ElasticNet, log1p/expm1))"
    )

    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("regressor", log_target_regressor),
        ]
    )


# ---------------------------------------------------------------------------
# 3.3.2 — XGBoost pipeline factory
# ---------------------------------------------------------------------------

def build_xgboost_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Build XGBoost pipeline with tree-mode ColumnTransformer and log-transform on target.

    Structure::

        Pipeline([
            ("pre",       ColumnTransformer(passthrough on numerics, ...)),
            ("regressor", TransformedTargetRegressor(
                              regressor=XGBRegressor(...),
                              func=np.log1p,
                              inverse_func=np.expm1,
                          )),
        ])

    Tree-based models are scale-invariant, so ``StandardScaler`` is omitted.
    The ``TransformedTargetRegressor`` applies ``np.log1p`` to *y* before
    fitting so that XGBoost optimises RMSE in log-space: large errors on high
    PM10 values (peaks) receive relatively more weight, reducing the systematic
    under-prediction of peaks visible in the diagnostic plot.

    Parameter-grid keys follow the nested path convention::

        "regressor__regressor__n_estimators"
        "regressor__regressor__max_depth"
        "regressor__regressor__learning_rate"
        "regressor__regressor__subsample"

    These paths match ``XGBOOST_PARAM_DIST`` (prefixed at call-site) in
    ``config.py``.

    Args:
        X_train: Training feature DataFrame — used only to build the
            ``ColumnTransformer`` (infers which columns are present).

    Returns:
        Unfitted :class:`~sklearn.pipeline.Pipeline` ready to be passed to
        :class:`~sklearn.model_selection.RandomizedSearchCV`.
    """
    preprocessor = build_preprocessor(X_train, model_type="tree")

    base_regressor = XGBRegressor(
        random_state=XGBOOST_FIT_PARAMS["random_state"],
        n_jobs=XGBOOST_FIT_PARAMS["n_jobs"],
        tree_method=XGBOOST_FIT_PARAMS["tree_method"],
    )

    regressor = TransformedTargetRegressor(
        regressor=base_regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    logger.info(
        "build_xgboost_pipeline: created Pipeline(ColumnTransformer[tree] "
        "+ TransformedTargetRegressor(XGBRegressor, log1p/expm1))"
    )

    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("regressor", regressor),
        ]
    )


# ---------------------------------------------------------------------------
# 3.3.3 — Random Forest pipeline factory
# ---------------------------------------------------------------------------

def build_random_forest_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Build Random Forest pipeline with tree-mode ColumnTransformer and log-transform on target.

    Structure::

        Pipeline([
            ("pre",       ColumnTransformer(passthrough on numerics, ...)),
            ("regressor", TransformedTargetRegressor(
                              regressor=RandomForestRegressor(...),
                              func=np.log1p,
                              inverse_func=np.expm1,
                          )),
        ])

    Tree-based models are scale-invariant, so ``StandardScaler`` is omitted.
    The ``TransformedTargetRegressor`` applies ``np.log1p`` to *y* before
    fitting, mirroring the same treatment applied to XGBoost and ElasticNet
    to improve peak prediction.

    Parameter-grid keys follow the nested path convention::

        "regressor__regressor__n_estimators"
        "regressor__regressor__max_depth"
        "regressor__regressor__min_samples_leaf"

    These paths match ``RANDOM_FOREST_PARAM_DIST`` (prefixed at call-site) in
    ``config.py``.

    Args:
        X_train: Training feature DataFrame — used only to build the
            ``ColumnTransformer`` (infers which columns are present).

    Returns:
        Unfitted :class:`~sklearn.pipeline.Pipeline` ready to be passed to
        :class:`~sklearn.model_selection.RandomizedSearchCV`.
    """
    preprocessor = build_preprocessor(X_train, model_type="tree")

    base_regressor = RandomForestRegressor(
        random_state=RANDOM_FOREST_FIT_PARAMS["random_state"],
        n_jobs=RANDOM_FOREST_FIT_PARAMS["n_jobs"],
    )

    regressor = TransformedTargetRegressor(
        regressor=base_regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    logger.info(
        "build_random_forest_pipeline: created Pipeline(ColumnTransformer[tree] "
        "+ TransformedTargetRegressor(RandomForestRegressor, log1p/expm1))"
    )

    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("regressor", regressor),
        ]
    )


# ---------------------------------------------------------------------------
# 3.3.1 — ElasticNet training
# ---------------------------------------------------------------------------

def train_elasticnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> GridSearchCV:
    """Train ElasticNet via exhaustive GridSearchCV with temporal CV splits.

    Grid: 4 alpha values × 4 l1_ratio values = 16 combinations × 5 folds = 80 fits.
    Exhaustive search is feasible because ElasticNet fits are very fast.

    The pipeline includes:
    - ``ColumnTransformer`` with ``StandardScaler`` on numeric features
      (required for regularised linear models to work correctly)
    - ``TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)``
      (stabilises variance on the right-skewed PM10 distribution)
    - ``ElasticNet`` (combines L1 sparsity with L2 shrinkage)

    The fitted ``GridSearchCV.best_estimator_`` is a complete pipeline that
    accepts raw (un-scaled) feature DataFrames and returns PM10 predictions
    in the original ug/m3 scale.

    Args:
        X_train: Training features (post-imputation, without ``data_giorno``).
        y_train: Training target — PM10 in ug/m3 (original scale).
        cv_splits: Temporal CV splits from
            :func:`shared.utils.make_temporal_cv_splits`.  Each element is a
            ``(train_indices, val_indices)`` tuple.

    Returns:
        Fitted :class:`~sklearn.model_selection.GridSearchCV` with
        ``best_estimator_``, ``best_params_``, and ``best_score_`` populated.
        ``best_score_`` is *negative* RMSE (sklearn convention); negate it to
        get the actual RMSE value.
    """
    pipeline = build_elasticnet_pipeline(X_train)

    n_alpha = len(ELASTICNET_PARAM_GRID["regressor__regressor__alpha"])
    n_l1 = len(ELASTICNET_PARAM_GRID["regressor__regressor__l1_ratio"])
    n_combinations = n_alpha * n_l1
    logger.info(
        "ElasticNet GridSearchCV: %d alpha × %d l1_ratio = %d combinations "
        "× %d folds = %d fits",
        n_alpha, n_l1, n_combinations, len(cv_splits), n_combinations * len(cv_splits),
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=ELASTICNET_PARAM_GRID,
        cv=cv_splits,
        scoring=CV_SCORING,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    logger.info("ElasticNet best params:   %s", search.best_params_)
    logger.info("ElasticNet best CV RMSE:  %.4f ug/m3", -search.best_score_)

    return search


# ---------------------------------------------------------------------------
# 3.3.2 — XGBoost training
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> RandomizedSearchCV:
    """Train XGBoost Regressor via RandomizedSearchCV with temporal CV splits.

    Random search: 30 iterations over a 3×3×3×3=81 combination space (≈37%
    sampled), giving ≈90% of the quality at 37% of the compute cost.

    The pipeline:
    - ``ColumnTransformer`` with ``passthrough`` on numeric features
      (tree-based models are scale-invariant)
    - ``TransformedTargetRegressor(XGBRegressor, log1p/expm1)`` — optimises
      RMSE in log-space to reduce systematic under-prediction of PM10 peaks.

    Args:
        X_train: Training features (post-imputation, without ``data_giorno``).
        y_train: Training target — PM10 in ug/m3 (original scale).
        cv_splits: Temporal CV splits from
            :func:`shared.utils.make_temporal_cv_splits`.  Each element is a
            ``(train_indices, val_indices)`` tuple.

    Returns:
        Fitted :class:`~sklearn.model_selection.RandomizedSearchCV` with
        ``best_estimator_``, ``best_params_``, and ``best_score_`` populated.
        ``best_score_`` is *negative* RMSE (sklearn convention); negate it to
        get the actual RMSE value.
    """
    pipeline = build_xgboost_pipeline(X_train)

    # Prefix param keys for Pipeline nesting: "regressor__regressor__<param>"
    # (Pipeline -> TransformedTargetRegressor -> XGBRegressor)
    param_dist = {f"regressor__regressor__{k}": v for k, v in XGBOOST_PARAM_DIST.items()}

    full_grid_size = int(np.prod([len(v) for v in XGBOOST_PARAM_DIST.values()]))
    logger.info(
        "XGBoost RandomizedSearchCV: %d/%d combinations × %d folds = %d fits",
        RANDOM_SEARCH_N_ITER,
        full_grid_size,
        len(cv_splits),
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

    logger.info("XGBoost best params:  %s", search.best_params_)
    logger.info("XGBoost best CV RMSE: %.4f ug/m3", -search.best_score_)

    return search


# ---------------------------------------------------------------------------
# 3.3.3 — Random Forest training
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> RandomizedSearchCV:
    """Train Random Forest Regressor via RandomizedSearchCV with temporal CV splits.

    Random search: 30 iterations over a 3×3×3=27 combination space, giving
    ~90% of the quality at a fraction of the exhaustive-search cost.

    The pipeline:
    - ``ColumnTransformer`` with ``passthrough`` on numeric features
      (tree-based models are scale-invariant)
    - ``TransformedTargetRegressor(RandomForestRegressor, log1p/expm1)`` — same
      log-space training as XGBoost and ElasticNet for consistent peak handling.

    Serves as the **baseline** ensemble: less prone to overfitting than XGBoost
    because it averages independent trees rather than boosting residuals.

    Args:
        X_train: Training features (post-imputation, without ``data_giorno``).
        y_train: Training target — PM10 in ug/m3 (original scale).
        cv_splits: Temporal CV splits from
            :func:`shared.utils.make_temporal_cv_splits`.  Each element is a
            ``(train_indices, val_indices)`` tuple.

    Returns:
        Fitted :class:`~sklearn.model_selection.RandomizedSearchCV` with
        ``best_estimator_``, ``best_params_``, and ``best_score_`` populated.
        ``best_score_`` is *negative* RMSE (sklearn convention); negate it to
        get the actual RMSE value.
    """
    pipeline = build_random_forest_pipeline(X_train)

    # Prefix param keys for Pipeline nesting: "regressor__regressor__<param>"
    # (Pipeline -> TransformedTargetRegressor -> RandomForestRegressor)
    param_dist = {f"regressor__regressor__{k}": v for k, v in RANDOM_FOREST_PARAM_DIST.items()}

    full_grid_size = int(np.prod([len(v) for v in RANDOM_FOREST_PARAM_DIST.values()]))
    logger.info(
        "RandomForest RandomizedSearchCV: %d/%d combinations × %d folds = %d fits",
        RANDOM_SEARCH_N_ITER,
        full_grid_size,
        len(cv_splits),
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

    logger.info("RandomForest best params:  %s", search.best_params_)
    logger.info("RandomForest best CV RMSE: %.4f ug/m3", -search.best_score_)

    return search


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def run_training(parquet_path: str) -> Dict[str, GridSearchCV]:
    """Train all regression models and persist the best pipelines.

    Data flow::

        parquet
          -> temporal_train_test_split (70% train / 30% test by day)
          -> impute_missing (fit on train, apply same medians to test)
          -> make_temporal_cv_splits (group-aware, no intra-day leakage)
          -> train_elasticnet    (GridSearchCV, 16 combinations)    [3.3.1]
          -> train_xgboost      (RandomizedSearchCV, 30 iters)    [3.3.2]
          -> train_random_forest (RandomizedSearchCV, 30 iters)   [3.3.3]

    Args:
        parquet_path: Path to ``daily_dataset_clean.parquet``.

    Returns:
        Dict mapping model name to its fitted :class:`GridSearchCV` object.
    """
    # 1. Load dataset
    df = pd.read_parquet(parquet_path)
    logger.info(
        "Loaded dataset: %d rows, %d columns from '%s'",
        len(df), df.shape[1], parquet_path,
    )

    # 2. Temporal train/test split — 70% of unique days for training
    X_train, y_train, X_test, y_test, train_dates = temporal_train_test_split(
        df, target_col=TARGET_COL, drop_cols=DROP_COLS
    )

    # 3. Imputation: fit medians on train only, apply same medians to test.
    #    This order prevents any test information from leaking into the model.
    X_train, train_medians = impute_missing(X_train)
    X_test, _ = impute_missing(X_test, medians=train_medians)

    # 4. Build group-aware CV splits.
    #    Reattach train_dates (returned directly by the split function) so
    #    make_temporal_cv_splits can group rows by day and prevent stations
    #    from the same day landing in both train and val.
    X_train_with_dates = X_train.copy()
    X_train_with_dates.insert(0, "data_giorno", train_dates)
    cv_splits = make_temporal_cv_splits(X_train_with_dates, n_splits=N_CV_SPLITS)

    # Ensure artifacts directory exists
    artifacts_dir = Path(ARTIFACTS_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    trained_models: Dict[str, GridSearchCV] = {}

    # 5. Task 3.3.1 — ElasticNet (StandardScaler + log target)
    logger.info("=== 3.3.1 ElasticNet ===")
    en_search = train_elasticnet(X_train, y_train, cv_splits)
    trained_models["elasticnet"] = en_search

    en_path = artifacts_dir / "elasticnet_best.joblib"
    joblib.dump(en_search.best_estimator_, en_path)
    logger.info("Saved best ElasticNet pipeline -> %s", en_path)

    # 6. Task 3.3.2 — XGBoost Regressor (tree-mode preprocessing, no log target)
    logger.info("=== 3.3.2 XGBoost ===")
    xgb_search = train_xgboost(X_train, y_train, cv_splits)
    trained_models["xgboost"] = xgb_search

    xgb_path = artifacts_dir / "xgboost_best.joblib"
    joblib.dump(xgb_search.best_estimator_, xgb_path)
    logger.info("Saved best XGBoost pipeline -> %s", xgb_path)

    # 7. Task 3.3.3 — Random Forest Regressor (baseline, tree-mode preprocessing)
    logger.info("=== 3.3.3 Random Forest ===")
    rf_search = train_random_forest(X_train, y_train, cv_splits)
    trained_models["random_forest"] = rf_search

    rf_path = artifacts_dir / "random_forest_best.joblib"
    joblib.dump(rf_search.best_estimator_, rf_path)
    logger.info("Saved best RandomForest pipeline -> %s", rf_path)

    return trained_models


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    _parquet = "step_3_eda/daily_dataset_clean.parquet"
    run_training(_parquet)
