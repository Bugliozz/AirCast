"""Hyperparameter search spaces for Step 4 — Regression models.

Strategy:
- ElasticNet: GridSearchCV (4×4 = 16 combinations — exhaustive is feasible)
- XGBoost:    RandomizedSearchCV(n_iter=50) — full grid is 4×4×3×3×3×3=1296 combos × 5 folds;
              random search covers ~50/1296 ≈ 4% but targets the most impactful dimensions
- RandomForest: RandomizedSearchCV(n_iter=50) — same rationale

All CV scoring uses ``neg_root_mean_squared_error`` with TimeSeriesSplit(5)
to respect temporal order and avoid leakage.
"""

from typing import Any, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Random seed — fixed for reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
N_CV_SPLITS: int = 5
CV_SCORING: str = "neg_root_mean_squared_error"

# ---------------------------------------------------------------------------
# RandomizedSearchCV shared settings
# ---------------------------------------------------------------------------
RANDOM_SEARCH_N_ITER: int = 50
RANDOM_SEARCH_N_JOBS: int = -1  # use all available cores

# ---------------------------------------------------------------------------
# ElasticNet — GridSearchCV
# 6 alpha × 4 l1_ratio = 24 combinations (exhaustive)
# ---------------------------------------------------------------------------
ELASTICNET_PARAM_GRID: Dict[str, Any] = {
    # Path: Pipeline step "regressor" -> TransformedTargetRegressor.regressor -> ElasticNet.alpha
    # NOTE: target is log1p-transformed (~[0, 5.5]); alpha range spans several orders of
    # magnitude to cover the correct regularisation strength for this scale.
    "regressor__regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    # Mix between L1 (sparse) and L2 (shrinkage)
    "regressor__regressor__l1_ratio": [0.1, 0.5, 0.7, 0.9],
}

ELASTICNET_FIT_PARAMS: Dict[str, Any] = {
    "max_iter": 2000,
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# XGBoost Regressor — RandomizedSearchCV(n_iter=50)
# Full grid: 4×4×3×3×3×3 = 1296; random search covers ~4% of space
# New params: min_child_weight (regularise peaks), colsample_bytree (decorrelate trees)
# ---------------------------------------------------------------------------
XGBOOST_PARAM_DIST: Dict[str, Any] = {
    # Number of boosting rounds
    "n_estimators": [100, 300, 500, 800],
    # Maximum tree depth — 9 allows capturing complex stagnation×lag×province interactions
    "max_depth": [3, 5, 7, 9],
    # Step size shrinkage — lower = more robust, needs more rounds
    "learning_rate": [0.01, 0.05, 0.1],
    # Fraction of training samples used per tree
    "subsample": [0.7, 0.8, 1.0],
    # Minimum sum of instance weights in a leaf — reduces overfitting on rare peaks
    "min_child_weight": [1, 3, 5],
    # Fraction of features used per tree — decorrelates trees, improves generalisation
    "colsample_bytree": [0.7, 0.8, 1.0],
}

XGBOOST_FIT_PARAMS: Dict[str, Any] = {
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    # Early stopping would need eval_set — omit here to keep grid-search clean
    "tree_method": "hist",  # faster on CPU
}

# ---------------------------------------------------------------------------
# Random Forest Regressor — RandomizedSearchCV(n_iter=30)
# Full grid: 3×3×3 = 27; random search still preferred for consistency
# ---------------------------------------------------------------------------
RANDOM_FOREST_PARAM_DIST: Dict[str, Any] = {
    # Number of trees in the forest
    "n_estimators": [100, 300, 500],
    # Maximum depth — None = grow until pure leaves (may overfit)
    "max_depth": [10, 20, None],
    # Minimum samples required in a leaf — controls over-fitting
    "min_samples_leaf": [2, 5, 10],
}

RANDOM_FOREST_FIT_PARAMS: Dict[str, Any] = {
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR: str = "step_4_regression/artifacts"
PLOTS_DIR: str = "step_4_regression/artifacts/plots"
METRICS_FILE: str = "step_4_regression/artifacts/regression_metrics.json"

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
TARGET_COL: str = "pm10"

# ---------------------------------------------------------------------------
# Identifier / drop columns — passed to shared.utils.temporal_train_test_split
# ---------------------------------------------------------------------------
DROP_COLS = ["idstazione", "nomestazione", "comune", "classe_allerta"]
