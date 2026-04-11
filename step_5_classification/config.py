"""Hyperparameter search spaces and constants for Step 5 — Classification models.

Strategy:
- Logistic Regression: GridSearchCV over C ∈ [0.01, 0.1, 1.0, 10.0] (4 combos — exhaustive)
- Random Forest:       RandomizedSearchCV(n_iter=50) — balances coverage vs. compute
- XGBoost:            RandomizedSearchCV(n_iter=50) — full grid far too large for exhaustive search

All CV scoring uses ``f1_macro`` with temporal CV splits (no shuffling) to respect
time order and avoid leakage. Class imbalance is handled via class_weight="balanced"
and compute_sample_weight for XGBoost.
"""

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Alert class label encoding (ordinal severity order)
# ---------------------------------------------------------------------------
LABEL_MAP: Dict[str, int] = {
    "verde": 0,
    "giallo": 1,
    "arancio": 2,
    "rosso": 3,
}

# ---------------------------------------------------------------------------
# Target and drop columns
# ---------------------------------------------------------------------------
TARGET_COL: str = "classe_allerta"
DROP_COLS = ["idstazione", "nomestazione", "comune", "pm10"]

# ---------------------------------------------------------------------------
# Random seed — fixed for reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
N_CV_SPLITS: int = 5
CV_SCORING: str = "f1_macro"

# ---------------------------------------------------------------------------
# RandomizedSearchCV shared settings
# ---------------------------------------------------------------------------
RANDOM_SEARCH_N_ITER: int = 50
RANDOM_SEARCH_N_JOBS: int = -1  # use all available cores

# ---------------------------------------------------------------------------
# Logistic Regression — GridSearchCV
# 4 values of C, elasticnet penalty with fixed l1_ratio=0.5 (set in train.py)
# ---------------------------------------------------------------------------
LOGISTIC_PARAM_GRID: Dict[str, Any] = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0],
}

# ---------------------------------------------------------------------------
# Random Forest Classifier — RandomizedSearchCV(n_iter=50)
# Full grid: 4×4×3 = 48; random search for consistency with other models
# ---------------------------------------------------------------------------
RANDOM_FOREST_PARAM_DIST: Dict[str, Any] = {
    "n_estimators": [100, 300, 500, 800],
    "max_depth": [10, 20, 30, None],
    "min_samples_leaf": [1, 2, 5],
}

# ---------------------------------------------------------------------------
# XGBoost Classifier — RandomizedSearchCV(n_iter=50)
# multi:softprob with num_class=4; sample weights handle class imbalance
# Full grid: 4×4×3×3×3×3 = 1296; random search covers ~4% of space
# ---------------------------------------------------------------------------
XGBOOST_PARAM_DIST: Dict[str, Any] = {
    "n_estimators": [100, 300, 500, 800],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR: str = "step_5_classification/artifacts"
PLOTS_DIR: str = "step_5_classification/artifacts/plots"
METRICS_FILE: str = "step_5_classification/artifacts/classification_metrics.json"
