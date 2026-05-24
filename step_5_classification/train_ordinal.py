"""Phase 3 benchmark: OrdinalClassifier (Frank & Hall 2001) wrapping XGBoost.

Inherits hyperparameters from the already-tuned xgboost_best.joblib — no new
RandomizedSearchCV. Phase 3 is an informational benchmark, not a production
candidate, so re-tuning is not warranted and the cost of 3 × 50 × 5 extra fits
would exceed the signal gained.

Usage (from project root)::

    python -m step_5_classification.train_ordinal
"""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path
from typing import Dict

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from shared.utils import build_preprocessor
from step_5_classification.config import (
    ARTIFACTS_DIR,
    METRICS_FILE,
    PLOTS_DIR,
    RANDOM_STATE,
)
from step_5_classification.evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_permutation_importance,
)
from step_5_classification.guardrails import load_json, write_json
from step_5_classification.ordinal_classifier import OrdinalClassifier

logger = logging.getLogger(__name__)

ORDINAL_STRATEGY_NAME = "xgboost_ordinal"
ORDINAL_ARTIFACT_NAME = "xgboost_ordinal.joblib"

# Only inherit the params that were actually tuned in the XGBoost search space;
# objective and num_class are multiclass-specific and must be replaced.
_TUNED_PARAMS = frozenset({
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "min_child_weight",
    "colsample_bytree",
})


def _extract_tuned_xgb_params(artifacts_dir: Path) -> Dict:
    """Load the best XGBoost pipeline and extract its tuned hyperparameters."""
    xgb_path = artifacts_dir / "xgboost_best.joblib"
    if not xgb_path.exists():
        raise FileNotFoundError(
            f"xgboost_best.joblib not found at '{xgb_path}'. "
            "Run 'python -m step_5_classification.train' first."
        )
    pipeline: Pipeline = joblib.load(xgb_path)
    xgb_clf: XGBClassifier = pipeline.named_steps["classifier"]
    all_params = xgb_clf.get_params()
    tuned = {k: v for k, v in all_params.items() if k in _TUNED_PARAMS}
    logger.info("Inherited tuned XGBoost params: %s", tuned)
    return tuned


def build_ordinal_pipeline(artifacts_dir: Path, X_train) -> Pipeline:
    """Build Pipeline(ColumnTransformer[tree], OrdinalClassifier(XGBClassifier))."""
    tuned_params = _extract_tuned_xgb_params(artifacts_dir)

    xgb_binary = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        **tuned_params,
    )

    ordinal = OrdinalClassifier(xgb_binary, balanced_weights=True)
    preprocessor = build_preprocessor(X_train, model_type="tree")

    return Pipeline(steps=[("pre", preprocessor), ("classifier", ordinal)])


def run() -> Dict:
    """Train ordinal XGBoost and evaluate on the held-out test set."""
    artifacts_dir = Path(ARTIFACTS_DIR)

    train_path = artifacts_dir / "train_data.joblib"
    test_path = artifacts_dir / "test_data.joblib"
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Artifact not found: '{p}'. "
                "Run 'python -m step_5_classification.train' first."
            )

    train_payload = joblib.load(train_path)
    test_payload = joblib.load(test_path)

    X_train = train_payload["X_train"]
    y_train = train_payload["y_train"].astype(int)
    X_test = test_payload["X_test"]
    y_test = test_payload["y_test"].astype(int)

    logger.info(
        "Data loaded — train: %d samples, test: %d samples.",
        len(X_train), len(X_test),
    )
    logger.info(
        "Label distribution (train): %s",
        dict(zip(*np.unique(y_train, return_counts=True))),
    )
    logger.info(
        "Label distribution (test):  %s",
        dict(zip(*np.unique(y_test, return_counts=True))),
    )

    pipeline = build_ordinal_pipeline(artifacts_dir, X_train)

    logger.info(
        "Fitting OrdinalClassifier(XGBoost) — 3 binary classifiers "
        "(k=0: P(y>0), k=1: P(y>1), k=2: P(y>2)) ..."
    )
    pipeline.fit(X_train, y_train)
    logger.info("Fit complete.")

    artifact_path = artifacts_dir / ORDINAL_ARTIFACT_NAME
    joblib.dump(pipeline, artifact_path)
    logger.info("Saved ordinal pipeline -> %s", artifact_path)

    y_pred = np.asarray(pipeline.predict(X_test), dtype=int)
    logger.info("--- %s ---", ORDINAL_STRATEGY_NAME.upper())
    metrics = compute_classification_metrics(
        y_test,
        y_pred,
        model_name=ORDINAL_STRATEGY_NAME,
        strategy_name=ORDINAL_STRATEGY_NAME,
    )

    plot_confusion_matrix(y_test, y_pred, ORDINAL_STRATEGY_NAME, PLOTS_DIR)
    plot_permutation_importance(
        pipeline, X_test, y_test, ORDINAL_STRATEGY_NAME, PLOTS_DIR
    )

    metrics_path = Path(METRICS_FILE)
    metrics_doc = load_json(metrics_path) if metrics_path.exists() else {}
    models_doc = metrics_doc.setdefault("models", {})
    models_doc[ORDINAL_STRATEGY_NAME] = metrics
    metrics_doc["generated_at"] = (
        datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    )
    write_json(metrics_path, metrics_doc)
    logger.info(
        "classification_metrics.json updated with '%s' -> %s",
        ORDINAL_STRATEGY_NAME,
        metrics_path,
    )

    logger.info(
        "Ordinal result — f1_macro=%.4f  recall_rosso=%.4f  severe_error_rate=%.4f  "
        "over_alert_rate=%.4f",
        metrics["f1_macro"],
        metrics["recall_rosso"],
        metrics["severe_error_rate"],
        metrics["over_alert_rate"],
    )

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run()
