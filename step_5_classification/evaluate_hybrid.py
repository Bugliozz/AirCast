"""Final holdout evaluation for the hybrid alert strategy.

Usage from project root:

    python -m step_5_classification.evaluate_hybrid
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from shared.regression_to_class import (
    ALERT_THRESHOLDS,
    CLASS_NAMES,
    hybrid_regression_classifier_decision,
    order_class_probabilities,
)
from step_4_regression.config import ARTIFACTS_DIR as REGRESSION_ARTIFACTS_DIR
from step_5_classification.config import ARTIFACTS_DIR, LABEL_MAP, METRICS_FILE, PLOTS_DIR
from step_5_classification.evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
)
from step_5_classification.guardrails import load_json, write_json
from step_5_classification.tune_hybrid import (
    CLASS_TARGET_COL,
    DATE_COL,
    DEFAULT_PARQUET_PATH,
    PARAMS_FILENAME,
    PM10_TARGET_COL,
    _load_best_regressor_name,
    _select_classifier_name,
    _utcnow_iso,
)

logger = logging.getLogger(__name__)

HYBRID_METRICS_FILENAME = "hybrid_metrics.json"


def _load_test_payload(artifacts_dir: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    test_path = artifacts_dir / "test_data.joblib"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data artifact not found at '{test_path}'. "
            "Run 'python -m step_5_classification.train' first."
        )
    payload = joblib.load(test_path)
    X_test = payload["X_test"].reset_index(drop=True)
    y_test = payload["y_test"].astype(int).reset_index(drop=True)
    test_dates = pd.Series(payload["test_dates"]).reset_index(drop=True)
    if not (len(X_test) == len(y_test) == len(test_dates)):
        raise ValueError("X_test, y_test, and test_dates are not aligned.")
    return X_test, y_test, test_dates


def _load_pm10_test_targets(
    parquet_path: Path,
    test_dates: pd.Series,
    y_test: pd.Series,
) -> pd.Series:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Clean parquet not found at '{parquet_path}'.")

    df = pd.read_parquet(parquet_path).copy()
    df_sorted = df.sort_values(DATE_COL).reset_index(drop=True)
    sorted_dates = pd.to_datetime(df_sorted[DATE_COL], errors="raise")
    test_dates_dt = pd.to_datetime(test_dates, errors="raise")

    test_mask = (
        (sorted_dates >= test_dates_dt.min())
        & (sorted_dates <= test_dates_dt.max())
    )
    test_df = df_sorted.loc[test_mask].copy()
    mapped_labels = test_df[CLASS_TARGET_COL].map(LABEL_MAP)
    valid_rows = mapped_labels.notna()
    test_df = test_df.loc[valid_rows].reset_index(drop=True)
    mapped_labels = mapped_labels.loc[valid_rows].astype(int).reset_index(drop=True)

    if len(test_df) != len(y_test):
        raise ValueError(
            "Rebuilt test rows do not match test_data.joblib: "
            f"{len(test_df)} != {len(y_test)}."
        )
    if not np.array_equal(mapped_labels.to_numpy(), y_test.to_numpy()):
        raise ValueError("Rebuilt test labels do not match test_data.joblib.")

    return test_df[PM10_TARGET_COL].astype(float).reset_index(drop=True)


def _load_hybrid_params(params_path: Path) -> Dict:
    if not params_path.exists():
        raise FileNotFoundError(
            f"Hybrid params not found at '{params_path}'. "
            "Run 'python -m step_5_classification.tune_hybrid' first."
        )
    params = load_json(params_path)
    for key in ("best_delta", "best_p_threshold"):
        if key not in params:
            raise ValueError(f"hybrid_params.json missing required key: {key}")
    return params


def _load_calibrated_classifier(artifacts_dir: Path, classifier_name: str) -> Tuple[object, Path]:
    candidate_path = artifacts_dir / f"{classifier_name}_hybrid_calibrated.joblib"
    legacy_path = artifacts_dir / "best_model_calibrated.joblib"
    for path in (candidate_path, legacy_path):
        if path.exists():
            return joblib.load(path), path
    raise FileNotFoundError(
        "Calibrated classifier artifact not found. "
        "Run 'python -m step_5_classification.calibrate' first."
    )


def _rmse_rosso(
    y_test: pd.Series,
    pm10_true: pd.Series,
    pm10_hat: np.ndarray,
) -> Optional[float]:
    red_mask = y_test.to_numpy(dtype=int) == 3
    if not np.any(red_mask):
        return None
    errors = np.asarray(pm10_hat, dtype=float)[red_mask] - pm10_true.to_numpy(dtype=float)[red_mask]
    return float(np.sqrt(np.mean(np.square(errors))))


def _extract_existing_metrics(metrics_doc: Mapping[str, object], key: str) -> Optional[Dict]:
    models = metrics_doc.get("models")
    if isinstance(models, Mapping):
        value = models.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return None


def run_evaluation(
    artifacts_dir_path: Optional[str] = None,
    parquet_path: str = DEFAULT_PARQUET_PATH,
) -> Dict:
    """Evaluate the tuned hybrid strategy on the untouched test set."""
    artifacts_dir = Path(artifacts_dir_path or ARTIFACTS_DIR)
    params_path = artifacts_dir / PARAMS_FILENAME
    metrics_path = Path(METRICS_FILE)
    hybrid_metrics_path = artifacts_dir / HYBRID_METRICS_FILENAME

    params = _load_hybrid_params(params_path)
    regressor_name = str(params.get("regressor_name") or _load_best_regressor_name())
    classifier_name = str(params.get("classifier_name") or _select_classifier_name(metrics_path))
    strategy_name = str(
        params.get("strategy_name")
        or f"hybrid_{regressor_name}_reg_{classifier_name}_cls"
    )
    delta = float(params["best_delta"])
    p_threshold = float(params["best_p_threshold"])

    X_test, y_test, test_dates = _load_test_payload(artifacts_dir)
    y_pm10_test = _load_pm10_test_targets(
        parquet_path=Path(parquet_path),
        test_dates=test_dates,
        y_test=y_test,
    )

    regressor_path = Path(REGRESSION_ARTIFACTS_DIR) / "best_model.joblib"
    if not regressor_path.exists():
        raise FileNotFoundError(f"Regression artifact not found: {regressor_path}")
    regressor = joblib.load(regressor_path)
    calibrated_classifier, calibrated_path = _load_calibrated_classifier(
        artifacts_dir,
        classifier_name,
    )

    pm10_hat = np.asarray(regressor.predict(X_test), dtype=float)
    raw_proba = calibrated_classifier.predict_proba(X_test)
    proba = order_class_probabilities(
        raw_proba,
        getattr(calibrated_classifier, "classes_", None),
        n_classes=len(CLASS_NAMES),
    )
    decision = hybrid_regression_classifier_decision(
        pm10_hat,
        proba,
        delta=delta,
        p_threshold=p_threshold,
        thresholds=ALERT_THRESHOLDS,
    )

    class_reg = decision["class_reg"]
    class_calibrated = np.argmax(proba, axis=1).astype(int)
    class_hybrid = decision["class_final"]

    regression_strategy_name = f"regression_to_class_{regressor_name}"
    calibrated_strategy_name = f"classifier_{classifier_name}_calibrated"

    regression_metrics = compute_classification_metrics(
        y_test,
        class_reg,
        regression_strategy_name,
        strategy_name=regression_strategy_name,
    )
    calibrated_metrics = compute_classification_metrics(
        y_test,
        class_calibrated,
        calibrated_strategy_name,
        strategy_name=calibrated_strategy_name,
    )
    hybrid_metrics = compute_classification_metrics(
        y_test,
        class_hybrid,
        strategy_name,
        strategy_name=strategy_name,
        class_reg_reference=class_reg,
    )
    hybrid_metrics.update(
        {
            "delta": delta,
            "p_threshold": p_threshold,
            "regressor_name": regressor_name,
            "classifier_name": classifier_name,
            "regression_artifact": regressor_path.as_posix(),
            "calibrated_classifier_artifact": calibrated_path.as_posix(),
            "threshold_zone_rate": float(decision["in_threshold_zone"].mean()),
            "discordance_rate": float(decision["discordant"].mean()),
            "prudential_activation_rate": float(decision["prudential_mask"].mean()),
            "rmse_rosso": _rmse_rosso(y_test, y_pm10_test, pm10_hat),
        }
    )

    plot_confusion_matrix(y_test, class_hybrid, strategy_name, PLOTS_DIR)
    plot_confusion_matrix(y_test, class_calibrated, calibrated_strategy_name, PLOTS_DIR)

    metrics_doc = load_json(metrics_path) if metrics_path.exists() else {}
    uncalibrated_metrics = _extract_existing_metrics(metrics_doc, classifier_name)

    comparison = {
        regression_strategy_name: regression_metrics,
        calibrated_strategy_name: calibrated_metrics,
        strategy_name: hybrid_metrics,
    }
    if uncalibrated_metrics is not None:
        comparison[classifier_name] = uncalibrated_metrics

    tradeoff = {
        "hybrid_vs_regression_to_class": {
            "delta_recall_rosso": float(
                hybrid_metrics["recall_rosso"] - regression_metrics["recall_rosso"]
            ),
            "delta_over_alert_rate": float(
                hybrid_metrics["over_alert_rate"] - regression_metrics["over_alert_rate"]
            ),
            "delta_severe_error_rate": float(
                hybrid_metrics["severe_error_rate"]
                - regression_metrics["severe_error_rate"]
            ),
            "delta_f1_macro": float(
                hybrid_metrics["f1_macro"] - regression_metrics["f1_macro"]
            ),
        },
        "hybrid_vs_classifier_calibrated": {
            "delta_recall_rosso": float(
                hybrid_metrics["recall_rosso"] - calibrated_metrics["recall_rosso"]
            ),
            "delta_over_alert_rate": float(
                hybrid_metrics["over_alert_rate"] - calibrated_metrics["over_alert_rate"]
            ),
            "delta_severe_error_rate": float(
                hybrid_metrics["severe_error_rate"]
                - calibrated_metrics["severe_error_rate"]
            ),
            "delta_f1_macro": float(
                hybrid_metrics["f1_macro"] - calibrated_metrics["f1_macro"]
            ),
        },
    }
    if uncalibrated_metrics is not None:
        tradeoff["hybrid_vs_classifier_uncalibrated"] = {
            "delta_recall_rosso": float(
                hybrid_metrics["recall_rosso"] - uncalibrated_metrics["recall_rosso"]
            ),
            "delta_over_alert_rate": float(
                hybrid_metrics["over_alert_rate"]
                - uncalibrated_metrics["over_alert_rate"]
            ),
            "delta_severe_error_rate": float(
                hybrid_metrics["severe_error_rate"]
                - uncalibrated_metrics["severe_error_rate"]
            ),
            "delta_f1_macro": float(
                hybrid_metrics["f1_macro"] - uncalibrated_metrics["f1_macro"]
            ),
        }

    hybrid_doc = {
        "generated_at": _utcnow_iso(),
        "params_path": params_path.as_posix(),
        "hybrid_params": {
            "delta": delta,
            "p_threshold": p_threshold,
            "oof_score": params.get("best_oof_score"),
            "grid_is_flat": params.get("grid_is_flat"),
        },
        "models": comparison,
        "tradeoff": tradeoff,
        "test_set_note": "OOF cache is used only for tuning; these metrics use the held-out test set.",
    }
    write_json(hybrid_metrics_path, hybrid_doc)
    logger.info("Hybrid metrics saved -> %s", hybrid_metrics_path)

    models_doc = metrics_doc.setdefault("models", {})
    models_doc[regression_strategy_name] = regression_metrics
    models_doc[calibrated_strategy_name] = calibrated_metrics
    models_doc[strategy_name] = hybrid_metrics
    metrics_doc["hybrid_evaluation"] = {
        "hybrid_metrics_path": hybrid_metrics_path.as_posix(),
        "params_path": params_path.as_posix(),
        "tradeoff": tradeoff,
        "generated_at": hybrid_doc["generated_at"],
    }
    evaluated = {
        key: value
        for key, value in models_doc.items()
        if isinstance(value, Mapping) and "f1_macro" in value
    }
    if evaluated:
        metrics_doc["best_evaluated_model"] = max(
            evaluated,
            key=lambda key: float(evaluated[key]["f1_macro"]),
        )
    write_json(metrics_path, metrics_doc)
    logger.info("classification_metrics.json updated with hybrid rows -> %s", metrics_path)

    logger.info(
        "Hybrid tradeoff vs regression_to_class: recall_rosso %+0.4f, "
        "over_alert_rate %+0.4f",
        tradeoff["hybrid_vs_regression_to_class"]["delta_recall_rosso"],
        tradeoff["hybrid_vs_regression_to_class"]["delta_over_alert_rate"],
    )
    return hybrid_doc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run_evaluation()
