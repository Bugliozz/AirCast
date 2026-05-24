"""Out-of-fold tuning for the hybrid regression/classifier alert strategy.

Usage from project root:

    python -m step_5_classification.tune_hybrid
"""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

from shared.regression_to_class import (
    ALERT_THRESHOLDS,
    CLASS_NAMES,
    hybrid_regression_classifier_decision,
    order_class_probabilities,
    pm10_to_alert_class,
)
from shared.utils import make_temporal_cv_splits
from step_4_regression.config import (
    ARTIFACTS_DIR as REGRESSION_ARTIFACTS_DIR,
    METRICS_FILE as REGRESSION_METRICS_FILE,
)
from step_5_classification.calibrate import rank_hybrid_classifier_candidates
from step_5_classification.config import (
    ARTIFACTS_DIR,
    HYBRID_ALPHA,
    HYBRID_BETA,
    HYBRID_DELTA_GRID,
    HYBRID_PROB_GRID,
    LABEL_MAP,
    METRICS_FILE,
    N_CV_SPLITS,
)
from step_5_classification.guardrails import load_json, write_json

logger = logging.getLogger(__name__)

DATE_COL = "data_giorno"
CLASS_TARGET_COL = "classe_allerta"
PM10_TARGET_COL = "pm10"
DEFAULT_PARQUET_PATH = "step_3_eda/daily_dataset_clean.parquet"
OOF_FILENAME = "hybrid_oof_predictions.parquet"
PARAMS_FILENAME = "hybrid_params.json"
FLAT_GRID_SCORE_EPS = 0.005


def _utcnow_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _load_best_regressor_name() -> str:
    metrics_path = Path(REGRESSION_METRICS_FILE)
    if not metrics_path.exists():
        return "best_model"
    return str(load_json(metrics_path).get("best_model", "best_model"))


def _select_classifier_name(metrics_path: Path) -> str:
    if not metrics_path.exists():
        logger.warning("Metrics file missing; defaulting hybrid classifier to xgboost.")
        return "xgboost"

    metrics_doc = load_json(metrics_path)
    support = metrics_doc.get("hybrid_support_classifier")
    if isinstance(support, Mapping) and support.get("selected_model"):
        return str(support["selected_model"])

    ranking = rank_hybrid_classifier_candidates(metrics_doc)
    return str(ranking[0]["model_name"])


def _load_train_payload(artifacts_dir: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    train_path = artifacts_dir / "train_data.joblib"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Train data artifact not found at '{train_path}'. "
            "Run 'python -m step_5_classification.train' first."
        )

    payload = joblib.load(train_path)
    X_train = payload["X_train"].reset_index(drop=True)
    y_train = payload["y_train"].astype(int).reset_index(drop=True)
    train_dates = pd.Series(payload["train_dates"]).reset_index(drop=True)
    if not (len(X_train) == len(y_train) == len(train_dates)):
        raise ValueError("X_train, y_train, and train_dates are not aligned.")
    return X_train, y_train, train_dates


def _load_pm10_train_targets(
    parquet_path: Path,
    train_dates: pd.Series,
    y_train: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Rebuild the saved temporal train rows and return PM10 plus row ids."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Clean parquet not found at '{parquet_path}'.")

    df = pd.read_parquet(parquet_path).copy()
    df["__original_index"] = np.arange(len(df), dtype=np.int64)
    df_sorted = df.sort_values(DATE_COL).reset_index(drop=True)
    sorted_dates = pd.to_datetime(df_sorted[DATE_COL], errors="raise")
    train_dates_dt = pd.to_datetime(train_dates, errors="raise")

    train_mask = (
        (sorted_dates >= train_dates_dt.min())
        & (sorted_dates <= train_dates_dt.max())
    )
    train_df = df_sorted.loc[train_mask].copy()
    mapped_labels = train_df[CLASS_TARGET_COL].map(LABEL_MAP)
    valid_rows = mapped_labels.notna()
    train_df = train_df.loc[valid_rows].reset_index(drop=True)
    mapped_labels = mapped_labels.loc[valid_rows].astype(int).reset_index(drop=True)

    if len(train_df) != len(y_train):
        raise ValueError(
            "Rebuilt train rows do not match train_data.joblib: "
            f"{len(train_df)} != {len(y_train)}."
        )
    if not np.array_equal(mapped_labels.to_numpy(), y_train.to_numpy()):
        raise ValueError("Rebuilt train labels do not match train_data.joblib.")

    return (
        train_df[PM10_TARGET_COL].astype(float).reset_index(drop=True),
        train_df["__original_index"].astype(int).reset_index(drop=True),
    )


def _fit_calibrated_classifier_on_fold(
    base_classifier: object,
    X_fold: pd.DataFrame,
    y_fold: pd.Series,
    dates_fold: pd.Series,
) -> CalibratedClassifierCV:
    """Fit an isotonic calibrator using only the current outer train fold."""
    X_local = X_fold.reset_index(drop=True)
    y_local = y_fold.astype(int).reset_index(drop=True)
    dates_local = pd.Series(dates_fold).reset_index(drop=True)

    X_with_dates = X_local.copy()
    X_with_dates.insert(0, DATE_COL, dates_local)
    inner_cv = make_temporal_cv_splits(
        X_with_dates,
        y_local,
        n_splits=N_CV_SPLITS,
    )

    calibrated = CalibratedClassifierCV(
        estimator=clone(base_classifier),
        method="isotonic",
        cv=inner_cv,
    )
    calibrated.fit(X_local, y_local)
    return calibrated


def build_oof_predictions(
    *,
    artifacts_dir: Path,
    parquet_path: Path,
    regressor_path: Path,
    classifier_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Build and persist out-of-fold predictions used for hybrid tuning."""
    X_train, y_train, train_dates = _load_train_payload(artifacts_dir)
    y_pm10_train, original_indices = _load_pm10_train_targets(
        parquet_path=parquet_path,
        train_dates=train_dates,
        y_train=y_train,
    )

    if not regressor_path.exists():
        raise FileNotFoundError(f"Regression artifact not found: {regressor_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier artifact not found: {classifier_path}")

    base_regressor = joblib.load(regressor_path)
    base_classifier = joblib.load(classifier_path)

    X_with_dates = X_train.copy()
    X_with_dates.insert(0, DATE_COL, train_dates)
    outer_cv = make_temporal_cv_splits(
        X_with_dates,
        y_train,
        n_splits=N_CV_SPLITS,
    )

    n_samples = len(X_train)
    pm10_hat_oof = np.full(n_samples, np.nan, dtype=float)
    proba_oof = np.full((n_samples, len(CLASS_NAMES)), np.nan, dtype=float)
    fold_oof = np.full(n_samples, -1, dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv, start=1):
        logger.info(
            "Hybrid OOF fold %d/%d: train=%d rows, val=%d rows",
            fold_idx,
            len(outer_cv),
            len(train_idx),
            len(val_idx),
        )

        X_fold_train = X_train.iloc[train_idx].reset_index(drop=True)
        y_pm10_fold = y_pm10_train.iloc[train_idx].reset_index(drop=True)
        y_cls_fold = y_train.iloc[train_idx].reset_index(drop=True)
        dates_fold = train_dates.iloc[train_idx].reset_index(drop=True)
        X_fold_val = X_train.iloc[val_idx]

        regressor = clone(base_regressor)
        regressor.fit(X_fold_train, y_pm10_fold)
        pm10_hat_oof[val_idx] = np.asarray(regressor.predict(X_fold_val), dtype=float)

        calibrated_classifier = _fit_calibrated_classifier_on_fold(
            base_classifier=base_classifier,
            X_fold=X_fold_train,
            y_fold=y_cls_fold,
            dates_fold=dates_fold,
        )
        raw_proba = calibrated_classifier.predict_proba(X_fold_val)
        proba_oof[val_idx, :] = order_class_probabilities(
            raw_proba,
            getattr(calibrated_classifier, "classes_", None),
            n_classes=len(CLASS_NAMES),
        )
        fold_oof[val_idx] = fold_idx

    valid_oof = fold_oof >= 0
    if not np.any(valid_oof):
        raise RuntimeError("No OOF predictions were produced.")
    if np.isnan(pm10_hat_oof[valid_oof]).any() or np.isnan(proba_oof[valid_oof]).any():
        raise RuntimeError("OOF prediction cache is incomplete for validation rows.")
    logger.info(
        "Hybrid OOF cache covers %d/%d train rows; the initial TimeSeriesSplit "
        "training block is excluded from tuning.",
        int(valid_oof.sum()),
        n_samples,
    )

    class_reg_oof = pm10_to_alert_class(
        pm10_hat_oof[valid_oof],
        thresholds=ALERT_THRESHOLDS,
    )
    oof = pd.DataFrame(
        {
            "original_index": original_indices.to_numpy(dtype=int)[valid_oof],
            "y_true": y_train.to_numpy(dtype=int)[valid_oof],
            "pm10_hat_oof": pm10_hat_oof[valid_oof],
            "class_reg_oof": class_reg_oof,
            "fold": fold_oof[valid_oof],
        }
    )
    for class_idx, class_name in enumerate(CLASS_NAMES):
        oof[f"proba_cls_oof_{class_name}"] = proba_oof[valid_oof, class_idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(output_path, index=False)
    logger.info("Hybrid OOF predictions saved -> %s", output_path)
    return oof


def _tuning_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    return {
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "precision_rosso": float(report["rosso"]["precision"]),
        "recall_rosso": float(report["rosso"]["recall"]),
        "severe_error_rate": float((np.abs(y_pred_arr - y_true_arr) >= 2).mean()),
        "over_alert_rate": float((y_pred_arr > y_true_arr).mean()),
        "under_alert_rate": float((y_pred_arr < y_true_arr).mean()),
    }


def tune_from_oof(
    oof: pd.DataFrame,
    *,
    delta_grid: List[float],
    prob_grid: List[float],
    alpha: float,
    beta: float,
) -> Tuple[Dict, List[Dict]]:
    """Grid-search the hybrid rule using only OOF predictions."""
    y_true = oof["y_true"].to_numpy(dtype=int)
    pm10_hat = oof["pm10_hat_oof"].to_numpy(dtype=float)
    proba = oof[[f"proba_cls_oof_{name}" for name in CLASS_NAMES]].to_numpy(dtype=float)

    score_curve: List[Dict] = []
    for delta in delta_grid:
        for p_threshold in prob_grid:
            decision = hybrid_regression_classifier_decision(
                pm10_hat,
                proba,
                delta=float(delta),
                p_threshold=float(p_threshold),
                thresholds=ALERT_THRESHOLDS,
            )
            y_pred = decision["class_final"]
            metrics = _tuning_metrics(y_true, y_pred)
            score = (
                metrics["f1_macro"]
                - float(alpha) * metrics["severe_error_rate"]
                - float(beta) * (1.0 - metrics["recall_rosso"])
            )
            score_curve.append(
                {
                    "delta": float(delta),
                    "p_threshold": float(p_threshold),
                    "score": float(score),
                    **metrics,
                    "hybrid_override_rate": float((y_pred != decision["class_reg"]).mean()),
                    "hybrid_escalation_rate": float(decision["escalation_mask"].mean()),
                    "n_overrides": int((y_pred != decision["class_reg"]).sum()),
                    "n_escalations": int(decision["escalation_mask"].sum()),
                }
            )

    best = max(
        score_curve,
        key=lambda row: (
            row["score"],
            row["f1_macro"],
            row["recall_rosso"],
            -row["severe_error_rate"],
            -row["over_alert_rate"],
            -row["delta"],
            row["p_threshold"],
        ),
    )
    return best, score_curve


def run_tuning(
    artifacts_dir_path: Optional[str] = None,
    parquet_path: str = DEFAULT_PARQUET_PATH,
) -> Dict:
    """Build OOF predictions, tune (delta, p_threshold), and save JSON artifacts."""
    artifacts_dir = Path(artifacts_dir_path or ARTIFACTS_DIR)
    metrics_path = Path(METRICS_FILE)
    regressor_name = _load_best_regressor_name()
    classifier_name = _select_classifier_name(metrics_path)

    regressor_path = Path(REGRESSION_ARTIFACTS_DIR) / "best_model.joblib"
    classifier_path = artifacts_dir / f"{classifier_name}_best.joblib"
    oof_path = artifacts_dir / OOF_FILENAME
    params_path = artifacts_dir / PARAMS_FILENAME

    logger.info(
        "Tuning hybrid strategy with regressor=%s, classifier=%s",
        regressor_name,
        classifier_name,
    )
    oof = build_oof_predictions(
        artifacts_dir=artifacts_dir,
        parquet_path=Path(parquet_path),
        regressor_path=regressor_path,
        classifier_path=classifier_path,
        output_path=oof_path,
    )
    best, score_curve = tune_from_oof(
        oof,
        delta_grid=[float(v) for v in HYBRID_DELTA_GRID],
        prob_grid=[float(v) for v in HYBRID_PROB_GRID],
        alpha=float(HYBRID_ALPHA),
        beta=float(HYBRID_BETA),
    )

    scores = [row["score"] for row in score_curve]
    grid_score_range = float(max(scores) - min(scores))
    grid_is_flat = grid_score_range < FLAT_GRID_SCORE_EPS
    if grid_is_flat:
        logger.warning(
            "Hybrid grid is flat: score range %.6f < %.6f. "
            "The threshold zone is not adding useful signal; hybrid degenerates "
            "toward regression_to_class.",
            grid_score_range,
            FLAT_GRID_SCORE_EPS,
        )

    params_doc = {
        "strategy_name": f"hybrid_{regressor_name}_reg_{classifier_name}_cls",
        "regressor_name": regressor_name,
        "classifier_name": classifier_name,
        "alert_thresholds": [float(v) for v in ALERT_THRESHOLDS],
        "best_delta": float(best["delta"]),
        "best_p_threshold": float(best["p_threshold"]),
        "best_oof_score": float(best["score"]),
        "best_oof_metrics": {
            key: best[key]
            for key in (
                "f1_macro",
                "precision_rosso",
                "recall_rosso",
                "severe_error_rate",
                "over_alert_rate",
                "under_alert_rate",
                "hybrid_override_rate",
                "hybrid_escalation_rate",
                "n_overrides",
                "n_escalations",
            )
        },
        "objective": {
            "formula": "f1_macro - alpha * severe_error_rate - beta * (1 - recall_rosso)",
            "alpha": float(HYBRID_ALPHA),
            "beta": float(HYBRID_BETA),
        },
        "score_curve": score_curve,
        "grid_is_flat": bool(grid_is_flat),
        "grid_score_range": grid_score_range,
        "flat_grid_threshold": float(FLAT_GRID_SCORE_EPS),
        "oof_predictions_path": oof_path.as_posix(),
        "generated_at": _utcnow_iso(),
    }

    write_json(params_path, params_doc)
    logger.info(
        "Hybrid params saved -> %s (delta=%.3f, p_threshold=%.3f, score=%.4f)",
        params_path,
        params_doc["best_delta"],
        params_doc["best_p_threshold"],
        params_doc["best_oof_score"],
    )
    return params_doc


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run_tuning()
