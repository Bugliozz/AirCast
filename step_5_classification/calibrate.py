"""Probability calibration for Step 5 — Classification (step 5.5).

Fits ``CalibratedClassifierCV`` on the best model using temporal CV splits
(isotonic regression), then generates reliability diagrams (one-vs-rest
calibration curves) comparing uncalibrated vs. calibrated probabilities for
each alert class.

Usage (from project root)::

    python -m step_5_classification.calibrate
"""

import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline

from shared.utils import make_temporal_cv_splits
from step_5_classification.config import (
    ARTIFACTS_DIR,
    METRICS_FILE,
    N_CV_SPLITS,
    PLOTS_DIR,
    RANDOM_STATE,
)
from step_5_classification.guardrails import (
    assert_cv_guardrails,
    assert_manifest_guardrails,
    assert_temporal_holdout,
    load_json,
    write_json,
)

logger = logging.getLogger(__name__)

CLASS_NAMES: List[str] = ["verde", "giallo", "arancio", "rosso"]
CLASSIFIER_CANDIDATES: Tuple[str, ...] = (
    "xgboost",
    "random_forest",
    "logistic_regression",
)
HYBRID_TIE_F1_THRESHOLD: float = 0.005
ECE_N_BINS: int = 10


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_train_data(
    artifacts_dir: Path,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict]:
    """Load training data persisted by train.py.

    Returns:
        Tuple ``(X_train, y_train, train_dates, payload)``.
    """
    train_path = artifacts_dir / "train_data.joblib"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Train data not found at '{train_path}'. "
            "Run 'python -m step_5_classification.train' first."
        )

    payload = joblib.load(train_path)
    X_train: pd.DataFrame = payload["X_train"]
    y_train: pd.Series = payload["y_train"].astype(int)
    train_dates: pd.Series = payload["train_dates"]

    logger.info(
        "Loaded train data: %d samples, %d features", len(X_train), X_train.shape[1],
    )
    return X_train, y_train, train_dates, payload


def _load_test_data(
    artifacts_dir: Path,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Load test data persisted by train.py."""
    test_path = artifacts_dir / "test_data.joblib"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at '{test_path}'. "
            "Run 'python -m step_5_classification.train' first."
        )

    payload = joblib.load(test_path)
    return payload["X_test"], payload["y_test"].astype(int), payload


# ---------------------------------------------------------------------------
# Hybrid classifier selection and calibration metrics
# ---------------------------------------------------------------------------

def _load_optional_training_manifest(artifacts_dir: Path) -> Optional[Dict]:
    """Load training_manifest.json when available.

    Older artifacts in this project pre-date the manifest. In that case the
    calibration step still validates the persisted train/test dates and CV
    splits available in train_data.joblib.
    """
    manifest_path = artifacts_dir / "training_manifest.json"
    if not manifest_path.exists():
        logger.warning(
            "training_manifest.json not found; using legacy train/test metadata."
        )
        return None
    return load_json(manifest_path)


def _load_metrics_doc(metrics_path: Path) -> Dict:
    """Load classification_metrics.json, failing clearly if absent."""
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found at '{metrics_path}'. "
            "Run 'python -m step_5_classification.evaluate' first."
        )
    return load_json(metrics_path)


def rank_hybrid_classifier_candidates(metrics_doc: Mapping[str, object]) -> List[Dict]:
    """Rank only the Step 5 classifiers admitted for the hybrid strategy."""
    models = metrics_doc.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("classification_metrics.json is missing the 'models' block.")

    ranking: List[Dict] = []
    for model_name in CLASSIFIER_CANDIDATES:
        model_metrics = models.get(model_name)
        if not isinstance(model_metrics, Mapping):
            logger.warning("Metrics missing for candidate '%s'; skipping.", model_name)
            continue
        if "f1_macro" not in model_metrics:
            logger.warning("f1_macro missing for candidate '%s'; skipping.", model_name)
            continue
        ranking.append(
            {
                "model_name": model_name,
                "f1_macro": float(model_metrics["f1_macro"]),
            }
        )

    if not ranking:
        raise ValueError(
            "No admitted Step 5 classifier has f1_macro in classification_metrics.json."
        )

    ranking.sort(key=lambda item: item["f1_macro"], reverse=True)
    return ranking


def select_hybrid_calibration_candidates(
    ranking: Sequence[Mapping[str, object]],
) -> List[Dict]:
    """Select the best classifier, plus the runner-up only within the tie band."""
    selected = [dict(ranking[0])]
    if len(ranking) > 1:
        best_f1 = float(ranking[0]["f1_macro"])
        second_f1 = float(ranking[1]["f1_macro"])
        if best_f1 - second_f1 <= HYBRID_TIE_F1_THRESHOLD:
            selected.append(dict(ranking[1]))
    return selected


def _assert_selected_artifacts_exist(
    artifacts_dir: Path,
    selected_candidates: Sequence[Mapping[str, object]],
) -> None:
    """Ensure every selected classifier has a saved Step 5 artifact."""
    for candidate in selected_candidates:
        model_name = str(candidate["model_name"])
        model_path = artifacts_dir / f"{model_name}_best.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Selected classifier artifact not found: {model_path}")


def _validate_legacy_split_metadata(
    train_dates: pd.Series,
    test_payload: Mapping[str, object],
    metrics_doc: Mapping[str, object],
) -> Dict[str, object]:
    """Validate train/test dates when no full training manifest is available."""
    test_dates = test_payload.get("test_dates")
    if test_dates is None:
        logger.warning("test_data.joblib has no test_dates; date guard is limited.")
        return {}

    train_dt, test_dt = assert_temporal_holdout(train_dates, test_dates)
    split_metadata = {
        "train_date_range": [str(train_dt.min()), str(train_dt.max())],
        "test_date_range": [str(test_dt.min()), str(test_dt.max())],
        "n_train_samples": int(len(train_dt)),
        "n_test_samples": int(len(test_dt)),
    }

    metrics_split = metrics_doc.get("split_metadata")
    if isinstance(metrics_split, Mapping):
        for key in (
            "train_date_range",
            "test_date_range",
            "n_train_samples",
            "n_test_samples",
        ):
            if key in metrics_split and metrics_split[key] != split_metadata[key]:
                raise AssertionError(
                    f"Split metadata mismatch for {key}: "
                    f"{metrics_split[key]} != {split_metadata[key]}"
                )

    return split_metadata


def _build_and_validate_calibration_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_dates: pd.Series,
    train_payload: Mapping[str, object],
    training_manifest: Optional[Mapping[str, object]],
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict]:
    """Build temporal CV splits and compare them with available metadata."""
    X_train_with_dates = X_train.copy()
    X_train_with_dates.insert(0, "data_giorno", train_dates)
    tscv = make_temporal_cv_splits(
        X_train_with_dates, y_train, n_splits=N_CV_SPLITS,
    )
    cv_metadata = assert_cv_guardrails(tscv, expected_n_splits=N_CV_SPLITS)

    if training_manifest is not None:
        manifest_cv_hash = training_manifest["cv_metadata"]["cv_signature_hash"]
        if cv_metadata["cv_signature_hash"] != manifest_cv_hash:
            raise AssertionError(
                "Calibration CV split differs from training_manifest.json."
            )
    elif "cv_metadata" in train_payload:
        payload_cv = train_payload["cv_metadata"]
        if isinstance(payload_cv, Mapping):
            payload_cv_hash = payload_cv.get("cv_signature_hash")
            if payload_cv_hash and cv_metadata["cv_signature_hash"] != payload_cv_hash:
                raise AssertionError("Calibration CV split differs from train_data.joblib.")
    elif "cv_splits" in train_payload:
        legacy_cv_metadata = assert_cv_guardrails(
            train_payload["cv_splits"],
            expected_n_splits=N_CV_SPLITS,
        )
        if (
            cv_metadata["cv_signature_hash"]
            != legacy_cv_metadata["cv_signature_hash"]
        ):
            raise AssertionError("Calibration CV split differs from saved cv_splits.")
    elif train_payload.get("n_cv_splits") != N_CV_SPLITS:
        raise AssertionError(
            f"Expected {N_CV_SPLITS} calibration folds, got "
            f"{train_payload.get('n_cv_splits')!r} in train_data.joblib."
        )

    return tscv, cv_metadata


def _predict_proba_by_class(model: object, X: pd.DataFrame) -> np.ndarray:
    """Return predict_proba columns ordered as classes 0, 1, 2, 3."""
    probabilities = model.predict_proba(X)
    classes = getattr(model, "classes_", np.arange(probabilities.shape[1]))
    ordered = np.zeros((probabilities.shape[0], len(CLASS_NAMES)), dtype=float)

    for source_idx, class_value in enumerate(classes):
        class_idx = int(class_value)
        if 0 <= class_idx < len(CLASS_NAMES):
            ordered[:, class_idx] = probabilities[:, source_idx]

    return ordered


def expected_calibration_error(
    y_binary: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = ECE_N_BINS,
) -> float:
    """Compute expected calibration error for a one-vs-rest class."""
    y_arr = np.asarray(y_binary, dtype=float)
    p_arr = np.asarray(probabilities, dtype=float)
    if y_arr.shape[0] != p_arr.shape[0]:
        raise ValueError("y_binary and probabilities must have the same length.")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = float(len(p_arr))

    for bin_idx in range(n_bins):
        lower = bin_edges[bin_idx]
        upper = bin_edges[bin_idx + 1]
        if bin_idx == n_bins - 1:
            in_bin = (p_arr >= lower) & (p_arr <= upper)
        else:
            in_bin = (p_arr >= lower) & (p_arr < upper)

        if not np.any(in_bin):
            continue

        bin_weight = float(in_bin.sum()) / n_samples
        bin_accuracy = float(y_arr[in_bin].mean())
        bin_confidence = float(p_arr[in_bin].mean())
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_ece_per_class(
    y_true: pd.Series,
    probabilities: np.ndarray,
    n_bins: int = ECE_N_BINS,
) -> Dict[str, float]:
    """Compute one-vs-rest ECE for all alert classes."""
    y_values = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
    ece: Dict[str, float] = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        y_binary = (y_values == class_idx).astype(int)
        ece[class_name] = expected_calibration_error(
            y_binary,
            probabilities[:, class_idx],
            n_bins=n_bins,
        )
    return ece


def _write_calibration_metrics(
    metrics_path: Path,
    metrics_doc: Dict,
    ranking: Sequence[Mapping[str, object]],
    calibrated_records: Sequence[Mapping[str, object]],
    cv_metadata: Mapping[str, object],
) -> None:
    """Persist hybrid classifier calibration metadata into metrics JSON."""
    calibrated_models: Dict[str, Dict] = {}
    models_doc = metrics_doc.setdefault("models", {})

    for record in calibrated_records:
        model_name = str(record["model_name"])
        calibrated_models[model_name] = {
            "artifact_path": record["artifact_path"],
            "legacy_artifact_path": record.get("legacy_artifact_path"),
            "reliability_diagram": record["reliability_diagram"],
            "f1_macro_baseline": float(record["f1_macro_baseline"]),
            "expected_calibration_error": record["expected_calibration_error"],
            "ece_per_classe": record["ece_per_classe"],
        }

        model_metrics = models_doc.setdefault(model_name, {})
        model_metrics["strategy_name"] = f"classifier_{model_name}_calibrated"
        model_metrics["calibrated_artifact"] = record["artifact_path"]
        model_metrics["reliability_diagram"] = record["reliability_diagram"]
        model_metrics["ece_per_classe"] = record["ece_per_classe"]
        model_metrics["expected_calibration_error"] = record[
            "expected_calibration_error"
        ]

    selected_model = str(calibrated_records[0]["model_name"])
    metrics_doc["hybrid_support_classifier"] = {
        "selected_model": selected_model,
        "candidate_ranking": list(ranking),
        "calibrated_models": calibrated_models,
        "selection_rule": {
            "primary_metric": "f1_macro",
            "tie_f1_threshold": HYBRID_TIE_F1_THRESHOLD,
            "second_best_calibrated": len(calibrated_records) > 1,
        },
        "calibration": {
            "method": "isotonic",
            "cv": "make_temporal_cv_splits",
            "n_cv_splits": int(N_CV_SPLITS),
            "cv_signature_hash": cv_metadata["cv_signature_hash"],
            "ece_n_bins": int(ECE_N_BINS),
        },
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }

    write_json(metrics_path, metrics_doc)


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagrams(
    uncalibrated: Pipeline,
    calibrated: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: str,
    output_filename: str = "reliability_diagram_before_after.png",
    n_bins: int = 10,
) -> Path:
    """Plot one-vs-rest reliability diagrams before/after calibration.

    For each of the 4 alert classes, a subplot compares the calibration curve
    of the uncalibrated model against the calibrated model.  A perfectly
    calibrated classifier follows the diagonal.

    Args:
        uncalibrated: Fitted pipeline (best model, before calibration).
        calibrated: Fitted ``CalibratedClassifierCV`` wrapper.
        X_test: Test feature matrix.
        y_test: True integer labels (0-3).
        plots_dir: Directory to save the PNG.
        output_filename: Name of the PNG file to write.
        n_bins: Number of bins for the calibration curve.

    Returns:
        Path of the saved PNG file.
    """
    prob_uncal = _predict_proba_by_class(uncalibrated, X_test)
    prob_cal = _predict_proba_by_class(calibrated, X_test)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.ravel()

    for idx, (cls_name, ax) in enumerate(zip(CLASS_NAMES, axes_flat)):
        y_binary = (y_test.values == idx).astype(int)

        # Uncalibrated curve
        frac_pos_uncal, mean_pred_uncal = calibration_curve(
            y_binary, prob_uncal[:, idx], n_bins=n_bins, strategy="uniform",
        )
        ax.plot(
            mean_pred_uncal,
            frac_pos_uncal,
            marker="s",
            linewidth=1.5,
            label="Prima (non calibrato)",
            color="steelblue",
        )

        # Calibrated curve
        frac_pos_cal, mean_pred_cal = calibration_curve(
            y_binary, prob_cal[:, idx], n_bins=n_bins, strategy="uniform",
        )
        ax.plot(
            mean_pred_cal,
            frac_pos_cal,
            marker="o",
            linewidth=1.5,
            label="Dopo (calibrato)",
            color="darkorange",
        )

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfetta")

        ax.set_xlabel("Probabilita' media predetta")
        ax.set_ylabel("Frazione positivi osservata")
        ax.set_title(f"Classe: {cls_name}", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle(
        "Reliability Diagram — Prima vs Dopo Calibrazione (one-vs-rest)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Reliability diagram saved -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main calibration entry point
# ---------------------------------------------------------------------------

def run_calibration(artifacts_dir_path: str | None = None) -> Path:
    """Calibrate the Step 5 classifier selected for the hybrid strategy.

    Only the admitted Step 5 classifiers are considered. The best candidate by
    ``f1_macro`` is calibrated; the runner-up is calibrated too only if it is
    within ``HYBRID_TIE_F1_THRESHOLD``.
    """
    artifacts_dir = Path(artifacts_dir_path or ARTIFACTS_DIR)
    plots_dir = PLOTS_DIR
    metrics_path = Path(METRICS_FILE)

    training_manifest = _load_optional_training_manifest(artifacts_dir)
    if training_manifest is not None:
        assert_manifest_guardrails(
            training_manifest,
            artifacts_dir=artifacts_dir,
            model_names=CLASSIFIER_CANDIDATES + ("best_model",),
            expected_n_splits=N_CV_SPLITS,
            expected_random_state=RANDOM_STATE,
            verify_files=True,
        )

    metrics_doc = _load_metrics_doc(metrics_path)
    ranking = rank_hybrid_classifier_candidates(metrics_doc)
    selected_candidates = select_hybrid_calibration_candidates(ranking)
    _assert_selected_artifacts_exist(artifacts_dir, selected_candidates)

    logger.info("Hybrid classifier ranking: %s", ranking)
    logger.info(
        "Selected for calibration: %s",
        [candidate["model_name"] for candidate in selected_candidates],
    )

    X_train, y_train, train_dates, train_payload = _load_train_data(artifacts_dir)
    X_test, y_test, test_payload = _load_test_data(artifacts_dir)
    if training_manifest is None:
        _validate_legacy_split_metadata(train_dates, test_payload, metrics_doc)

    tscv, cv_metadata = _build_and_validate_calibration_cv(
        X_train=X_train,
        y_train=y_train,
        train_dates=train_dates,
        train_payload=train_payload,
        training_manifest=training_manifest,
    )
    logger.info("Built %d temporal CV splits for calibration", len(tscv))

    calibrated_records: List[Dict] = []
    primary_calibrated_path: Optional[Path] = None

    for candidate in selected_candidates:
        model_name = str(candidate["model_name"])
        model_path = artifacts_dir / f"{model_name}_best.joblib"
        uncalibrated_model: Pipeline = joblib.load(model_path)
        logger.info("Loaded hybrid classifier candidate '%s' from %s", model_name, model_path)

        logger.info(
            "Fitting %s with CalibratedClassifierCV(method='isotonic') ...",
            model_name,
        )
        calibrated_model = CalibratedClassifierCV(
            estimator=uncalibrated_model,
            method="isotonic",
            cv=tscv,
        )
        calibrated_model.fit(X_train, y_train)
        logger.info("Calibration fitting complete for %s", model_name)

        cal_path = artifacts_dir / f"{model_name}_hybrid_calibrated.joblib"
        joblib.dump(calibrated_model, cal_path)
        logger.info("Saved hybrid calibrated classifier -> %s", cal_path)

        legacy_path: Optional[Path] = None
        metrics_best_model = metrics_doc.get("best_model") or metrics_doc.get(
            "best_evaluated_model"
        )
        manifest_best_model = (
            training_manifest.get("best_model")
            if isinstance(training_manifest, Mapping)
            else None
        )
        if model_name in {metrics_best_model, manifest_best_model}:
            legacy_path = artifacts_dir / "best_model_calibrated.joblib"
            joblib.dump(calibrated_model, legacy_path)
            logger.info("Updated legacy calibrated artifact -> %s", legacy_path)

        plot_path = plot_reliability_diagrams(
            uncalibrated=uncalibrated_model,
            calibrated=calibrated_model,
            X_test=X_test,
            y_test=y_test,
            plots_dir=plots_dir,
            output_filename=f"{model_name}_hybrid_reliability_diagram_before_after.png",
            n_bins=ECE_N_BINS,
        )

        prob_uncalibrated = _predict_proba_by_class(uncalibrated_model, X_test)
        prob_calibrated = _predict_proba_by_class(calibrated_model, X_test)
        ece_before = compute_ece_per_class(
            y_test,
            prob_uncalibrated,
            n_bins=ECE_N_BINS,
        )
        ece_after = compute_ece_per_class(
            y_test,
            prob_calibrated,
            n_bins=ECE_N_BINS,
        )

        calibrated_records.append(
            {
                "model_name": model_name,
                "artifact_path": cal_path.as_posix(),
                "legacy_artifact_path": legacy_path.as_posix() if legacy_path else None,
                "reliability_diagram": plot_path.as_posix(),
                "f1_macro_baseline": float(candidate["f1_macro"]),
                "ece_per_classe": ece_after,
                "expected_calibration_error": {
                    "n_bins": int(ECE_N_BINS),
                    "per_class": {
                        class_name: {
                            "before": ece_before[class_name],
                            "after": ece_after[class_name],
                        }
                        for class_name in CLASS_NAMES
                    },
                },
            }
        )

        if primary_calibrated_path is None:
            primary_calibrated_path = cal_path

    _write_calibration_metrics(
        metrics_path=metrics_path,
        metrics_doc=metrics_doc,
        ranking=ranking,
        calibrated_records=calibrated_records,
        cv_metadata=cv_metadata,
    )
    logger.info("Calibration metrics saved -> %s", metrics_path)

    if primary_calibrated_path is None:
        raise RuntimeError("No hybrid classifier was calibrated.")
    return primary_calibrated_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run_calibration()
