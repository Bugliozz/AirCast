"""Probability calibration for Step 5 — Classification (step 5.5).

Fits ``CalibratedClassifierCV`` on the best model using temporal CV splits
(isotonic regression), then generates reliability diagrams (one-vs-rest
calibration curves) comparing uncalibrated vs. calibrated probabilities for
each alert class.

Usage (from project root)::

    python -m step_5_classification.calibrate
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

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
    N_CV_SPLITS,
    PLOTS_DIR,
)

logger = logging.getLogger(__name__)

CLASS_NAMES: List[str] = ["verde", "giallo", "arancio", "rosso"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_train_data(
    artifacts_dir: Path,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load training data persisted by train.py.

    Returns:
        Tuple ``(X_train, y_train, train_dates)``.
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
    return X_train, y_train, train_dates


def _load_test_data(
    artifacts_dir: Path,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load test data persisted by train.py."""
    test_path = artifacts_dir / "test_data.joblib"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at '{test_path}'. "
            "Run 'python -m step_5_classification.train' first."
        )

    payload = joblib.load(test_path)
    return payload["X_test"], payload["y_test"].astype(int)


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagrams(
    uncalibrated: Pipeline,
    calibrated: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: str,
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
        n_bins: Number of bins for the calibration curve.

    Returns:
        Path of the saved PNG file.
    """
    prob_uncal = uncalibrated.predict_proba(X_test)
    prob_cal = calibrated.predict_proba(X_test)

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
    out_path = out_dir / "reliability_diagram_before_after.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Reliability diagram saved -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main calibration entry point
# ---------------------------------------------------------------------------

def run_calibration(artifacts_dir_path: str | None = None) -> Path:
    """Calibrate the best classification model and save it.

    Steps:
    1. Load best_model.joblib (uncalibrated) and training data.
    2. Rebuild temporal CV splits from the training dates.
    3. Fit ``CalibratedClassifierCV(method="isotonic", cv=tscv)`` on the
       training set (never on the test set).
    4. Save ``best_model_calibrated.joblib``.
    5. Plot reliability diagrams (before/after) on the test set.

    Args:
        artifacts_dir_path: Override for artifacts directory.

    Returns:
        Path to the saved calibrated model.
    """
    artifacts_dir = Path(artifacts_dir_path or ARTIFACTS_DIR)
    plots_dir = PLOTS_DIR

    # 1. Load best model
    best_model_path = artifacts_dir / "best_model.joblib"
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at '{best_model_path}'. "
            "Run 'python -m step_5_classification.train' first."
        )
    best_model: Pipeline = joblib.load(best_model_path)
    logger.info("Loaded best model from '%s'", best_model_path)

    # 2. Load train and test data
    X_train, y_train, train_dates = _load_train_data(artifacts_dir)
    X_test, y_test = _load_test_data(artifacts_dir)

    # 3. Build temporal CV splits (same logic as train.py)
    X_train_with_dates = X_train.copy()
    X_train_with_dates.insert(0, "data_giorno", train_dates)
    tscv = make_temporal_cv_splits(
        X_train_with_dates, y_train, n_splits=N_CV_SPLITS,
    )
    logger.info("Built %d temporal CV splits for calibration", len(tscv))

    # 4. Fit calibrated model (isotonic regression)
    logger.info("Fitting CalibratedClassifierCV(method='isotonic') ...")
    calibrated_model = CalibratedClassifierCV(
        estimator=best_model,
        method="isotonic",
        cv=tscv,
    )
    calibrated_model.fit(X_train, y_train)
    logger.info("Calibration fitting complete")

    # 5. Save calibrated model
    cal_path = artifacts_dir / "best_model_calibrated.joblib"
    joblib.dump(calibrated_model, cal_path)
    logger.info("Saved calibrated model -> %s", cal_path)

    # 6. Reliability diagrams (before vs after) on test set
    plot_reliability_diagrams(
        uncalibrated=best_model,
        calibrated=calibrated_model,
        X_test=X_test,
        y_test=y_test,
        plots_dir=plots_dir,
    )

    return cal_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run_calibration()
