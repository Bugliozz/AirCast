"""Evaluation pipeline for Step 5 — Classification.

Loads trained model artifacts and evaluates them on the holdout test set
that was persisted by ``train.py`` (``artifacts/test_data.joblib``).

Tasks covered:
- Predict with each trained model on X_test
- ``classification_report`` (f1_macro, per-class precision/recall/f1)
- ``severe_error_rate``: fraction of predictions where |y_pred − y_true| ≥ 2
- Confusion matrix heatmap (seaborn.heatmap with absolute counts) per model
- Permutation importance plot for the best model on the test set
- Save ``classification_metrics.json``

Usage (from project root)::

    python -m step_5_classification.evaluate
"""

import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from step_5_classification.config import (
    ARTIFACTS_DIR,
    LABEL_MAP,
    METRICS_FILE,
    PLOTS_DIR,
)

logger = logging.getLogger(__name__)

# Alert class names in severity order (index 0–3 corresponds to LABEL_MAP values)
CLASS_NAMES: List[str] = ["verde", "giallo", "arancio", "rosso"]

# Model artifact names — must match filenames written by train.py
MODEL_NAMES: List[str] = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_test_data(artifacts_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the test set persisted by train.py.

    ``train.py`` saves ``test_data.joblib`` containing a dict
    ``{"X_test": pd.DataFrame, "y_test": pd.Series}`` with integer labels
    (0 = verde, 1 = giallo, 2 = arancio, 3 = rosso).

    Args:
        artifacts_dir: Path to the ``artifacts/`` directory.

    Returns:
        Tuple ``(X_test, y_test)`` — feature matrix and integer target Series.

    Raises:
        FileNotFoundError: if ``test_data.joblib`` is missing (run train.py first).
    """
    test_path = artifacts_dir / "test_data.joblib"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data artifact not found at '{test_path}'. "
            "Run 'python -m step_5_classification.train' first."
        )

    payload = joblib.load(test_path)
    X_test: pd.DataFrame = payload["X_test"]
    y_test: pd.Series = payload["y_test"].astype(int)

    logger.info(
        "Loaded test data from '%s': %d samples, %d features",
        test_path, len(X_test), X_test.shape[1],
    )
    logger.info(
        "Label distribution (test): %s",
        y_test.value_counts().sort_index().to_dict(),
    )

    return X_test, y_test


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> Dict:
    """Compute classification metrics for a single model.

    Metrics:
    - ``f1_macro``: macro-averaged F1 across all 4 alert classes.
    - ``per_class``: dict of per-class precision, recall, f1-score, support
      (keyed by class name: verde/giallo/arancio/rosso).
    - ``severe_error_rate``: fraction of samples where predicted class is ≥ 2
      severity levels away from the true class — a domain-relevant safety metric.

    Args:
        y_true: True integer labels (0–3), test set.
        y_pred: Predicted integer labels (0–3).
        model_name: Used only for logging.

    Returns:
        Dict with keys ``f1_macro``, ``per_class``, ``severe_error_rate``.
    """
    report: Dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    f1_macro = float(report["macro avg"]["f1-score"])

    per_class: Dict[str, Dict] = {}
    for cls in CLASS_NAMES:
        per_class[cls] = {
            "precision": float(report[cls]["precision"]),
            "recall": float(report[cls]["recall"]),
            "f1": float(report[cls]["f1-score"]),
            "support": int(report[cls]["support"]),
        }

    severe_error_rate = float((np.abs(y_pred - y_true.values) >= 2).mean())

    logger.info(
        "%-22s  f1_macro=%.4f  severe_error_rate=%.4f",
        model_name, f1_macro, severe_error_rate,
    )
    for cls in CLASS_NAMES:
        pc = per_class[cls]
        logger.info(
            "  %-10s  P=%.3f  R=%.3f  F1=%.3f  (n=%d)",
            cls, pc["precision"], pc["recall"], pc["f1"], pc["support"],
        )

    return {
        "f1_macro": f1_macro,
        "per_class": per_class,
        "severe_error_rate": severe_error_rate,
    }


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    plots_dir: str,
) -> Path:
    """Plot and save an annotated confusion matrix heatmap.

    Uses ``seaborn.heatmap`` with absolute counts as annotations.
    Rows = true class; columns = predicted class.  Classes are shown in
    severity order: verde → giallo → arancio → rosso.

    Args:
        y_true: True integer labels (0–3).
        y_pred: Predicted integer labels (0–3).
        model_name: Used for the figure title and output filename.
        plots_dir: Directory to save the PNG (created if absent).

    Returns:
        :class:`pathlib.Path` of the saved PNG file.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        linecolor="lightgray",
        ax=ax,
    )
    ax.set_xlabel("Classe Predetta", fontsize=11)
    ax.set_ylabel("Classe Reale", fontsize=11)
    ax.set_title(
        f"Matrice di Confusione — {model_name}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Confusion matrix saved -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Feature importance — helpers
# ---------------------------------------------------------------------------

def _get_feature_names(pipeline: Pipeline) -> List[str]:
    """Extract output feature names from the fitted ColumnTransformer step.

    Relies on ``verbose_feature_names_out=False`` set in
    :func:`shared.utils.build_preprocessor`, which preserves original column
    names without transformer prefixes.

    Args:
        pipeline: A *fitted* sklearn Pipeline whose first step is named ``"pre"``
            and is a :class:`~sklearn.compose.ColumnTransformer`.

    Returns:
        List of feature name strings.  Falls back to ``["feature_0", ...]``
        if ``get_feature_names_out`` is unavailable.
    """
    try:
        return list(pipeline.named_steps["pre"].get_feature_names_out())
    except AttributeError:
        logger.warning(
            "_get_feature_names: get_feature_names_out unavailable — "
            "using generic names."
        )
        n = pipeline.named_steps["pre"].n_features_in_
        return [f"feature_{i}" for i in range(n)]


def plot_permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    plots_dir: str,
    top_n: int = 15,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Path:
    """Compute and plot permutation feature importance on the test set.

    Scoring: ``f1_macro`` — consistent with the CV tuning metric.
    Importance = decrease in f1_macro when the feature is shuffled.
    Larger values → feature is more important for generalisation.

    Args:
        pipeline: Fitted sklearn Pipeline (best model).
        X_test: Test-set feature matrix.
        y_test: True integer labels (0–3) for the test set.
        model_name: Used for the figure title and output filename.
        plots_dir: Directory to save the PNG (created if absent).
        top_n: Number of top features to display (default 15).
        n_repeats: Permutation repetitions per feature (default 10).
        random_state: Seed for reproducibility.

    Returns:
        :class:`pathlib.Path` of the saved PNG file.
    """
    logger.info(
        "Permutation importance for '%s' on test set "
        "(n_repeats=%d, top_%d, scoring=f1_macro) ...",
        model_name, n_repeats, top_n,
    )

    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="f1_macro",
        n_jobs=-1,
    )

    feature_names = _get_feature_names(pipeline)
    means = perm.importances_mean
    stds = perm.importances_std

    # Sort descending by mean importance, keep top_n
    sorted_idx = np.argsort(means)[::-1][:top_n]
    top_names = [
        feature_names[i] if i < len(feature_names) else f"feature_{i}"
        for i in sorted_idx
    ]
    top_means = means[sorted_idx]
    top_stds = stds[sorted_idx]

    # Horizontal bar chart — largest importance at the TOP
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_names))
    ax.barh(
        y_pos,
        top_means[::-1],
        xerr=top_stds[::-1],
        align="center",
        color="steelblue",
        ecolor="gray",
        capsize=3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Diminuzione media di F1-macro quando la feature è permutata")
    ax.set_title(
        f"Permutation Importance — {model_name}  (top {top_n}, test set)",
        fontweight="bold",
    )
    ax.axvline(0.0, color="red", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_permutation_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        "Permutation importance saved -> %s  |  Top 3: %s",
        out_path,
        ", ".join(
            f"{top_names[i]} ({top_means[i]:.4f})"
            for i in range(min(3, len(top_names)))
        ),
    )
    return out_path


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary_table(results: Dict[str, Dict]) -> None:
    """Log a formatted summary table of all model metrics."""
    header = f"{'Model':<25}  {'F1-macro':>10}  {'SevereErrRate':>15}"
    sep = "-" * len(header)

    logger.info("=" * len(header))
    logger.info("CLASSIFICATION TEST-HOLDOUT EVALUATION")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info(sep)

    for name, m in results.items():
        logger.info(
            "%-25s  %10.4f  %15.4f",
            name, m["f1_macro"], m["severe_error_rate"],
        )

    logger.info("=" * len(header))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_evaluation(artifacts_dir_path: Optional[str] = None) -> Dict[str, Dict]:
    """Evaluate trained classification models on the holdout test set.

    Data flow::

        test_data.joblib  (X_test, y_test — saved by train.py)
          -> load each model pipeline from .joblib
          -> model.predict(X_test)  → y_pred (int 0–3)
          -> classification_report  → f1_macro + per_class metrics
          -> severe_error_rate      → fraction |y_pred - y_true| >= 2
          -> confusion matrix plot  → artifacts/plots/
        best_model.joblib
          -> permutation_importance → artifacts/plots/
        -> classification_metrics.json

    Args:
        artifacts_dir_path: Optional override for ``ARTIFACTS_DIR``.
            Defaults to ``ARTIFACTS_DIR`` from config.

    Returns:
        Dict mapping model name → metrics dict.
        Empty dict if no artifact is found.
    """
    artifacts_dir = Path(artifacts_dir_path or ARTIFACTS_DIR)
    plots_dir = PLOTS_DIR

    # 1. Load test data (persisted by train.py to avoid re-running the split)
    X_test, y_test = _load_test_data(artifacts_dir)

    results: Dict[str, Dict] = {}
    pipelines: Dict[str, Pipeline] = {}

    # 2. Evaluate each model
    for model_name in MODEL_NAMES:
        model_path = artifacts_dir / f"{model_name}_best.joblib"
        if not model_path.exists():
            logger.warning(
                "Artifact not found: '%s' — skipping (run train.py first).",
                model_path,
            )
            continue

        pipeline: Pipeline = joblib.load(model_path)
        pipelines[model_name] = pipeline
        logger.info("Loaded '%s' from %s", model_name, model_path)

        y_pred: np.ndarray = pipeline.predict(X_test)

        # --- Metrics ---
        logger.info("--- %s ---", model_name.upper())
        metrics = compute_classification_metrics(y_test, y_pred, model_name)
        results[model_name] = metrics

        # --- Confusion matrix plot ---
        plot_confusion_matrix(y_test, y_pred, model_name, plots_dir)

    if not results:
        logger.warning(
            "No model artifacts found in '%s'. Run train.py first.", artifacts_dir
        )
        return results

    # 3. Summary table
    _print_summary_table(results)

    # 4. Identify best model by test-set f1_macro
    best_model_name = max(results, key=lambda m: results[m]["f1_macro"])
    logger.info(
        "Best model by test f1_macro: '%s' (f1_macro=%.4f)",
        best_model_name, results[best_model_name]["f1_macro"],
    )

    # 5. Permutation importance for the best model
    # Load from best_model.joblib (identical pipeline saved by train.py)
    best_model_path = artifacts_dir / "best_model.joblib"
    if best_model_path.exists():
        best_pipeline: Pipeline = joblib.load(best_model_path)
        plot_permutation_importance(
            best_pipeline,
            X_test,
            y_test,
            best_model_name,
            plots_dir,
        )
    else:
        logger.warning(
            "best_model.joblib not found at '%s' — skipping permutation importance.",
            best_model_path,
        )

    # 6. Save classification_metrics.json
    metrics_path = Path(METRICS_FILE)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    models_payload: Dict = {}
    for name, m in results.items():
        models_payload[name] = {
            "f1_macro": m["f1_macro"],
            "per_class": m["per_class"],
            "severe_error_rate": m["severe_error_rate"],
        }

    metrics_doc: Dict = {
        "best_model": best_model_name,
        "generated_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_test_samples": int(len(y_test)),
        "models": models_payload,
    }

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_doc, fh, indent=2)

    logger.info("Metrics saved -> %s", metrics_path)
    logger.info(
        "Plots directory: %s  (%d PNG files)",
        plots_dir,
        len(list(Path(plots_dir).glob("*.png"))),
    )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run_evaluation()
