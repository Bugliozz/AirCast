"""Evaluation pipeline for Step 4 — Regression.

Loads trained model artifacts and evaluates them on the holdout test set.

Task 3.5.1 — Metrics on holdout test:
- RMSE (primary metric)
- MAE (robustness to outliers)
- R² (explained variance)
- RMSE stratified by alert class: verde / giallo / arancio / rosso
- Summary table of all models

Task 3.5.2 — Diagnostic plots:
- Scatter y_pred vs y_true (with identity line)
- Residuals vs y_pred (homoscedasticity)
- Residuals vs time (no trend)
- Residual distribution (normal, centred on 0)

Tasks 3.5.3–3.5.4 (feature importance, salvataggio artifacts)
are implemented in the corresponding sub-tasks.

Usage (from project root)::

    python -m step_4_regression.evaluate
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from shared.utils import impute_missing, temporal_train_test_split
from step_3_eda.config import PM10_LABELS, PM10_THRESHOLDS
from step_4_regression.config import (
    ARTIFACTS_DIR,
    DROP_COLS,
    METRICS_FILE,
    PLOTS_DIR,
    TARGET_COL,
)

logger = logging.getLogger(__name__)

# Alert class labels in severity order
ALERT_CLASSES: List[str] = list(PM10_LABELS)
ALERT_THRESHOLDS: List[float] = [float(v) for v in PM10_THRESHOLDS[1:-1]]
NEAR_THRESHOLD_DELTA: float = 5.0

# Model artifact names — must match filenames written by train.py
MODEL_NAMES: List[str] = ["elasticnet", "xgboost", "random_forest"]

# Train/test split ratio — must match the value used in train.py
_TRAIN_RATIO: float = 0.70


# ---------------------------------------------------------------------------
# 3.5.1 — Helper: reproduce test split
# ---------------------------------------------------------------------------

def _load_test_split(
    parquet_path: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, Dict]:
    """Reproduce the temporal train/test split used during training.

    ``classe_allerta`` is derived from ``pm10`` (the regression target) and
    therefore constitutes direct label leakage if used as a model feature.
    It is listed in ``DROP_COLS`` to exclude it from the feature matrix, but
    it is extracted here *as metadata only* to enable the per-class RMSE
    breakdown required by task 3.5.1.

    ``data_giorno`` is also extracted as metadata for the residuals-vs-time
    plot required by task 3.5.2.

    Args:
        parquet_path: Path to ``daily_dataset_clean.parquet``.

    Returns:
        Tuple ``(X_test, y_test, alert_class_test, dates_test, split_metadata)``
        where:
        - ``X_test``: imputed feature matrix for the test period.
        - ``y_test``: true PM10 values (original scale, ug/m3).
        - ``alert_class_test``: alert-class labels aligned row-for-row with
          ``y_test``, used only to stratify per-class metrics.
        - ``dates_test``: date Series (``data_giorno``) aligned row-for-row
          with ``y_test``, used only for the temporal residuals plot.
    """
    df = pd.read_parquet(parquet_path)
    logger.info(
        "Loaded dataset: %d rows, %d columns from '%s'",
        len(df), df.shape[1], parquet_path,
    )

    if "classe_allerta" not in df.columns:
        raise ValueError(
            "Column 'classe_allerta' not found in dataset. "
            "Re-run build_dataset.py to regenerate the parquet."
        )

    # --- Reproduce the EXACT same temporal cutoff as train.py ----------------
    # temporal_train_test_split sorts by data_giorno and takes the first
    # int(n_days * train_ratio) days as training; the rest are the test set.
    df_sorted = df.sort_values("data_giorno").reset_index(drop=True)
    unique_days = (
        df_sorted["data_giorno"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    n_train_days = int(len(unique_days) * _TRAIN_RATIO)
    cutoff_day = unique_days.iloc[n_train_days - 1]

    # Extract classe_allerta and data_giorno for the test portion as metadata
    # (neither is passed to the model — both are used only for diagnostics)
    test_mask = df_sorted["data_giorno"] > cutoff_day
    train_mask = ~test_mask
    train_df = df_sorted.loc[train_mask].reset_index(drop=True)
    test_df = df_sorted.loc[test_mask].reset_index(drop=True)
    valid_train_mask = (
        train_df["classe_allerta"].notna()
        & train_df[TARGET_COL].notna()
    ).reset_index(drop=True)
    valid_test_mask = (
        test_df["classe_allerta"].notna()
        & test_df[TARGET_COL].notna()
    ).reset_index(drop=True)
    train_df_valid = train_df.loc[valid_train_mask].reset_index(drop=True)
    test_df_valid = test_df.loc[valid_test_mask].reset_index(drop=True)

    n_dropped_test = int((~valid_test_mask).sum())
    if n_dropped_test:
        logger.warning(
            "Dropped %d test rows with invalid target/alert class before evaluation.",
            n_dropped_test,
        )

    alert_class_test = (
        test_df_valid["classe_allerta"]
        .reset_index(drop=True)
        .astype(str)  # ensure plain str, not pd.Categorical, for == comparisons
    )
    dates_test = test_df_valid["data_giorno"].reset_index(drop=True)
    train_dates = train_df_valid["data_giorno"].reset_index(drop=True)

    train_stations = train_df_valid["idstazione"].astype(str)
    test_stations = test_df_valid["idstazione"].astype(str)
    station_intersection = set(train_stations) & set(test_stations)
    split_metadata = {
        "train_date_range": [
            str(pd.to_datetime(train_dates, errors="raise").min()),
            str(pd.to_datetime(train_dates, errors="raise").max()),
        ],
        "test_date_range": [
            str(pd.to_datetime(dates_test, errors="raise").min()),
            str(pd.to_datetime(dates_test, errors="raise").max()),
        ],
        "n_train_samples": int(len(train_df_valid)),
        "n_test_samples": int(len(test_df_valid)),
        "n_stations_train": int(train_stations.nunique()),
        "n_stations_test": int(test_stations.nunique()),
        "n_stations_intersection": int(len(station_intersection)),
    }

    logger.info(
        "Alert-class distribution in test set: %s",
        alert_class_test.value_counts().to_dict(),
    )

    # --- Reproduce train/test feature matrices -------------------------------
    X_train, y_train, X_test, y_test, _ = temporal_train_test_split(
        df, target_col=TARGET_COL, drop_cols=DROP_COLS
    )

    # Imputation: fit medians on train only, then apply the same medians to test
    X_train, train_medians = impute_missing(X_train)
    X_test, _ = impute_missing(X_test, medians=train_medians)
    X_test = X_test.loc[valid_test_mask].reset_index(drop=True)
    y_test = y_test.loc[valid_test_mask].reset_index(drop=True)

    return X_test, y_test, alert_class_test, dates_test, split_metadata


# ---------------------------------------------------------------------------
# 3.5.1 — Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    alert_class: pd.Series,
    near_threshold_delta: float = NEAR_THRESHOLD_DELTA,
) -> Dict:
    """Compute RMSE, MAE, R² overall and RMSE broken down by alert class.

    Args:
        y_true: True PM10 values (test set, original ug/m3 scale).
        y_pred: Predicted PM10 values (same scale).
        alert_class: Alert-class label Series aligned row-for-row with
            ``y_true``; values in {``'verde'``, ``'giallo'``, ``'arancio'``,
            ``'rosso'``}.

    Returns:
        Dict with keys:
        - ``rmse`` (float): overall RMSE in ug/m3.
        - ``mae``  (float): overall MAE  in ug/m3.
        - ``r2``   (float): overall R².
        - ``rmse_per_class`` (dict[str, float]): RMSE for each alert class;
          ``NaN`` when a class is absent from the test set.
        - ``n_per_class`` (dict[str, int]): sample count per class.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    rmse_per_fascia: Dict[str, float] = {}
    mae_per_fascia: Dict[str, float] = {}
    n_per_fascia: Dict[str, int] = {}

    for cls in ALERT_CLASSES:
        mask = alert_class == cls
        n = int(mask.sum())
        n_per_fascia[cls] = n

        if n == 0:
            logger.warning(
                "Alert class '%s' absent from test set — RMSE set to NaN.", cls
            )
            rmse_per_fascia[cls] = float("nan")
            mae_per_fascia[cls] = float("nan")
            continue

        cls_rmse = float(
            np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        )
        cls_mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
        rmse_per_fascia[cls] = cls_rmse
        mae_per_fascia[cls] = cls_mae
        logger.info(
            "  %-8s: RMSE=%7.4f | MAE=%7.4f ug/m3  (n=%d)",
            cls, cls_rmse, cls_mae, n,
        )

    y_true_arr = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    rmse_near_thresholds: Dict[str, float] = {}
    n_near_thresholds: Dict[str, int] = {}
    for threshold in ALERT_THRESHOLDS:
        key = f"{threshold:.1f}"
        mask = (
            (y_true_arr >= threshold - near_threshold_delta)
            & (y_true_arr <= threshold + near_threshold_delta)
        )
        n = int(mask.sum())
        n_near_thresholds[key] = n
        if n == 0:
            rmse_near_thresholds[key] = float("nan")
            continue
        rmse_near_thresholds[key] = float(
            np.sqrt(mean_squared_error(y_true_arr[mask], y_pred_arr[mask]))
        )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmse_per_fascia": rmse_per_fascia,
        "mae_per_fascia": mae_per_fascia,
        "n_per_fascia": n_per_fascia,
        "rmse_near_thresholds": rmse_near_thresholds,
        "n_near_thresholds": n_near_thresholds,
        "near_threshold_delta": float(near_threshold_delta),
        "rmse_per_class": rmse_per_fascia,
        "mae_per_class": mae_per_fascia,
        "n_per_class": n_per_fascia,
    }


# ---------------------------------------------------------------------------
# 3.5.1 — Summary table printer
# ---------------------------------------------------------------------------

def _print_summary_table(results: Dict[str, Dict]) -> None:
    """Log a formatted summary table of all model metrics.

    Columns: Model | RMSE | MAE | R² | RMSE-verde | RMSE-giallo | RMSE-arancio | RMSE-rosso
    """
    col_headers = ["RMSE", "MAE", "R²"] + [f"RMSE-{c}" for c in ALERT_CLASSES]
    header_str = f"{'Model':<22}" + "".join(f"{h:>13}" for h in col_headers)
    sep = "-" * len(header_str)

    logger.info("=" * len(header_str))
    logger.info("REGRESSION TEST-HOLDOUT EVALUATION (task 3.5.1)")
    logger.info("=" * len(header_str))
    logger.info(header_str)
    logger.info(sep)

    for model_name, metrics in results.items():
        overall = [metrics["rmse"], metrics["mae"], metrics["r2"]]
        per_class = [metrics["rmse_per_class"].get(c, float("nan")) for c in ALERT_CLASSES]
        values = overall + per_class
        row = f"{model_name:<22}" + "".join(f"{v:>13.4f}" for v in values)
        logger.info(row)

    logger.info("=" * len(header_str))


# ---------------------------------------------------------------------------
# 3.5.2 — Diagnostic plots
# ---------------------------------------------------------------------------

def plot_diagnostics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    dates: pd.Series,
    model_name: str,
    plots_dir: str,
) -> Path:
    """Generate 4 diagnostic plots for a regression model and save to PNG.

    Plots (2×2 grid):
    1. **Predicted vs Actual** — scatter y_pred vs y_true with identity line.
       Ideal: points on the ``y = x`` diagonal.
    2. **Residuals vs Predicted** — residuals ``y_true − y_pred`` against y_pred.
       Ideal: random band around zero (homoscedasticity).
    3. **Residuals Over Time** — residuals against ``data_giorno``.
       Ideal: no systematic trend or seasonality.
    4. **Residual Distribution** — histogram + fitted normal PDF.
       Ideal: roughly Gaussian, centred at zero.

    Args:
        y_true: True PM10 values (test set, ug/m3), aligned with ``y_pred``.
        y_pred: Predicted PM10 values (same scale and length as ``y_true``).
        dates: ``data_giorno`` Series aligned row-for-row with ``y_true``.
        model_name: Model identifier used for the figure title and filename.
        plots_dir: Directory where the PNG is saved (created if absent).

    Returns:
        :class:`pathlib.Path` of the saved PNG file.
    """
    residuals = y_true.values - y_pred
    dates_dt = pd.to_datetime(dates)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Diagnostic Plots - {model_name}",
        fontsize=14,
        fontweight="bold",
    )

    # ------------------------------------------------------------------
    # 1. Scatter: y_pred vs y_true with identity line
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    all_vals = np.concatenate([y_true.values, y_pred])
    lim_low = float(all_vals.min()) - 5.0
    lim_high = float(all_vals.max()) + 5.0
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue", label="predictions")
    ax.plot([lim_low, lim_high], [lim_low, lim_high], "r--", linewidth=1.5, label="identity (y=x)")
    ax.set_xlim(lim_low, lim_high)
    ax.set_ylim(lim_low, lim_high)
    ax.set_xlabel("y_true (PM10 µg/m³)")
    ax.set_ylabel("y_pred (PM10 µg/m³)")
    ax.set_title("Predicted vs Actual")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # 2. Residuals vs y_pred (homoscedasticity)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.3, s=10, color="darkorange")
    ax.axhline(0.0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel("y_pred (PM10 µg/m³)")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.set_title("Residuals vs Predicted (homoscedasticity)")
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # 3. Residuals vs time (no trend)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    ax.scatter(dates_dt, residuals, alpha=0.3, s=8, color="green")
    ax.axhline(0.0, color="red", linewidth=1.5, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.set_title("Residuals Over Time (no trend)")
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # 4. Distribution of residuals with normal PDF overlay
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    ax.hist(
        residuals,
        bins=50,
        density=True,
        alpha=0.65,
        color="steelblue",
        edgecolor="white",
        label="empirical distribution",
    )
    mu = float(np.mean(residuals))
    sigma = float(np.std(residuals))
    x_range = np.linspace(float(residuals.min()), float(residuals.max()), 300)
    normal_pdf = (
        np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
        / (sigma * np.sqrt(2.0 * np.pi))
    )
    ax.plot(
        x_range,
        normal_pdf,
        "r-",
        linewidth=2,
        label=f"N(µ={mu:.1f}, σ={sigma:.1f})",
    )
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle=":", label="zero")
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_diagnostics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved diagnostic plots for '%s' -> %s", model_name, out_path)
    return out_path


# ---------------------------------------------------------------------------
# 3.5.3 — Feature importance helpers
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
        List of feature name strings in the same column order as the
        transformer output.  Falls back to ``["feature_0", ...]`` if
        ``get_feature_names_out`` is unavailable.
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
    """Compute and plot permutation importance for the best model on the test set.

    Permutation importance measures the *increase in RMSE* when a single
    feature's values are randomly shuffled.  Larger values → feature matters
    more.  Being computed on the *test* set (not training), it reflects
    generalisation performance and is free of overfitting bias.

    Scoring uses ``neg_root_mean_squared_error`` (same metric as CV), so
    importance values are expressed in ug/m³ RMSE units.

    Args:
        pipeline: Fitted sklearn Pipeline (best model, any estimator type).
        X_test: Test-set feature matrix (pre-imputed, raw columns — the
            pipeline's internal preprocessor handles the rest).
        y_test: Test-set target values (PM10 in ug/m³, original scale).
        model_name: Model identifier used for the plot title and filename.
        plots_dir: Directory where the PNG is saved (created if absent).
        top_n: Number of top features to display (default 15).
        n_repeats: Permutation repetitions per feature (default 10).
        random_state: Seed for reproducibility.

    Returns:
        :class:`pathlib.Path` of the saved PNG file.
    """
    logger.info(
        "3.5.3 — Permutation importance for '%s' on test set "
        "(n_repeats=%d, top_%d) ...",
        model_name, n_repeats, top_n,
    )

    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    feature_names = _get_feature_names(pipeline)
    means = perm.importances_mean   # shape (n_features,) — RMSE delta
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
    # Reverse so that index 0 (highest) appears at the top of the chart
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
    ax.set_xlabel("Mean RMSE increase when the feature is permuted (µg/m³)")
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


def plot_native_feature_importance(
    pipeline: Pipeline,
    model_name: str,
    plots_dir: str,
    top_n: int = 15,
) -> Optional[Path]:
    """Plot native feature importances for tree-based or linear models.

    Used as a **sanity check** for non-best models (task 3.5.3):

    - **XGBoost / RandomForest**: ``.feature_importances_`` (gain-based /
      mean decrease in impurity — fast but biased towards high-cardinality
      features; useful for comparison with permutation results).
    - **ElasticNet**: ``abs(coef_)`` from the underlying linear model, valid
      because features are StandardScaled inside the pipeline so coefficients
      are comparable across features.

    Args:
        pipeline: Fitted sklearn Pipeline.
        model_name: Model identifier used for the plot title and filename.
        plots_dir: Directory where the PNG is saved (created if absent).
        top_n: Number of top features to display (default 15).

    Returns:
        :class:`pathlib.Path` of the saved PNG, or ``None`` if importances
        cannot be extracted.
    """
    feature_names = _get_feature_names(pipeline)
    regressor = pipeline.named_steps["regressor"]

    # Determine importance values based on estimator type
    if hasattr(regressor, "feature_importances_"):
        # XGBoost, RandomForest — native tree importances (gain-based)
        importances = np.array(regressor.feature_importances_)
        importance_label = "Native importance (gain)"
    elif hasattr(regressor, "regressor_") and hasattr(regressor.regressor_, "coef_"):
        # TransformedTargetRegressor wrapping ElasticNet
        importances = np.abs(regressor.regressor_.coef_)
        importance_label = "|coef| (standardized scale)"
    else:
        logger.warning(
            "plot_native_feature_importance: no .feature_importances_ or .coef_ "
            "found for '%s' — skipping.",
            model_name,
        )
        return None

    if len(importances) != len(feature_names):
        logger.warning(
            "plot_native_feature_importance: mismatch — %d importances vs "
            "%d feature names for '%s'. Using generic names.",
            len(importances), len(feature_names), model_name,
        )
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    # Sort descending, keep top_n
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in sorted_idx]
    top_vals = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_vals[::-1], align="center", color="darkorange")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel(importance_label)
    ax.set_title(
        f"Native Feature Importance - {model_name}  (top {top_n})",
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_native_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        "Native importance saved -> %s  |  Top 3: %s",
        out_path,
        ", ".join(
            f"{top_names[i]} ({top_vals[i]:.4f})"
            for i in range(min(3, len(top_names)))
        ),
    )
    return out_path


# ---------------------------------------------------------------------------
# 3.5.1 — Main entry point
# ---------------------------------------------------------------------------

def _json_safe(value):
    """Convert NaN metric values to JSON null while preserving plain numbers."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)) and np.isnan(value):
        return None
    return value


def run_evaluation(parquet_path: str) -> Dict[str, Dict]:
    """Evaluate trained regression models on the holdout test set.

    Loads model artifacts from ``ARTIFACTS_DIR``, reproduces the temporal
    split used during training, and computes for each model:

    - RMSE (primary metric, ug/m3)
    - MAE  (robustness to outliers, ug/m3)
    - R²   (explained variance)
    - RMSE per class: verde / giallo / arancio / rosso

    Finally prints a summary table (task 3.5.1), generates all diagnostic and
    feature-importance plots (tasks 3.5.2–3.5.3), saves the best pipeline as
    ``best_model.joblib``, and writes a JSON metrics file (task 3.5.4).

    Args:
        parquet_path: Path to ``daily_dataset_clean.parquet``.

    Returns:
        Dict mapping model name → metrics dict.
        Empty dict if no artifact is found.
    """
    X_test, y_test, alert_class_test, dates_test, split_metadata = _load_test_split(
        parquet_path
    )

    artifacts_dir = Path(ARTIFACTS_DIR)
    results: Dict[str, Dict] = {}
    # Keep loaded pipelines so we can reuse them for feature importance
    # without a second joblib.load call.
    pipelines: Dict[str, Pipeline] = {}

    for model_name in MODEL_NAMES:
        model_path = artifacts_dir / f"{model_name}_best.joblib"
        if not model_path.exists():
            logger.warning(
                "Artifact not found: '%s' — skipping (run train.py first).",
                model_path,
            )
            continue

        pipeline = joblib.load(model_path)
        pipelines[model_name] = pipeline
        logger.info("Loaded '%s' from %s", model_name, model_path)

        y_pred: np.ndarray = pipeline.predict(X_test)

        logger.info("--- %s ---", model_name.upper())
        metrics = compute_metrics(y_test, y_pred, alert_class_test)
        logger.info(
            "  Overall — RMSE: %.4f | MAE: %.4f | R²: %.4f",
            metrics["rmse"], metrics["mae"], metrics["r2"],
        )
        results[model_name] = metrics

        # --- Task 3.5.2: diagnostic plots ------------------------------------
        plot_diagnostics(y_test, y_pred, dates_test, model_name, PLOTS_DIR)

    if not results:
        logger.warning(
            "No model artifacts found in '%s'. Run train.py first.", ARTIFACTS_DIR
        )
        return results

    # --- Task 3.5.1: summary table ------------------------------------------
    _print_summary_table(results)

    # --- Task 3.5.3: feature importance -------------------------------------
    # Best model: permutation importance on test set (model-agnostic, unbiased).
    # Other models: native importances as a faster sanity check.
    best_model_name = min(results, key=lambda m: results[m]["rmse"])
    logger.info(
        "3.5.3 — Best model by test RMSE: '%s' (RMSE=%.4f ug/m³). "
        "Running permutation importance on test set.",
        best_model_name,
        results[best_model_name]["rmse"],
    )
    plot_permutation_importance(
        pipelines[best_model_name],
        X_test,
        y_test,
        best_model_name,
        PLOTS_DIR,
    )
    for mn, pl in pipelines.items():
        if mn != best_model_name:
            logger.info(
                "3.5.3 — '%s' (non-best): plotting native feature importance "
                "as sanity check.",
                mn,
            )
            plot_native_feature_importance(pl, mn, PLOTS_DIR)

    # --- Task 3.5.4: Save artifacts ----------------------------------
    artifacts_dir = Path(ARTIFACTS_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 3.5.4a — Save the best pipeline as best_model.joblib for easy downstream
    # consumption (callers don't need to know which model won).
    best_pipeline = pipelines[best_model_name]
    best_model_path = artifacts_dir / "best_model.joblib"
    joblib.dump(best_pipeline, best_model_path)
    logger.info(
        "3.5.4 — Best model pipeline ('%s') saved -> %s",
        best_model_name,
        best_model_path,
    )

    # 3.5.4b — Persist all metrics to JSON with metadata.
    # NaN values (class absent from test set) are serialised as null.
    metrics_path = Path(METRICS_FILE)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    import datetime  # local import to avoid top-level side-effect on import

    models_payload: Dict = {}
    for model_name, m in results.items():
        models_payload[model_name] = {
            "strategy_name": f"regression_{model_name}",
            "rmse": m["rmse"],
            "mae": m["mae"],
            "r2": m["r2"],
            "rmse_per_fascia": _json_safe(m["rmse_per_fascia"]),
            "mae_per_fascia": _json_safe(m["mae_per_fascia"]),
            "n_per_fascia": m["n_per_fascia"],
            "rmse_near_thresholds": _json_safe(m["rmse_near_thresholds"]),
            "n_near_thresholds": m["n_near_thresholds"],
            "near_threshold_delta": m["near_threshold_delta"],
            "rmse_per_class": _json_safe(m["rmse_per_class"]),
            "mae_per_class": _json_safe(m["mae_per_class"]),
            "n_per_class": m["n_per_class"],
        }

    metrics_doc: Dict = {
        "best_model": best_model_name,
        "generated_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_path": parquet_path,
        "train_date_range": split_metadata["train_date_range"],
        "test_date_range": split_metadata["test_date_range"],
        "n_test_samples": int(len(y_test)),
        "n_stations_train": int(split_metadata["n_stations_train"]),
        "n_stations_test": int(split_metadata["n_stations_test"]),
        "n_stations_intersection": int(split_metadata["n_stations_intersection"]),
        "split_metadata": split_metadata,
        "models": models_payload,
    }

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_doc, fh, indent=2)
    logger.info("3.5.4 — Metrics saved -> %s", metrics_path)
    logger.info(
        "3.5.4 — Plots directory: %s  (%d PNG files)",
        PLOTS_DIR,
        len(list(Path(PLOTS_DIR).glob("*.png"))),
    )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    _parquet = "step_3_eda/daily_dataset_clean.parquet"
    run_evaluation(_parquet)
