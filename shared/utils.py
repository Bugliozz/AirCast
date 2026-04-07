"""Shared utilities for Step 4 (Regression) and Step 5 (Classification).

This module provides reusable functions for:
- Temporal train-test splitting (no shuffle, stratified by date)
- Missing value imputation post-split
- ColumnTransformer preprocessing pipelines
- TimeSeriesSplit cross-validation wrapper with proper date-based folding
"""

import logging
from typing import Tuple, Optional, Dict, List, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


def temporal_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    date_col: str = "data_giorno",
    train_ratio: float = 0.70,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into train/test by time, keeping all stations of a day together.

    All rows belonging to the same day are assigned to the same set (train or test),
    preventing temporal leakage. No shuffle is applied.

    Args:
        df: Input DataFrame with one row per station-day.
        target_col: Name of the target column (y).
        date_col: Name of the date column used for ordering.
        train_ratio: Fraction of unique days to use for training (default 0.70).
        drop_cols: Additional columns to drop from X (e.g. identifiers).

    Returns:
        Tuple (X_train, y_train, X_test, y_test, train_dates) where train_dates
        is a Series of the date_col values aligned with X_train rows.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Sort entire dataset by date to ensure temporal order
    df_sorted = df.sort_values(date_col).reset_index(drop=True)

    # Unique days in sorted order — split point is on days, NOT rows
    unique_days = df_sorted[date_col].drop_duplicates().sort_values().reset_index(drop=True)
    n_train_days = int(len(unique_days) * train_ratio)

    if n_train_days == 0 or n_train_days >= len(unique_days):
        raise ValueError(
            f"train_ratio={train_ratio} yields {n_train_days}/{len(unique_days)} train days. "
            "Adjust the ratio so that both sets are non-empty."
        )

    cutoff_day = unique_days.iloc[n_train_days - 1]
    first_test_day = unique_days.iloc[n_train_days]

    logger.info(
        "Temporal split: %d train days (up to %s) | %d test days (from %s)",
        n_train_days,
        cutoff_day,
        len(unique_days) - n_train_days,
        first_test_day,
    )

    train_mask = df_sorted[date_col] <= cutoff_day
    train_df = df_sorted[train_mask]
    test_df = df_sorted[~train_mask]

    logger.info(
        "Row counts — train: %d | test: %d (%.1f%% / %.1f%%)",
        len(train_df),
        len(test_df),
        100 * len(train_df) / len(df_sorted),
        100 * len(test_df) / len(df_sorted),
    )

    # Columns to exclude from X
    id_cols = [date_col, target_col]
    if drop_cols:
        id_cols = list(set(id_cols + drop_cols))

    feature_cols = [c for c in df_sorted.columns if c not in id_cols]

    X_train = train_df[feature_cols].reset_index(drop=True)
    y_train = train_df[target_col].reset_index(drop=True)
    X_test = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df[target_col].reset_index(drop=True)
    train_dates = train_df[date_col].reset_index(drop=True)

    return X_train, y_train, X_test, y_test, train_dates


def impute_missing(
    df: pd.DataFrame,
    *,
    medians: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Median imputation — MUST be called AFTER train/test split.

    Call on train set first (no medians) to fit, then reuse the returned
    medians dict on the test set to avoid leakage:

        X_train, train_medians = impute_missing(X_train)
        X_test, _              = impute_missing(X_test, medians=train_medians)

    Args:
        df: DataFrame to impute (copied internally — original is not mutated).
        medians: Pre-computed medians from the training set. If None, medians
            are computed from *df* itself (training / fit mode).

    Returns:
        Tuple (imputed_df, medians_dict). The dict can be persisted and reused
        on the test set (or future inference data).
    """
    df = df.copy()
    fit_mode = medians is None

    # Identify columns that need imputation by naming convention
    weather_prefixes = (
        "temp_", "humidity_", "dewpoint_", "precip_", "pressure_",
        "cloud_", "wind_", "visibility_", "radiation_", "blh_", "fog_",
    )
    weather_cols = [
        c for c in df.columns
        if c.startswith(weather_prefixes) and not c.endswith(("_lag1", "_lag2", "_roll3"))
    ]
    pollutant_cols = [c for c in df.columns if c.startswith(("no2_", "o3_", "co_", "pm25_"))]
    lag_cols = [c for c in df.columns if c.endswith(("_lag1", "_lag2", "_roll3", "_roll7", "_diff"))]
    cols_to_impute = weather_cols + pollutant_cols + lag_cols

    if fit_mode:
        medians = {col: float(df[col].median()) for col in cols_to_impute if df[col].isna().any()}
        logger.info("impute_missing (fit): computed medians for %d columns.", len(medians))
    else:
        medians = dict(medians)  # defensive copy

    for col in cols_to_impute:
        if df[col].isna().any():
            med = medians.get(col)
            if med is not None:
                df[col] = df[col].fillna(med)
            else:
                logger.warning(
                    "impute_missing: column '%s' has NaN but no median provided.", col
                )

    n_remaining = int(df.isnull().sum().sum())
    if n_remaining:
        logger.info(
            "impute_missing: %d NaN remaining in non-imputed columns.", n_remaining
        )

    return df, medians


# Default columns to drop (identifiers with no predictive value, and
# target-derived columns that would cause label leakage)
_DEFAULT_DROP_COLS: List[str] = [
    "idstazione", "nomestazione", "comune", "data_giorno", "classe_allerta",
]

# Default categorical columns and their known categories for OrdinalEncoder
_STAGIONE_CATEGORIES: List[str] = ["inverno", "primavera", "estate", "autunno"]
_DEFAULT_CAT_COLS: List[str] = ["stagione", "provincia"]


def build_preprocessor(
    X: pd.DataFrame,
    *,
    model_type: Literal["linear", "tree"] = "linear",
    categorical_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build a ColumnTransformer for preprocessing feature matrices.

    Two modes controlled by *model_type*:

    - ``"linear"``: numeric features are standardised with :class:`StandardScaler`
      (required for ElasticNet / Logistic Regression).
    - ``"tree"``: numeric features pass through unchanged — tree-based models
      (XGBoost, RandomForest) are scale-invariant.

    In both modes:

    - Categorical columns (``stagione``, ``provincia`` by default) are encoded
      with :class:`OrdinalEncoder`.  Unknown categories seen at inference time
      are mapped to ``-1`` (``handle_unknown="use_encoded_value"``).
    - Identifier columns (``idstazione``, ``nomestazione``, ``comune``,
      ``data_giorno`` by default) are **dropped**.

    The transformer is *not* fitted here; call ``preprocessor.fit(X_train)``
    followed by ``preprocessor.transform(X_test)`` in the training pipeline,
    or pass it as the first step of an sklearn ``Pipeline``.

    Args:
        X: Feature DataFrame (post-split, post-imputation).  Used only to
            discover which columns are present — it is not mutated or fitted.
        model_type: ``"linear"`` applies StandardScaler; ``"tree"`` passes
            numeric features through unchanged.
        categorical_cols: Columns to encode with OrdinalEncoder.  Defaults to
            ``["stagione", "provincia"]`` (filtered to those present in *X*).
        drop_cols: Columns to drop entirely.  Defaults to
            ``["idstazione", "nomestazione", "comune", "data_giorno"]``
            (filtered to those present in *X*).

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`.

    Raises:
        ValueError: If *model_type* is not ``"linear"`` or ``"tree"``.

    Example::

        X_train, y_train, X_test, y_test, train_dates = temporal_train_test_split(df, "pm10")
        X_train, train_medians = impute_missing(X_train)
        X_test, _ = impute_missing(X_test, medians=train_medians)

        pre = build_preprocessor(X_train, model_type="linear")
        pipeline = Pipeline([("pre", pre), ("model", ElasticNet())])
        pipeline.fit(X_train, y_train)
    """
    if model_type not in ("linear", "tree"):
        raise ValueError(f"model_type must be 'linear' or 'tree', got '{model_type}'.")

    # Resolve defaults, keeping only columns that actually exist in X
    drop_cols = [c for c in (_DEFAULT_DROP_COLS if drop_cols is None else drop_cols) if c in X.columns]
    categorical_cols = [c for c in (_DEFAULT_CAT_COLS if categorical_cols is None else categorical_cols) if c in X.columns]

    excluded = set(drop_cols) | set(categorical_cols)
    numeric_cols = [c for c in X.columns if c not in excluded]

    logger.info(
        "build_preprocessor(model_type=%s): %d numeric | %d categorical | %d dropped",
        model_type,
        len(numeric_cols),
        len(categorical_cols),
        len(drop_cols),
    )

    # Build the OrdinalEncoder; if 'stagione' is present, supply known categories
    # so the encoder does not depend on the order they happen to appear in X_train.
    # For other categorical columns, derive unique categories from X so that
    # the categories list is always a list-of-lists (sklearn requirement).
    if categorical_cols:
        cat_categories: List = []
        for col in categorical_cols:
            if col == "stagione":
                cat_categories.append(_STAGIONE_CATEGORIES)
            else:
                # Derive sorted unique values from X; unknown values at inference
                # time are handled by handle_unknown="use_encoded_value".
                cat_categories.append(sorted(X[col].dropna().unique().tolist()))
        cat_transformer = OrdinalEncoder(
            categories=cat_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )
    else:
        cat_transformer = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )

    if model_type == "linear":
        # Impute any residual NaNs (columns not covered by impute_missing)
        # before scaling — ElasticNet cannot handle NaN.
        num_transformer = SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        num_transformer = "passthrough"

    transformers: list = []
    if numeric_cols:
        transformers.append(("num", num_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_transformer, categorical_cols))
    if drop_cols:
        transformers.append(("drop", "drop", drop_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def make_temporal_cv_splits(
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    *,
    date_col: str = "data_giorno",
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build TimeSeriesSplit indices grouped by day to prevent intra-day leakage.

    Sklearn's :class:`TimeSeriesSplit` operates on row indices — with multiple
    stations per day, rows from the same day could land in both train and
    validation folds.  This function splits on **unique days** first, then
    expands the day-level fold boundaries back to row indices.

    Args:
        X_train: Training feature DataFrame (must contain *date_col*).
        y_train: Optional training target Series.  When provided, a per-fold
            class-coverage check is logged (useful for classification targets).
        date_col: Column with the date used for temporal ordering.
        n_splits: Number of CV folds (default 5).

    Returns:
        List of ``(train_indices, val_indices)`` numpy arrays, one per fold.
        These can be passed directly to sklearn search objects via the ``cv``
        parameter.

    Raises:
        ValueError: If *date_col* is missing or there are fewer unique days
            than ``n_splits + 1``.
    """
    if date_col not in X_train.columns:
        raise ValueError(
            f"Date column '{date_col}' not found in X_train. "
            "Make sure identifiers are still present before calling this function."
        )

    # --- 1. Map each unique day to the row indices that belong to it ----------
    unique_days = np.sort(X_train[date_col].unique())
    n_days = len(unique_days)

    if n_days < n_splits + 1:
        raise ValueError(
            f"Need at least {n_splits + 1} unique days for {n_splits} folds, "
            f"but only {n_days} found."
        )

    # day → array of row positions (integer-location based)
    day_to_rows: Dict[object, np.ndarray] = {}
    for day in unique_days:
        day_to_rows[day] = np.where(X_train[date_col].values == day)[0]

    # --- 2. Split *days* using sklearn TimeSeriesSplit -------------------------
    tscv = TimeSeriesSplit(n_splits=n_splits)
    day_indices = np.arange(n_days)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for fold_idx, (day_train_idx, day_val_idx) in enumerate(tscv.split(day_indices)):
        # Expand day-level indices to row-level indices
        train_rows = np.concatenate(
            [day_to_rows[unique_days[i]] for i in day_train_idx]
        )
        val_rows = np.concatenate(
            [day_to_rows[unique_days[i]] for i in day_val_idx]
        )

        splits.append((train_rows, val_rows))

        train_day_range = (unique_days[day_train_idx[0]], unique_days[day_train_idx[-1]])
        val_day_range = (unique_days[day_val_idx[0]], unique_days[day_val_idx[-1]])

        logger.info(
            "CV fold %d/%d — train: %d days (%s → %s), %d rows | "
            "val: %d days (%s → %s), %d rows",
            fold_idx + 1,
            n_splits,
            len(day_train_idx),
            train_day_range[0],
            train_day_range[1],
            len(train_rows),
            len(day_val_idx),
            val_day_range[0],
            val_day_range[1],
            len(val_rows),
        )

        # --- 3. Class-coverage sanity check (classification targets) ----------
        if y_train is not None:
            train_classes = set(y_train.iloc[train_rows].unique())
            val_classes = set(y_train.iloc[val_rows].unique())
            all_classes = set(y_train.unique())

            missing_in_train = all_classes - train_classes
            missing_in_val = all_classes - val_classes

            if missing_in_train:
                logger.warning(
                    "CV fold %d/%d — classes MISSING in train: %s",
                    fold_idx + 1, n_splits, sorted(missing_in_train),
                )
            if missing_in_val:
                logger.warning(
                    "CV fold %d/%d — classes MISSING in val: %s",
                    fold_idx + 1, n_splits, sorted(missing_in_val),
                )

    logger.info(
        "make_temporal_cv_splits: generated %d folds from %d unique days.",
        n_splits, n_days,
    )

    return splits
