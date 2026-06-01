"""Step 3 — Build the daily analysis DataFrame.

Joins PM10 measurements with aggregated weather and other pollutants,
adds temporal and derived features, and assigns alert classes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql

from step_3_eda.config import (
    MISSING_THRESHOLD,
    MONTH_TO_SEASON,
    PM10_LABELS,
    PM10_THRESHOLDS,
    STAGNATION_BLH_THRESHOLD,
    STAGNATION_INDEX_BLH_MIN,
    STAGNATION_INDEX_WIND_MIN,
    STAGNATION_PRESSURE_THRESHOLD,
    STAGNATION_WIND_THRESHOLD,
)
from step_3_eda.db import (
    load_pm10_daily,
    load_pollutants_daily_agg,
    load_weather_daily_agg,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pivot helper
# ──────────────────────────────────────────────────────────────────────────────

def _pivot_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form pollutant rows into wide columns.

    Input columns: idstazione, tiposensore, data_giorno, valore_mean, valore_max
    Output: one row per (idstazione, data_giorno) with columns like
            no2_mean, no2_max, o3_mean, o3_max, co_mean, co_max, pm25_mean, pm25_max
    """
    if df.empty:
        return pd.DataFrame()

    # Map full ARPA Italian sensor names → short codes used as column prefixes
    _TIPO_MAP: dict[str, str] = {
        "biossido di azoto":   "no2",
        "ozono":               "o3",
        "monossido di carbonio": "co",
        "particolato (pm2,5)": "pm25",  # Italian decimal comma variant
        "particolato (pm2.5)": "pm25",
        "particolato (pm25)":  "pm25",  # after dot-removal fallback
    }

    df = df.assign(
        tipo=df["tiposensore"]
            .str.lower()
            .str.strip()
            .map(_TIPO_MAP)
    )
    df = df.dropna(subset=["tipo"])
    if df.empty:
        return pd.DataFrame()

    mean_pivot = df.pivot_table(
        index=["idstazione", "data_giorno"],
        columns="tipo",
        values="valore_mean",
        aggfunc="mean",
    ).rename(columns=lambda c: f"{c}_mean")
    mean_pivot.columns.name = None

    max_pivot = df.pivot_table(
        index=["idstazione", "data_giorno"],
        columns="tipo",
        values="valore_max",
        aggfunc="max",
    ).rename(columns=lambda c: f"{c}_max")
    max_pivot.columns.name = None

    wide = mean_pivot.join(max_pivot).reset_index()
    return wide


# ──────────────────────────────────────────────────────────────────────────────
# Time-Series Utilities (Lags & Rolling)
# ──────────────────────────────────────────────────────────────────────────────

def _fill_date_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure continuous daily time series per station to safely compute lags."""
    df = df.copy()
    df["data_giorno"] = pd.to_datetime(df["data_giorno"])
    
    station_ranges = df.groupby("idstazione")["data_giorno"].agg(["min", "max"])
    all_idx = []
    for st_id, row in station_ranges.iterrows():
        idx = pd.MultiIndex.from_product(
            [[st_id], pd.date_range(row["min"], row["max"])],
            names=["idstazione", "data_giorno"]
        )
        all_idx.extend(idx.values)
        
    full_idx = pd.MultiIndex.from_tuples(all_idx, names=["idstazione", "data_giorno"])
    df_filled = df.set_index(["idstazione", "data_giorno"]).reindex(full_idx).reset_index()
    
    # Forward/backward fill the static metadata across the newly created blank dates
    static_cols = ["nomestazione", "provincia", "comune", "quota", "lat", "lng"]
    static_cols = [c for c in static_cols if c in df_filled.columns]
    if static_cols:
        df_filled[static_cols] = df_filled.groupby("idstazione")[static_cols].transform(lambda s: s.ffill().bfill())
        
    return df_filled


def _add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-day, 2-day lags and 3-day rolling means for key variables."""
    df = df.sort_values(["idstazione", "data_giorno"])

    if "pm10" in df.columns:
        df["pm10_lag1"] = df.groupby("idstazione")["pm10"].shift(1)
        df["pm10_lag2"] = df.groupby("idstazione")["pm10"].shift(2)
        df["pm10_roll7"] = df.groupby("idstazione")["pm10"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
        df["pm10_roll3"] = df.groupby("idstazione")["pm10"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        df["pm10_diff"] = df["pm10_lag1"] - df["pm10_lag2"]

    # As per Brainstorm: Lag of key weather variables (1-day, 2-day lag)
    for col in ["pressure_mean", "wind_speed_mean", "blh_mean", "temp_mean"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("idstazione")[col].shift(1)
            df[f"{col}_lag2"] = df.groupby("idstazione")[col].shift(2)

    if "pressure_mean" in df.columns:
        df["pressure_roll3"] = df.groupby("idstazione")["pressure_mean"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=3).mean()
        )
    if "wind_speed_mean" in df.columns:
        df["wind_speed_roll3"] = df.groupby("idstazione")["wind_speed_mean"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=3).mean()
        )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Temporal features
# ──────────────────────────────────────────────────────────────────────────────

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, season, day-of-week, weekend flag, and cyclic encodings."""
    dt = df["data_giorno"]
    mese = dt.dt.month
    dow = dt.dt.dayofweek
    return df.assign(
        mese=mese,
        stagione=mese.map(MONTH_TO_SEASON),
        giorno_settimana=dow,                       # 0=Mon … 6=Sun
        is_weekend=dow.isin([5, 6]),
        heating_season=mese.isin([10, 11, 12, 1, 2, 3]).astype(int),
        # Cyclic encoding: December (12) and January (1) end up close to each other
        mese_sin=np.sin(2 * np.pi * mese / 12),
        mese_cos=np.cos(2 * np.pi * mese / 12),
        dow_sin=np.sin(2 * np.pi * dow / 7),
        dow_cos=np.cos(2 * np.pi * dow / 7),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Derived features
# ──────────────────────────────────────────────────────────────────────────────

def _add_stagnation_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary stagnation flag: high pressure + low wind + low BLH."""
    needed = {"pressure_mean", "wind_speed_mean", "blh_min"}
    if not needed.issubset(df.columns):
        log.warning("Cannot compute stagnation flag — missing columns: %s", needed - set(df.columns))
        df["stagnation_flag"] = np.nan
        return df

    has_data = df[["pressure_mean", "wind_speed_mean", "blh_min"]].notna().all(axis=1)
    high_pressure = df["pressure_mean"] > STAGNATION_PRESSURE_THRESHOLD
    low_wind = df["wind_speed_mean"] < STAGNATION_WIND_THRESHOLD
    low_blh = df["blh_min"] < STAGNATION_BLH_THRESHOLD

    flag = high_pressure & low_wind & low_blh
    df = df.assign(stagnation_flag=flag.where(has_data))  # NaN where any input is missing
    return df


def _add_stagnation_index(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous stagnation index: inverse of dispersion capacity.

    Formula: 1 / (wind_speed_mean * blh_min * (1 + precip_sum))

    Wind and BLH are clipped to physical minimums so the index stays bounded:
    max value = 1 / (STAGNATION_INDEX_WIND_MIN * STAGNATION_INDEX_BLH_MIN * 1) = 1.0

    Higher values → stronger stagnation → higher PM10 accumulation expected.
    """
    needed = {"wind_speed_mean", "blh_min"}
    if not needed.issubset(df.columns):
        log.warning(
            "Cannot compute stagnation_index — missing columns: %s",
            needed - set(df.columns),
        )
        df["stagnation_index"] = np.nan
        return df

    precip = df["precip_sum"].fillna(0.0) if "precip_sum" in df.columns else pd.Series(0.0, index=df.index)

    stagnation_index = 1.0 / (
        df["wind_speed_mean"].clip(lower=STAGNATION_INDEX_WIND_MIN)
        * df["blh_min"].clip(lower=STAGNATION_INDEX_BLH_MIN)
        * (1.0 + precip)
    )
    df = df.assign(stagnation_index=stagnation_index)
    return df


def _add_alert_class(df: pd.DataFrame) -> pd.DataFrame:
    """Assign the 4-class alert label from PM10 value."""
    df = df.assign(
        classe_allerta=pd.cut(
            df["pm10"],
            bins=PM10_THRESHOLDS,
            labels=PM10_LABELS,
            right=False,
        ),
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Redundant-feature removal
# ──────────────────────────────────────────────────────────────────────────────

# Columns with |r| > 0.98 against a kept alternative — dropping them improves
# ElasticNet conditioning and reduces noise in tree models.
_REDUNDANT_COLS = ["temp_max", "temp_min", "pressure_mean_lag2"]


def _drop_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop highly-correlated features identified in the EDA audit."""
    cols_to_drop = [c for c in _REDUNDANT_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


# ──────────────────────────────────────────────────────────────────────────────
# Missing-value strategy
# ──────────────────────────────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Safe pre-split cleaning: no global statistics, no data leakage.

    1. Drop columns with >50 % missing.
    2. Drop stations with sparse PM10 coverage.
    3. Forward-fill pollutants and lag/rolling columns per station (past-only).
    4. Drop rows with missing target (PM10).
    """
    df = df.copy()
    df = df.sort_values(["idstazione", "data_giorno"]).reset_index(drop=True)
    n_before = len(df)

    # 1 — Drop columns above threshold
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    protected = {"idstazione", "data_giorno", "pm10", "classe_allerta"}
    cols_to_drop = [c for c in cols_to_drop if c not in protected]
    if cols_to_drop:
        log.info("Dropping %d columns (>%.0f%% missing): %s",
                 len(cols_to_drop), MISSING_THRESHOLD * 100, cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    # 2 — Drop stations with sparse PM10 coverage
    total_days = df["data_giorno"].nunique()
    station_days = df.dropna(subset=["pm10"]).groupby("idstazione")["data_giorno"].nunique()
    bad_stations = station_days[station_days < total_days * (1 - MISSING_THRESHOLD)].index
    if len(bad_stations):
        log.info("Dropping %d stations with <%.0f%% day coverage.",
                 len(bad_stations), (1 - MISSING_THRESHOLD) * 100)
        df = df[~df["idstazione"].isin(bad_stations)]

    # 3 — Forward-fill per station (past-only, no leakage)
    pollutant_cols = [c for c in df.columns if c.startswith(("no2_", "o3_", "co_", "pm25_"))]
    if pollutant_cols:
        df[pollutant_cols] = df.groupby("idstazione")[pollutant_cols].transform(lambda s: s.ffill())

    lag_cols = [c for c in df.columns if c.endswith(("_lag1", "_lag2", "_roll3", "_roll7", "_diff"))]
    if lag_cols:
        df[lag_cols] = df.groupby("idstazione")[lag_cols].transform(lambda s: s.ffill())

    # 4 — Drop rows with missing target
    df = df.dropna(subset=["pm10"])

    log.info("clean_dataset: %d → %d rows.", n_before, len(df))
    return df


def impute_missing(
    df: pd.DataFrame,
    *,
    medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Median imputation — MUST be called AFTER train/test split.

    Parameters
    ----------
    df : DataFrame to impute.
    medians : Pre-computed medians (from training set). If *None*, medians are
              computed from *df* itself (training mode).

    Returns
    -------
    (imputed_df, medians_dict) — the dict can be saved and reused on the test set.
    """
    df = df.copy()
    fit_mode = medians is None

    # Identify columns that need imputation
    weather_prefixes = (
        "temp_", "humidity_", "dewpoint_", "precip_", "pressure_",
        "cloud_", "wind_", "visibility_", "radiation_", "blh_", "fog_",
    )
    weather_cols = [c for c in df.columns if c.startswith(weather_prefixes)
                    and not c.endswith(("_lag1", "_lag2", "_roll3"))]
    pollutant_cols = [c for c in df.columns if c.startswith(("no2_", "o3_", "co_", "pm25_"))]
    lag_cols = [c for c in df.columns if c.endswith(("_lag1", "_lag2", "_roll3", "_roll7", "_diff"))]
    cols_to_impute = weather_cols + pollutant_cols + lag_cols

    if fit_mode:
        medians = {col: df[col].median() for col in cols_to_impute if df[col].isna().any()}
    else:
        medians = dict(medians)  # defensive copy

    for col in cols_to_impute:
        if df[col].isna().any():
            med = medians.get(col)
            if med is not None:
                df[col] = df[col].fillna(med)
            else:
                log.warning("impute_missing: column '%s' has NaN but no median provided.", col)

    n_remaining = df.isnull().sum().sum()
    if n_remaining:
        log.info("impute_missing: %d NaN remaining in non-imputed columns.", n_remaining)

    return df, medians


def apply_missing_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper — calls clean_dataset + impute_missing.

    .. deprecated::
        Use ``clean_dataset`` + ``impute_missing`` separately in ML pipelines
        to avoid data leakage (median must be fit on training data only).
    """
    log.warning("apply_missing_strategy is deprecated: use clean_dataset + impute_missing "
                "separately to avoid data leakage in train/test splits.")
    df = clean_dataset(df)
    df, _ = impute_missing(df)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main builder
# ──────────────────────────────────────────────────────────────────────────────

def build_daily_dataset(conn: pymysql.Connection) -> pd.DataFrame:
    """Build the full daily analysis DataFrame from MySQL."""

    # ── Load raw tables ──────────────────────────────────────────────────────
    pm10 = load_pm10_daily(conn)
    weather = load_weather_daily_agg(conn)
    pollutants_raw = load_pollutants_daily_agg(conn)
    pollutants = _pivot_pollutants(pollutants_raw)

    # ── Reindex to fill date gaps ────────────────────────────────────────────
    df = _fill_date_gaps(pm10)

    # ── Join ─────────────────────────────────────────────────────────────────
    if not weather.empty:
        weather["data_giorno"] = pd.to_datetime(weather["data_giorno"])
        df = df.merge(weather, on=["idstazione", "data_giorno"], how="left")

    if not pollutants.empty:
        pollutants["data_giorno"] = pd.to_datetime(pollutants["data_giorno"])
        df = df.merge(pollutants, on=["idstazione", "data_giorno"], how="left")

    # ── Industrial proximity (static per-station feature) ────────────────
    industrial_path = Path(__file__).parent.parent / "data" / "raw" / "industrial_proximity.parquet"
    if industrial_path.exists():
        industrial = pd.read_parquet(industrial_path)
        df = df.merge(industrial, on="idstazione", how="left")
        log.info("Merged industrial proximity features.")
    else:
        log.warning("No industrial_proximity.parquet found — feature skipped.")

    log.info("Joined dataset: %d rows, %d columns.", *df.shape)

    # ── Feature engineering ──────────────────────────────────────────────────
    df = _add_temporal_features(df)
    df = _add_lag_and_rolling_features(df)
    df = _add_alert_class(df)
    df = _add_stagnation_flag(df)
    df = _add_stagnation_index(df)
    df = _drop_redundant_features(df)

    return df


def save_dataset(df: pd.DataFrame, output_dir: Path, filename: str = "daily_dataset.parquet") -> Path:
    """Save the analysis DataFrame as Parquet."""
    path = output_dir / filename
    df.to_parquet(path, index=False)
    log.info("Saved dataset: %s (%d rows, %d cols).", path, *df.shape)
    return path
