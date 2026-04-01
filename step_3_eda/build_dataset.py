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

    # Normalise sensor names for column naming
    df = df.assign(
        tipo=df["tiposensore"].str.lower().str.replace(".", "", regex=False),
    )

    mean_pivot = df.pivot_table(
        index=["idstazione", "data_giorno"],
        columns="tipo",
        values="valore_mean",
        aggfunc="first",
    ).rename(columns=lambda c: f"{c}_mean")

    max_pivot = df.pivot_table(
        index=["idstazione", "data_giorno"],
        columns="tipo",
        values="valore_max",
        aggfunc="first",
    ).rename(columns=lambda c: f"{c}_max")

    wide = mean_pivot.join(max_pivot).reset_index()
    return wide


# ──────────────────────────────────────────────────────────────────────────────
# Temporal features
# ──────────────────────────────────────────────────────────────────────────────

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, season, day-of-week, and weekend flag."""
    dt = df["data_giorno"]
    return df.assign(
        mese=dt.dt.month,
        stagione=dt.dt.month.map(MONTH_TO_SEASON),
        giorno_settimana=dt.dt.dayofweek,          # 0=Mon … 6=Sun
        is_weekend=dt.dt.dayofweek.isin([5, 6]),
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

    high_pressure = df["pressure_mean"] > df["pressure_mean"].median()
    low_wind = df["wind_speed_mean"] < df["wind_speed_mean"].median()
    low_blh = df["blh_min"] < 500

    df = df.assign(stagnation_flag=high_pressure & low_wind & low_blh)
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
# Missing-value strategy
# ──────────────────────────────────────────────────────────────────────────────

def apply_missing_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the missing-value strategy defined in the brainstorm.

    1. Drop rows with missing PM10 (target).
    2. Drop columns with >50 % missing.
    3. Drop stations with >50 % missing PM10.
    4. ffill→bfill pollutants within each station.
    5. Median imputation for weather columns.
    """
    n_before = len(df)

    # 1 — target must exist
    df = df.dropna(subset=["pm10"])

    # 2 — drop columns above threshold
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    # Never drop identifier / target columns
    protected = {"idstazione", "data_giorno", "pm10", "classe_allerta"}
    cols_to_drop = [c for c in cols_to_drop if c not in protected]
    if cols_to_drop:
        log.info("Dropping %d columns (>%.0f%% missing): %s",
                 len(cols_to_drop), MISSING_THRESHOLD * 100, cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    # 3 — drop stations with sparse PM10 coverage
    total_days = df["data_giorno"].nunique()
    station_days = df.groupby("idstazione")["data_giorno"].nunique()
    bad_stations = station_days[station_days < total_days * (1 - MISSING_THRESHOLD)].index
    if len(bad_stations):
        log.info("Dropping %d stations with <%.0f%% day coverage.",
                 len(bad_stations), (1 - MISSING_THRESHOLD) * 100)
        df = df[~df["idstazione"].isin(bad_stations)]

    # 4 — ffill / bfill pollutants per station
    pollutant_cols = [c for c in df.columns
                      if c.startswith(("no2_", "o3_", "co_", "pm25_"))]
    if pollutant_cols:
        df[pollutant_cols] = (
            df.sort_values(["idstazione", "data_giorno"])
              .groupby("idstazione")[pollutant_cols]
              .transform(lambda s: s.ffill().bfill())
        )

    # 5 — median imputation for weather
    weather_prefixes = (
        "temp_", "humidity_", "dewpoint_", "precip_", "pressure_",
        "cloud_", "wind_", "visibility_", "radiation_", "blh_", "fog_",
    )
    weather_cols = [c for c in df.columns if c.startswith(weather_prefixes)]
    for col in weather_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    log.info("Missing-value strategy: %d → %d rows.", n_before, len(df))
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

    # ── Join ─────────────────────────────────────────────────────────────────
    df = pm10.copy()

    if not weather.empty:
        df = df.merge(weather, on=["idstazione", "data_giorno"], how="left")

    if not pollutants.empty:
        df = df.merge(pollutants, on=["idstazione", "data_giorno"], how="left")

    log.info("Joined dataset: %d rows, %d columns.", *df.shape)

    # ── Feature engineering ──────────────────────────────────────────────────
    df = _add_temporal_features(df)
    df = _add_alert_class(df)
    df = _add_stagnation_flag(df)

    return df


def save_dataset(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save the analysis DataFrame as Parquet."""
    path = output_dir / "daily_dataset.parquet"
    df.to_parquet(path, index=False)
    log.info("Saved dataset: %s (%d rows, %d cols).", path, *df.shape)
    return path
