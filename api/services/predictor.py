"""Prediction service — load trained models and generate PM10 + alert forecasts.

Models (regression + classification) are loaded once at application startup.
The ``daily_dataset_clean.parquet`` artifact is used only as a **static
station registry** (lat/lng, industrial proximity, provincia, …).  Recent
observations for lag/rolling features are pulled ephemerally from GCS by
``api.services.recent_data`` so training data and serving data stay fully
separated — sporadic API calls cannot punch gaps in the training store.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from fastapi import HTTPException

from api.services.recent_data import (
    compute_data_quality,
    fetch_recent_for_station,
)
from api.services.weather import fetch_forecast_weather
from step_3_eda.config import (
    MONTH_TO_SEASON,
    PM10_LABELS,
    PM10_THRESHOLDS,
    STAGNATION_BLH_THRESHOLD,
    STAGNATION_INDEX_BLH_MIN,
    STAGNATION_INDEX_WIND_MIN,
    STAGNATION_PRESSURE_THRESHOLD,
    STAGNATION_WIND_THRESHOLD,
)
from step_5_classification.config import LABEL_MAP

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
REGRESSION_MODEL_PATH = _ROOT / "step_4_regression" / "artifacts" / "best_model.joblib"
CLASSIFICATION_MODEL_PATH = _ROOT / "step_5_classification" / "artifacts" / "best_model.joblib"
FEATURE_STORE_PATH = _ROOT / "step_3_eda" / "daily_dataset_clean.parquet"

_INDEX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# ---------------------------------------------------------------------------
# Module-level singletons — populated at startup by ``load_models()``
# ---------------------------------------------------------------------------
_regression_model: Any = None
_classification_model: Any = None
_feature_store: Optional[pd.DataFrame] = None


class ModelNotFoundError(RuntimeError):
    """Raised when required model artifacts are missing at startup."""


class StationNotFoundError(LookupError):
    """Raised when a requested station id has no historical records."""


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_models() -> None:
    """Load regression + classification models and the feature store once.

    Raises
    ------
    ModelNotFoundError
        If either ``.joblib`` file or the feature-store parquet is missing.
    """
    global _regression_model, _classification_model, _feature_store

    for path in (REGRESSION_MODEL_PATH, CLASSIFICATION_MODEL_PATH, FEATURE_STORE_PATH):
        if not path.exists():
            raise ModelNotFoundError(f"Required artifact missing: {path}")

    _regression_model = joblib.load(REGRESSION_MODEL_PATH)
    _classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    _feature_store = pd.read_parquet(FEATURE_STORE_PATH)
    _feature_store["data_giorno"] = pd.to_datetime(_feature_store["data_giorno"])

    log.info(
        "Predictor loaded: regression=%s, classification=%s, feature_store=%d rows.",
        REGRESSION_MODEL_PATH.name,
        CLASSIFICATION_MODEL_PATH.name,
        len(_feature_store),
    )


def _require_loaded() -> pd.DataFrame:
    if _feature_store is None or _regression_model is None or _classification_model is None:
        raise ModelNotFoundError("Predictor not initialised — call load_models() first.")
    return _feature_store


# ---------------------------------------------------------------------------
# Station / lag retrieval
# ---------------------------------------------------------------------------

def get_lag_features(station_id: str, conn: Any = None) -> pd.DataFrame:
    """Return the most recent daily rows for *station_id* from GCS.

    Downloads the rolling 7-day window from the bucket (no MySQL round-trip,
    no parquet read) and augments each row with the station's static
    metadata so downstream code keeps the same column contract.
    """
    _require_loaded()
    recent = fetch_recent_for_station(station_id)
    if recent.empty:
        raise StationNotFoundError(
            f"No recent GCS data found for station '{station_id}'."
        )

    meta = get_station_metadata(station_id)
    for col in (
        "nomestazione", "comune", "provincia", "quota", "lat", "lng",
        "dist_industrial_km", "n_industrial_zones_15km",
    ):
        recent[col] = meta.get(col)

    return recent.sort_values("data_giorno").tail(8).reset_index(drop=True)


def get_station_metadata(station_id: str) -> Dict[str, Any]:
    """Return static per-station metadata from the feature store."""
    fs = _require_loaded()
    rows = fs[fs["idstazione"].astype(str) == str(station_id)]
    if rows.empty:
        raise StationNotFoundError(f"Station '{station_id}' not found.")
    latest = rows.sort_values("data_giorno").iloc[-1]
    return {
        "idstazione": str(latest["idstazione"]),
        "nomestazione": latest["nomestazione"],
        "comune": latest["comune"],
        "provincia": latest["provincia"],
        "quota": float(latest["quota"]),
        "lat": float(latest["lat"]),
        "lng": float(latest["lng"]),
        "dist_industrial_km": float(latest["dist_industrial_km"]),
        "n_industrial_zones_15km": float(latest["n_industrial_zones_15km"]),
    }


# ---------------------------------------------------------------------------
# Feature engineering for a single future day
# ---------------------------------------------------------------------------

def _stagnation_flag(pressure: float, wind: float, blh_min: float) -> Optional[float]:
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (pressure, wind, blh_min)):
        return np.nan
    return float(
        pressure > STAGNATION_PRESSURE_THRESHOLD
        and wind < STAGNATION_WIND_THRESHOLD
        and blh_min < STAGNATION_BLH_THRESHOLD
    )


def _stagnation_index(wind: float, blh_min: float, precip: Optional[float]) -> Optional[float]:
    if wind is None or blh_min is None or (isinstance(wind, float) and np.isnan(wind)) \
            or (isinstance(blh_min, float) and np.isnan(blh_min)):
        return np.nan
    p = 0.0 if precip is None or (isinstance(precip, float) and np.isnan(precip)) else float(precip)
    return 1.0 / (
        max(float(wind), STAGNATION_INDEX_WIND_MIN)
        * max(float(blh_min), STAGNATION_INDEX_BLH_MIN)
        * (1.0 + p)
    )


def _compute_pm10_feature_context(lag_data: pd.DataFrame) -> Dict[str, Any]:
    """Compute PM10 lag/rolling values from the recent window."""
    pm10_series = pd.to_numeric(lag_data["pm10"], errors="coerce").ffill(limit=2)

    valid = pm10_series.dropna().reset_index(drop=True)
    roll7_window = pm10_series.tail(7).dropna()
    roll3_window = pm10_series.tail(3).dropna()

    lag1 = float(valid.iloc[-1]) if not valid.empty else np.nan
    lag2 = float(valid.iloc[-2]) if len(valid) >= 2 else np.nan

    roll7 = float(roll7_window.mean()) if len(roll7_window) >= 3 else np.nan
    roll3 = float(roll3_window.mean()) if len(roll3_window) >= 2 else np.nan

    return {
        "lag1": lag1,
        "lag2": lag2,
        "roll7": roll7,
        "roll3": roll3,
        "pm10_diff": lag1 - lag2 if not (np.isnan(lag1) or np.isnan(lag2)) else np.nan,
    }


def build_features(
    station_id: str,
    target_date: date,
    weather_daily: Dict[str, Any],
    lag_data: pd.DataFrame,
    pm10_context: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Construct a single-row feature frame for *target_date*.

    Parameters
    ----------
    station_id:
        Station identifier (for logging only — metadata is resolved from
        ``lag_data``).
    target_date:
        The calendar day the prediction refers to.
    weather_daily:
        One daily aggregated weather record (as returned by
        ``fetch_forecast_weather``).
    lag_data:
        Recent historical rows for the station (ascending by date).  The last
        row is used for lag-1 values, the one before it for lag-2 / diff, etc.
    """
    if lag_data.empty:
        raise StationNotFoundError(f"No historical data for station '{station_id}'.")

    last = lag_data.iloc[-1]
    prev = lag_data.iloc[-2] if len(lag_data) >= 2 else last

    # ---- Static metadata -------------------------------------------------
    row: Dict[str, Any] = {
        "provincia": last["provincia"],
        "quota": float(last["quota"]),
        "lat": float(last["lat"]),
        "lng": float(last["lng"]),
        "dist_industrial_km": float(last["dist_industrial_km"]),
        "n_industrial_zones_15km": float(last["n_industrial_zones_15km"]),
    }

    # ---- Weather (forecast) ---------------------------------------------
    weather_keys = [
        "temp_mean", "humidity_mean", "dewpoint_mean", "precip_sum",
        "pressure_mean", "cloud_cover_mean", "wind_speed_mean", "wind_speed_max",
        "radiation_mean", "blh_mean", "blh_min", "fog_hours",
    ]
    for k in weather_keys:
        row[k] = weather_daily.get(k)

    # ---- Pollutants (no forecast available) — carry forward -------------
    for k in ("no2_mean", "o3_mean", "no2_max", "o3_max"):
        row[k] = last.get(k)

    # ---- Temporal --------------------------------------------------------
    ts = pd.Timestamp(target_date)
    mese = int(ts.month)
    dow = int(ts.dayofweek)
    row.update({
        "mese": mese,
        "stagione": MONTH_TO_SEASON[mese],
        "giorno_settimana": dow,
        "is_weekend": bool(dow in (5, 6)),
        "heating_season": int(mese in (10, 11, 12, 1, 2, 3)),
        "mese_sin": float(np.sin(2 * np.pi * mese / 12)),
        "mese_cos": float(np.cos(2 * np.pi * mese / 12)),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
    })

    # ---- PM10 lags / rolling --------------------------------------------
    pm10_context = pm10_context or _compute_pm10_feature_context(lag_data)
    row.update({
        "pm10_lag1": pm10_context["lag1"],
        "pm10_lag2": pm10_context["lag2"],
        "pm10_roll7": pm10_context["roll7"],
        "pm10_roll3": pm10_context["roll3"],
        "pm10_diff": pm10_context["pm10_diff"],
    })

    # ---- Meteo lags / rolling — shift history by one day ----------------
    # Today's predicted-day "_lag1" = yesterday's observed value (last row).
    meteo_vars = ["pressure_mean", "wind_speed_mean", "blh_mean", "temp_mean"]
    for var in meteo_vars:
        row[f"{var}_lag1"] = float(last[var]) if pd.notna(last[var]) else np.nan
        row[f"{var}_lag2"] = float(prev[var]) if pd.notna(prev[var]) else np.nan

    pressure_hist = lag_data["pressure_mean"].astype(float).tail(3)
    wind_hist = lag_data["wind_speed_mean"].astype(float).tail(3)
    row["pressure_roll3"] = float(pressure_hist.mean()) if len(pressure_hist) == 3 else np.nan
    row["wind_speed_roll3"] = float(wind_hist.mean()) if len(wind_hist) == 3 else np.nan

    # ---- Stagnation (from forecast weather) ------------------------------
    row["stagnation_flag"] = _stagnation_flag(
        row["pressure_mean"], row["wind_speed_mean"], row["blh_min"],
    )
    row["stagnation_index"] = _stagnation_index(
        row["wind_speed_mean"], row["blh_min"], row["precip_sum"],
    )

    # ---- Align to the column order the trained pipeline expects ---------
    feature_names = list(_regression_model.feature_names_in_)
    return pd.DataFrame([{col: row.get(col) for col in feature_names}])


# ---------------------------------------------------------------------------
# Top-level prediction orchestration
# ---------------------------------------------------------------------------

def _pm10_to_label(pm10: float) -> str:
    for label, upper in zip(PM10_LABELS, PM10_THRESHOLDS[1:]):
        if pm10 < upper:
            return label
    return PM10_LABELS[-1]


def predict(station_id: str, days: int = 1, conn: Any = None) -> Dict[str, Any]:
    """Predict PM10 + alert class for the next *days* calendar days.

    The ``days=2`` case chains predictions: the day-1 PM10 prediction is used
    as the lag-1 value when building day-2 features.
    """
    if days not in (1, 2):
        raise ValueError("days must be 1 or 2.")

    _require_loaded()

    meta = get_station_metadata(station_id)

    dq = compute_data_quality(station_id)
    quality = dq.get("data_quality", "ok")
    valid_days = int(dq.get("valid_days_last_7", 0))
    if quality == "stale":
        raise HTTPException(
            status_code=422,
            detail=(
                "Dati recenti insufficienti per una previsione affidabile "
                f"(valid_days={valid_days} < 3)."
            ),
        )

    lag_data = get_lag_features(station_id, conn)
    weather_records = fetch_forecast_weather(meta["lat"], meta["lng"], days)
    if len(weather_records) < days:
        raise RuntimeError(
            f"Open-Meteo returned {len(weather_records)} days (expected {days})."
        )

    today = date.today()
    predictions: List[Dict[str, Any]] = []
    rolling_lag = lag_data.copy()

    for day_offset in range(1, days + 1):
        target_date = today + timedelta(days=day_offset)
        weather = weather_records[day_offset - 1]
        pm10_context = _compute_pm10_feature_context(rolling_lag)

        X_future = build_features(
            station_id,
            target_date,
            weather,
            rolling_lag,
            pm10_context=pm10_context,
        )

        pm10_pred = float(_regression_model.predict(X_future)[0])
        alert_index = int(_classification_model.predict(X_future)[0])
        alert_label = _INDEX_TO_LABEL.get(alert_index, _pm10_to_label(pm10_pred))

        predictions.append({
            "date": target_date,
            "pm10_predicted": round(pm10_pred, 2),
            "alert_class": alert_label,
            "alert_index": alert_index,
            "weather_used": {
                "temp_mean": weather.get("temp_mean"),
                "wind_speed_mean": weather.get("wind_speed_mean"),
                "boundary_layer_height_mean": weather.get("blh_mean"),
            },
        })

        # For the chained day+2 prediction: append the predicted row so the
        # next iteration sees updated lag history.
        new_row = rolling_lag.iloc[-1:].copy()
        new_row["data_giorno"] = pd.Timestamp(target_date)
        new_row["pm10"] = pm10_pred
        for k in ("pressure_mean", "wind_speed_mean", "blh_mean", "temp_mean",
                  "blh_min", "precip_sum", "humidity_mean", "dewpoint_mean",
                  "cloud_cover_mean", "wind_speed_max", "radiation_mean",
                  "fog_hours"):
            if k in new_row.columns and k in weather:
                new_row[k] = weather[k]
        rolling_lag = pd.concat([rolling_lag, new_row], ignore_index=True)

    return {
        "station_id": meta["idstazione"],
        "station_name": meta["nomestazione"],
        "predictions": predictions,
        "data_quality": quality,
        "valid_days_last_7": valid_days,
    }
