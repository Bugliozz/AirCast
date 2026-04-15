"""In-memory station registry + historical PM10 lookup.

Reads from the static ``daily_dataset_clean.parquet`` artifact already loaded
by :mod:`api.services.predictor` as the station registry.  This keeps the
container runtime completely MySQL-free: the training pipeline remains the
sole writer of the parquet, while the serving path only reads it.

Reusing ``predictor._feature_store`` avoids duplicating the DataFrame in
memory.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List, Optional

import pandas as pd

from api.schemas import HistoryRecord, StationOut
from api.services import predictor
from api.services.recent_data import (
    compute_data_quality,
    fetch_recent_window,
)
from step_3_eda.config import PM10_LABELS, PM10_THRESHOLDS

log = logging.getLogger(__name__)


def _sanitize_pm10(value: Optional[float]) -> Optional[float]:
    """Return a PM10 value only when it is strictly positive."""
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if value > 0 else None


def _pm10_to_alert(value: Optional[float]) -> Optional[str]:
    value = _sanitize_pm10(value)
    if value is None:
        return None
    for label, upper in zip(PM10_LABELS, PM10_THRESHOLDS[1:]):
        if value < upper:
            return label
    return PM10_LABELS[-1]


def _feature_store() -> pd.DataFrame:
    """Return the shared in-memory feature store loaded at startup."""
    fs = predictor._feature_store
    if fs is None:
        raise predictor.ModelNotFoundError(
            "Feature store not initialised — call predictor.load_models() first."
        )
    return fs


def get_stations() -> List[StationOut]:
    """Return all monitoring stations with coordinates and municipality."""
    fs = _feature_store()
    cols = ["idstazione", "nomestazione", "comune", "lat", "lng"]
    meta = (
        fs[cols]
        .dropna(subset=["lat", "lng"])
        .groupby("idstazione", as_index=False)
        .first()
        .sort_values("idstazione")
    )
    stations: List[StationOut] = []
    for _, row in meta.iterrows():
        idstazione = str(row["idstazione"])
        try:
            dq = compute_data_quality(idstazione)
            quality = dq.get("data_quality", "ok")
            valid_days = int(dq.get("valid_days_last_7", 7))
        except Exception as exc:  # noqa: BLE001 - quality is best-effort
            log.warning("Data quality check failed for %s: %s", idstazione, exc)
            quality, valid_days = "ok", 7
        stations.append(
            StationOut(
                idstazione=idstazione,
                nomestazione=str(row["nomestazione"]),
                comune=str(row["comune"]),
                lat=float(row["lat"]),
                lon=float(row["lng"]),
                data_quality=quality,
                valid_days_last_7=valid_days,
            )
        )
    return stations


def _recent_window_safe() -> pd.DataFrame:
    """Fetch the rolling GCS window, swallowing failures (offline / no creds)."""
    try:
        return fetch_recent_window()
    except Exception as exc:  # noqa: BLE001 - recent data is best-effort
        log.warning("Recent GCS window unavailable: %s", exc)
        return pd.DataFrame(columns=["idstazione", "data_giorno", "pm10"])


def get_latest_date() -> Optional[date]:
    """Return the most recent ``data_giorno`` across parquet + GCS window."""
    fs = _feature_store()
    candidates: List[pd.Timestamp] = []
    if not fs.empty and "data_giorno" in fs.columns:
        ts = fs["data_giorno"].max()
        if not pd.isna(ts):
            candidates.append(pd.Timestamp(ts))

    recent = _recent_window_safe()
    if not recent.empty and "data_giorno" in recent.columns:
        ts = recent["data_giorno"].max()
        if not pd.isna(ts):
            candidates.append(pd.Timestamp(ts))

    return max(candidates).date() if candidates else None


def get_history(station_id: str, from_date: date, to_date: date) -> List[HistoryRecord]:
    """Return daily PM10 history for *station_id* within the date range (inclusive).

    Combines the static training parquet with the rolling GCS window so the
    trailing days (published after the last parquet rebuild) also appear.
    """
    fs = _feature_store()

    mask = (
        (fs["idstazione"].astype(str) == str(station_id))
        & (fs["data_giorno"] >= pd.Timestamp(from_date))
        & (fs["data_giorno"] <= pd.Timestamp(to_date))
    )
    subset = fs.loc[mask, ["data_giorno", "pm10"]]

    parquet_max = fs.loc[fs["idstazione"].astype(str) == str(station_id), "data_giorno"].max()
    recent = _recent_window_safe()
    if not recent.empty and "pm10" in recent.columns:
        rmask = (
            (recent["idstazione"].astype(str) == str(station_id))
            & (recent["data_giorno"] >= pd.Timestamp(from_date))
            & (recent["data_giorno"] <= pd.Timestamp(to_date))
        )
        if not pd.isna(parquet_max):
            rmask &= recent["data_giorno"] > parquet_max
        extra = recent.loc[rmask, ["data_giorno", "pm10"]]
        if not extra.empty:
            subset = pd.concat([subset, extra], ignore_index=True)

    subset = subset.sort_values("data_giorno")

    return [
        HistoryRecord(
            date=row["data_giorno"].date(),
            pm10=_sanitize_pm10(row["pm10"]),
            alert_class=_pm10_to_alert(row["pm10"]),
        )
        for _, row in subset.iterrows()
    ]
