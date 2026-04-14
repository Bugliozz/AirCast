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
from step_3_eda.config import PM10_LABELS, PM10_THRESHOLDS

log = logging.getLogger(__name__)


def _pm10_to_alert(value: Optional[float]) -> Optional[str]:
    if value is None or pd.isna(value):
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
    return [
        StationOut(
            idstazione=str(row["idstazione"]),
            nomestazione=str(row["nomestazione"]),
            comune=str(row["comune"]),
            lat=float(row["lat"]),
            lon=float(row["lng"]),
        )
        for _, row in meta.iterrows()
    ]


def get_history(station_id: str, from_date: date, to_date: date) -> List[HistoryRecord]:
    """Return daily PM10 history for *station_id* within the date range (inclusive)."""
    fs = _feature_store()

    mask = (
        (fs["idstazione"].astype(str) == str(station_id))
        & (fs["data_giorno"] >= pd.Timestamp(from_date))
        & (fs["data_giorno"] <= pd.Timestamp(to_date))
    )
    subset = fs.loc[mask, ["data_giorno", "pm10"]].sort_values("data_giorno")

    return [
        HistoryRecord(
            date=row["data_giorno"].date(),
            pm10=None if pd.isna(row["pm10"]) else float(row["pm10"]),
            alert_class=_pm10_to_alert(row["pm10"]),
        )
        for _, row in subset.iterrows()
    ]
