"""Fetch the last 7 days of observations from GCS for inference.

Serving-only data path: the rolling 7-day window lives **only** on GCS and
is read ephemerally at prediction time.  It never touches MySQL nor the
training parquet, so sporadic API calls cannot create gaps in the
historical dataset used for training.

The module downloads the same ``{date}_measurements.json`` and
``{date}_weather.json`` blobs produced by the daily Cloud Run Job, then
aggregates hourly rows into daily per-station values using the same
rules as ``step_3_eda/db.py``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from google.cloud import storage as gcs

from api.services import live_arpa

log = logging.getLogger(__name__)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "exam-project-backfill")
GCS_DATA_PREFIX = "data/raw"
WINDOW_DAYS = int(os.environ.get("WINDOW_DAYS", "7"))
END_OFFSET_DAYS = int(os.environ.get("END_OFFSET_DAYS", "1"))
CACHE_TTL_SECONDS = int(os.environ.get("RECENT_CACHE_TTL", "900"))  # 15 min

_ROOT = Path(__file__).resolve().parent.parent.parent
_SENSORS_REGISTRY_PATH = _ROOT / "data" / "raw" / "sensors_registry.json"

_gcs_client: Optional[gcs.Client] = None
_sensors_map: Optional[pd.DataFrame] = None
_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
_MEASUREMENTS_BLOB_RE = re.compile(r"(?P<day>\d{4}-\d{2}-\d{2})_measurements\.json$")


def _client() -> gcs.Client:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = gcs.Client()
    return _gcs_client


def _load_sensors_map() -> pd.DataFrame:
    """Load idsensore → (idstazione, tiposensore) mapping from the cached registry."""
    global _sensors_map
    if _sensors_map is not None:
        return _sensors_map
    if not _SENSORS_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Sensor registry missing: {_SENSORS_REGISTRY_PATH}. "
            "Run step_2_ingestion/ingest.py once to populate it."
        )
    records = json.loads(_SENSORS_REGISTRY_PATH.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)[["idsensore", "idstazione", "nometiposensore"]]
    df = df.rename(columns={"nometiposensore": "tiposensore"})
    df["idsensore"] = df["idsensore"].astype(str)
    df["idstazione"] = df["idstazione"].astype(str)
    _sensors_map = df
    return df


def _target_window_by_date() -> List[str]:
    end = date.today() - timedelta(days=END_OFFSET_DAYS)
    start = end - timedelta(days=WINDOW_DAYS - 1)
    return [(start + timedelta(days=i)).isoformat() for i in range(WINDOW_DAYS)]


def _target_window_from_bucket() -> List[str]:
    """Resolve the rolling window from the latest measurement blobs on GCS.

    Falls back to the local date-based window when GCS listing is unavailable,
    for example in offline development or when credentials are missing.
    """
    try:
        blobs = _client().list_blobs(GCS_BUCKET, prefix=f"{GCS_DATA_PREFIX}/")
        available_dates: List[date] = []
        for blob in blobs:
            blob_name = blob.name.rsplit("/", maxsplit=1)[-1]
            match = _MEASUREMENTS_BLOB_RE.fullmatch(blob_name)
            if match is None:
                continue
            try:
                available_dates.append(date.fromisoformat(match.group("day")))
            except ValueError:
                log.warning("Skipping malformed recent-data blob name: %s", blob.name)

        if available_dates:
            window = [d.isoformat() for d in sorted(set(available_dates))[-WINDOW_DAYS:]]
            log.info(
                "Recent window resolved from bucket: %s..%s (%d day blobs).",
                window[0],
                window[-1],
                len(window),
            )
            return window

        log.warning(
            "No measurement blobs found under gs://%s/%s; falling back to date-based window.",
            GCS_BUCKET,
            GCS_DATA_PREFIX,
        )
    except Exception as exc:  # noqa: BLE001 - offline dev / missing creds fallback
        log.warning("Recent window listing failed, using date-based fallback: %s", exc)

    return _target_window_by_date()


def _download_json(blob_name: str) -> Optional[Any]:
    blob = _client().bucket(GCS_BUCKET).blob(blob_name)
    if not blob.exists():
        log.warning("GCS blob missing: %s", blob_name)
        return None
    return json.loads(blob.download_as_bytes())


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_POLLUTANT_MAP = {
    "azoto": "no2",
    "ozono": "o3",
    "carbonio": "co",
    "pm2": "pm25",
}


def _classify_sensor(tipo: str) -> Optional[str]:
    t = (tipo or "").lower()
    if "pm10" in t:
        return "pm10"
    for keyword, short in _POLLUTANT_MAP.items():
        if keyword in t:
            return short
    return None


def _aggregate_measurements_daily(
    measurements: List[dict], sensors: pd.DataFrame
) -> pd.DataFrame:
    """Daily per-station aggregates for PM10 + pollutants (NO2/O3/CO/PM2.5)."""
    empty_cols = ["idstazione", "data_giorno", "pm10",
                  "no2_mean", "no2_max", "o3_mean", "o3_max",
                  "co_mean", "co_max", "pm25_mean", "pm25_max"]
    if not measurements:
        return pd.DataFrame(columns=empty_cols)

    df = pd.DataFrame(measurements)
    df["idsensore"] = df["idsensore"].astype(str)
    df = df.merge(sensors, on="idsensore", how="inner")
    df["kind"] = df["tiposensore"].map(_classify_sensor)
    df = df.dropna(subset=["kind"])
    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    df["valore"] = pd.to_numeric(df["valore"], errors="coerce")
    invalid = df["valore"] <= 0
    if invalid.any():
        df.loc[invalid, "valore"] = pd.NA
    df = df.dropna(subset=["valore"])
    df["data_giorno"] = pd.to_datetime(df["data"]).dt.normalize()

    # PM10 → single 'pm10' column (daily mean)
    pm10 = (
        df[df["kind"] == "pm10"]
        .groupby(["idstazione", "data_giorno"], as_index=False)["valore"]
        .mean()
        .rename(columns={"valore": "pm10"})
    )

    # Other pollutants → *_mean / *_max
    others = df[df["kind"] != "pm10"]
    if not others.empty:
        agg = (
            others.groupby(["idstazione", "data_giorno", "kind"], as_index=False)["valore"]
            .agg(mean="mean", max="max")
        )
        mean_wide = agg.pivot_table(
            index=["idstazione", "data_giorno"], columns="kind", values="mean"
        ).rename(columns=lambda c: f"{c}_mean")
        max_wide = agg.pivot_table(
            index=["idstazione", "data_giorno"], columns="kind", values="max"
        ).rename(columns=lambda c: f"{c}_max")
        pollutants = mean_wide.join(max_wide).reset_index()
        pollutants.columns.name = None
    else:
        pollutants = pd.DataFrame(columns=["idstazione", "data_giorno"])

    if pm10.empty and pollutants.empty:
        return pd.DataFrame(columns=empty_cols)
    if pm10.empty:
        return pollutants
    if pollutants.empty:
        return pm10
    return pm10.merge(pollutants, on=["idstazione", "data_giorno"], how="outer")


def _aggregate_weather_daily(stations: List[dict]) -> pd.DataFrame:
    """Daily weather aggregates per station — mirrors load_weather_daily_agg."""
    rows: List[dict] = []
    for station in stations or []:
        idstazione = str(station["idstazione"])
        hourly = station.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            continue
        hdf = pd.DataFrame({"time": pd.to_datetime(times)})
        for var in (
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "precipitation", "surface_pressure", "cloud_cover",
            "wind_speed_10m", "wind_direction_10m", "visibility",
            "shortwave_radiation", "boundary_layer_height",
        ):
            hdf[var] = hourly.get(var, [None] * len(times))
        hdf["data_giorno"] = hdf["time"].dt.normalize()

        for day, g in hdf.groupby("data_giorno"):
            temp = pd.to_numeric(g["temperature_2m"], errors="coerce")
            dew = pd.to_numeric(g["dew_point_2m"], errors="coerce")
            rows.append({
                "idstazione": idstazione,
                "data_giorno": day,
                "temp_mean": temp.mean(),
                "temp_min": temp.min(),
                "temp_max": temp.max(),
                "humidity_mean": pd.to_numeric(g["relative_humidity_2m"], errors="coerce").mean(),
                "dewpoint_mean": dew.mean(),
                "precip_sum": pd.to_numeric(g["precipitation"], errors="coerce").sum(),
                "pressure_mean": pd.to_numeric(g["surface_pressure"], errors="coerce").mean(),
                "cloud_cover_mean": pd.to_numeric(g["cloud_cover"], errors="coerce").mean(),
                "wind_speed_mean": pd.to_numeric(g["wind_speed_10m"], errors="coerce").mean(),
                "wind_speed_max": pd.to_numeric(g["wind_speed_10m"], errors="coerce").max(),
                "visibility_mean": pd.to_numeric(g["visibility"], errors="coerce").mean(),
                "radiation_mean": pd.to_numeric(g["shortwave_radiation"], errors="coerce").mean(),
                "blh_mean": pd.to_numeric(g["boundary_layer_height"], errors="coerce").mean(),
                "blh_min": pd.to_numeric(g["boundary_layer_height"], errors="coerce").min(),
                "fog_hours": float(((temp - dew) < 2).sum()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_recent_window() -> pd.DataFrame:
    """Download + aggregate the latest recent-data window for all stations.

    Bucket blobs are the primary source.  Any day that is missing PM10 data
    from the bucket is gap-filled via the Socrata live API (validated
    ``stato='VA'`` records only).

    Cached for ``RECENT_CACHE_TTL`` seconds so multiple predictions in the
    same time window don't re-download the JSON files.
    """
    window = _target_window_from_bucket()
    cache_key = ",".join(window)
    hit = _cache.get(cache_key)
    if hit and (time.time() - hit[0]) < CACHE_TTL_SECONDS:
        return hit[1]

    sensors = _load_sensors_map()
    target_sensor_ids: set[str] = set(sensors["idsensore"].astype(str).tolist())

    pm10_frames: List[pd.DataFrame] = []
    weather_frames: List[pd.DataFrame] = []
    bucket_dates_with_pm10: set[str] = set()

    for d in window:
        meas = _download_json(f"{GCS_DATA_PREFIX}/{d}_measurements.json")
        if meas:
            day_df = _aggregate_measurements_daily(meas, sensors)
            pm10_frames.append(day_df)
            if "pm10" in day_df.columns and day_df["pm10"].notna().any():
                bucket_dates_with_pm10.add(d)
        wx = _download_json(f"{GCS_DATA_PREFIX}/{d}_weather.json")
        if wx:
            weather_frames.append(_aggregate_weather_daily(wx))

    # Gap-fill bucket misses with Socrata validated data
    gap_dates = set(window) - bucket_dates_with_pm10
    for d in sorted(gap_dates):
        live_records = live_arpa.fetch_socrata_day(d, target_sensor_ids)
        if live_records:
            day_df = _aggregate_measurements_daily(live_records, sensors)
            if "pm10" in day_df.columns and day_df["pm10"].notna().any():
                pm10_frames.append(day_df)
                log.info("Gap-filled %s with Socrata validated data.", d)

    pm10 = (
        pd.concat(pm10_frames, ignore_index=True)
        if pm10_frames
        else pd.DataFrame(columns=["idstazione", "data_giorno"])
    )
    weather = (
        pd.concat(weather_frames, ignore_index=True)
        if weather_frames
        else pd.DataFrame(columns=["idstazione", "data_giorno"])
    )

    if pm10.empty:
        merged = weather
    elif weather.empty:
        merged = pm10
    else:
        merged = pm10.merge(weather, on=["idstazione", "data_giorno"], how="outer")

    merged = merged.sort_values(["idstazione", "data_giorno"]).reset_index(drop=True)
    log.info(
        "Recent window aggregated: %d rows across %d stations (window=%s..%s).",
        len(merged),
        merged["idstazione"].nunique() if not merged.empty else 0,
        window[0],
        window[-1],
    )
    _cache[cache_key] = (time.time(), merged)
    return merged


def fetch_recent_for_station(station_id: str) -> pd.DataFrame:
    """Return the last 7 daily rows for a single station (ascending by date)."""
    window = fetch_recent_window()
    rows = window[window["idstazione"].astype(str) == str(station_id)]
    return rows.sort_values("data_giorno").reset_index(drop=True)


def compute_data_quality(station_id: str) -> Dict[str, Any]:
    """Assess freshness/completeness of the 7-day window for a station.

    A day counts as *valid* when a finite PM10 value is present.
    Thresholds: ok ≥ 6, partial ≥ 3, stale < 3.
    """
    rows = fetch_recent_for_station(station_id)

    if rows.empty or "pm10" not in rows.columns:
        return {
            "valid_days_last_7": 0,
            "data_quality": "stale",
            "last_valid_date": None,
        }

    valid = rows.dropna(subset=["pm10"])
    valid_days = int(valid["data_giorno"].dt.normalize().nunique())
    last_valid = (
        valid["data_giorno"].max().date() if not valid.empty else None
    )

    if valid_days >= 6:
        quality = "ok"
    elif valid_days >= 3:
        quality = "partial"
    else:
        quality = "stale"

    return {
        "valid_days_last_7": valid_days,
        "data_quality": quality,
        "last_valid_date": last_valid,
    }


def prefetch() -> None:
    """Warm the recent-data cache without failing the API startup."""
    try:
        fetch_recent_window()
    except Exception as exc:  # noqa: BLE001 - best effort only
        log.warning("Recent-data prefetch skipped: %s", exc)
