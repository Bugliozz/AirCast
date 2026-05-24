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

import asyncio
import json
import logging
import os
import re
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from google.cloud import storage as gcs

from api.config import GCS_BUCKET, SENSORS_REGISTRY_PATH
from api.services import live_arpa

log = logging.getLogger(__name__)

GCS_DATA_PREFIX = "data/raw"
WINDOW_DAYS = int(os.environ.get("WINDOW_DAYS", "7"))
END_OFFSET_DAYS = int(os.environ.get("END_OFFSET_DAYS", "1"))
# GCS blobs and Socrata validated records are immutable once written, so the
# historical tier only needs to be refreshed when the rolling window advances.
HISTORICAL_CACHE_TTL = int(os.environ.get("HISTORICAL_CACHE_TTL", "86400"))  # 24h
# The ARPA NRT feed evolves hour-by-hour throughout the day.
NRT_CACHE_TTL = int(os.environ.get("NRT_CACHE_TTL", "3600"))  # 1h
# Background refresh cadence — defaults to the NRT tier TTL so today's PM10
# never goes stale without a proactive refetch.
REFRESH_INTERVAL_SECONDS = int(os.environ.get("REFRESH_INTERVAL_SECONDS", str(NRT_CACHE_TTL)))

_SENSORS_REGISTRY_PATH = SENSORS_REGISTRY_PATH

_gcs_client: Optional[gcs.Client] = None
_sensors_map: Optional[pd.DataFrame] = None
_historical_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
_nrt_cache: Optional[Tuple[float, pd.DataFrame]] = None
_window_cache: Optional[Tuple[float, List[str]]] = None
_refresh_task: Optional["asyncio.Task[None]"] = None
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

def _resolve_window() -> List[str]:
    """Resolve the rolling window; cached alongside the historical tier (24h).

    Listing the bucket on every serving request is wasteful when the blob set
    only changes once per day (after the daily 3am upload).
    """
    global _window_cache
    if _window_cache and (time.time() - _window_cache[0]) < HISTORICAL_CACHE_TTL:
        return _window_cache[1]
    window = _target_window_from_bucket()
    _window_cache = (time.time(), window)
    return window


def _compute_historical(window: List[str], sensors: pd.DataFrame) -> pd.DataFrame:
    """GCS blobs + Socrata gap-fill for the window — no today/NRT piece.

    Both sources are immutable once written, so a single refetch per day is
    sufficient.
    """
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
    if not pm10.empty:
        # Bucket measurements.json may cover a day with only pollutants (no PM10),
        # and Socrata gap-fill then provides the PM10-only row for the same day.
        # Collapse both into a single (station, day) row, keeping the first
        # non-null value per column so pm10 and co-pollutants merge cleanly.
        pm10 = (
            pm10.sort_values(["idstazione", "data_giorno"])
            .groupby(["idstazione", "data_giorno"], as_index=False)
            .first()
        )
    weather = (
        pd.concat(weather_frames, ignore_index=True)
        if weather_frames
        else pd.DataFrame(columns=["idstazione", "data_giorno"])
    )
    if pm10.empty:
        return weather
    if weather.empty:
        return pm10
    return pm10.merge(weather, on=["idstazione", "data_giorno"], how="outer")


def _compute_nrt_today(sensors: pd.DataFrame) -> pd.DataFrame:
    """Today (T₀) PM10 from the provisional ARPA NRT feed.

    Only ~21% of PM10 stations are in the NRT feed; stations without NRT
    coverage keep their historical T−1 lag via the 3-level fallback in the
    predictor.
    """
    target_sensor_ids: set[str] = set(sensors["idsensore"].astype(str).tolist())
    nrt_records = live_arpa.fetch_nrt_today(target_sensor_ids)
    if not nrt_records:
        return pd.DataFrame(columns=["idstazione", "data_giorno"])
    today_df = _aggregate_measurements_daily(nrt_records, sensors)
    if "pm10" not in today_df.columns or not today_df["pm10"].notna().any():
        return pd.DataFrame(columns=["idstazione", "data_giorno"])
    log.info(
        "NRT refreshed: today's partial PM10 across %d stations from %d records.",
        today_df["idstazione"].nunique(),
        len(nrt_records),
    )
    return today_df


def _get_historical(window: List[str], sensors: pd.DataFrame) -> pd.DataFrame:
    cache_key = ",".join(window)
    hit = _historical_cache.get(cache_key)
    if hit and (time.time() - hit[0]) < HISTORICAL_CACHE_TTL:
        return hit[1]
    df = _compute_historical(window, sensors)
    _historical_cache[cache_key] = (time.time(), df)
    for stale_key in list(_historical_cache.keys()):
        if stale_key != cache_key:
            del _historical_cache[stale_key]
    return df


def _get_nrt(sensors: pd.DataFrame) -> pd.DataFrame:
    global _nrt_cache
    if _nrt_cache and (time.time() - _nrt_cache[0]) < NRT_CACHE_TTL:
        return _nrt_cache[1]
    df = _compute_nrt_today(sensors)
    _nrt_cache = (time.time(), df)
    return df


def fetch_recent_window() -> pd.DataFrame:
    """Return the latest 7-day + today window for all stations.

    Two-tier cache:

    * **Historical** (GCS blobs + Socrata gap-fill): cached for
      ``HISTORICAL_CACHE_TTL`` seconds (24h by default).  Both sources are
      immutable once written, so a single refetch per day suffices.
    * **NRT today**: cached for ``NRT_CACHE_TTL`` seconds (1h).  This is the
      only piece that actually evolves within the day.
    """
    window = _resolve_window()
    sensors = _load_sensors_map()
    historical = _get_historical(window, sensors)
    nrt_today = _get_nrt(sensors)

    frames = [f for f in (historical, nrt_today) if not f.empty]
    if not frames:
        empty_cols = ["idstazione", "data_giorno"]
        log.info("Recent window empty (window=%s..%s).", window[0], window[-1])
        return pd.DataFrame(columns=empty_cols)

    merged = pd.concat(frames, ignore_index=True)
    merged = (
        merged.sort_values(["idstazione", "data_giorno"])
        .groupby(["idstazione", "data_giorno"], as_index=False)
        .first()
        .reset_index(drop=True)
    )
    return merged


def fetch_recent_for_station(station_id: str) -> pd.DataFrame:
    """Return the last 7 daily rows for a single station (ascending by date)."""
    window = fetch_recent_window()
    rows = window[window["idstazione"].astype(str) == str(station_id)]
    return rows.sort_values("data_giorno").reset_index(drop=True)


def get_nrt_station_ids() -> set[str]:
    """Return the set of station ids that have a non-null PM10 row for today.

    Today's values come exclusively from the ARPA NRT feed (``ykhg-b8rs``),
    which covers only a subset of PM10 stations — so presence of a
    ``data_giorno == today`` row with non-null PM10 is a reliable signal that
    the station's lag-1 feature reflects T0 (live) rather than T-1.
    """
    try:
        window = fetch_recent_window()
    except Exception as exc:  # noqa: BLE001 - NRT flag is best-effort
        log.warning("NRT station set unavailable: %s", exc)
        return set()

    if window.empty or "pm10" not in window.columns:
        return set()

    today = pd.Timestamp(date.today())
    today_rows = window[
        (window["data_giorno"].dt.normalize() == today)
        & window["pm10"].notna()
    ]
    return set(today_rows["idstazione"].astype(str).unique())


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


def refresh() -> None:
    """Proactive refresh of both cache tiers (stale-while-revalidate).

    Always recomputes the NRT tier (hourly cadence).  Recomputes the historical
    tier only when the rolling window has advanced to a new day or the 24h TTL
    has elapsed.  On failure the existing cached DataFrames are preserved so
    serving never sees an empty window.
    """
    global _nrt_cache, _window_cache
    try:
        sensors = _load_sensors_map()
    except Exception as exc:  # noqa: BLE001 - registry load failure
        log.warning("Refresh aborted: sensors registry unavailable: %s", exc)
        return

    try:
        new_nrt = _compute_nrt_today(sensors)
        _nrt_cache = (time.time(), new_nrt)
    except Exception as exc:  # noqa: BLE001 - network/NRT failure
        log.warning("NRT refresh failed, keeping stale cache: %s", exc)

    try:
        window = _target_window_from_bucket()
        _window_cache = (time.time(), window)
        cache_key = ",".join(window)
        hit = _historical_cache.get(cache_key)
        if not hit or (time.time() - hit[0]) >= HISTORICAL_CACHE_TTL:
            new_hist = _compute_historical(window, sensors)
            _historical_cache[cache_key] = (time.time(), new_hist)
            for stale_key in list(_historical_cache.keys()):
                if stale_key != cache_key:
                    del _historical_cache[stale_key]
            log.info(
                "Historical tier refreshed (window=%s..%s).", window[0], window[-1]
            )
    except Exception as exc:  # noqa: BLE001 - GCS/Socrata failure
        log.warning("Historical refresh failed, keeping stale cache: %s", exc)


async def _refresh_loop(interval_s: int) -> None:
    """Background coroutine that calls :func:`refresh` at regular intervals."""
    while True:
        try:
            await asyncio.sleep(interval_s)
            await asyncio.to_thread(refresh)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            log.warning("Background refresh loop error: %s", exc)


def start_background_refresh(interval_s: Optional[int] = None) -> None:
    """Start the proactive background refresh task (idempotent)."""
    global _refresh_task
    if _refresh_task is not None and not _refresh_task.done():
        return
    interval = interval_s if interval_s is not None else REFRESH_INTERVAL_SECONDS
    _refresh_task = asyncio.create_task(_refresh_loop(interval))
    log.info("Background refresh started: interval=%ds.", interval)


async def stop_background_refresh() -> None:
    """Cancel the background refresh task, if running."""
    global _refresh_task
    if _refresh_task is None:
        return
    _refresh_task.cancel()
    try:
        await _refresh_task
    except asyncio.CancelledError:
        pass
    _refresh_task = None


def prefetch() -> None:
    """Warm the recent-data cache without failing the API startup."""
    try:
        fetch_recent_window()
    except Exception as exc:  # noqa: BLE001 - best effort only
        log.warning("Recent-data prefetch skipped: %s", exc)
