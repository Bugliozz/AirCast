"""Entry-point for the GCloud daily refresh job.

Keeps the rolling ``WINDOW_DAYS`` window fresh on GCS so that the API can
compute lag/rolling features at inference time.

Flow
----
1. Acquire a lock (prevents overlap with another daily run).
2. Build the target window: ``[today - END_OFFSET - WINDOW + 1, today - END_OFFSET]``.
3. For each date, check whether both ``{date}_measurements.json`` and
   ``{date}_weather.json`` already exist on GCS.  Fetch only the missing
   ones.  Exit early if all days are complete.
4. Upload new JSONs to ``gs://<bucket>/data/raw/`` alongside the historical
   backfill output (same path convention).

Designed for Cloud Run Jobs + Cloud Scheduler (once per day).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import date, timedelta
from typing import Any

import pandas as pd
from google.cloud import storage as gcs

from api_client import (
    fetch_arpa_measurements,
    fetch_sensor_registry,
    fetch_weather_archive,
)
from config import (
    COORD_ROUND_DP,
    END_OFFSET_DAYS,
    FORCE_REFRESH_LAST_N_DAYS,
    GCS_BUCKET,
    GCS_DATA_PREFIX,
    INTER_CELL_SLEEP_S,
    INTER_DAY_SLEEP_S,
    WINDOW_DAYS,
)
from lock import acquire_lock, release_lock
from rate_limiter import CircuitBreakerOpen, MaxRetriesExceeded, RateLimitedClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# -- GCS ----------------------------------------------------------------------

_gcs_client: gcs.Client | None = None


def _bucket() -> gcs.Bucket:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = gcs.Client()
    return _gcs_client.bucket(GCS_BUCKET)


def blob_exists(path: str) -> bool:
    return _bucket().blob(path).exists()


def upload_json(data: Any, blob_path: str) -> None:
    blob = _bucket().blob(blob_path)
    blob.upload_from_string(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        content_type="application/json",
    )


# -- Window / skip-check ------------------------------------------------------

def target_window() -> list[str]:
    """Return the ISO-date list ``[today-END-WINDOW+1 ... today-END]``."""
    end = date.today() - timedelta(days=END_OFFSET_DAYS)
    start = end - timedelta(days=WINDOW_DAYS - 1)
    return [
        (start + timedelta(days=i)).isoformat()
        for i in range((end - start).days + 1)
    ]


def dates_to_fetch(dates: list[str]) -> list[str]:
    """Return the dates that need to be (re-)fetched.

    A date is included if:
    - either the measurements OR the weather JSON is absent on GCS, OR
    - it falls within the last ``FORCE_REFRESH_LAST_N_DAYS`` of the window
      (tail days are always re-fetched to close ARPA preliminary-data gaps).

    The returned list preserves chronological order with no duplicates.
    """
    force_refresh_set = set(dates[-FORCE_REFRESH_LAST_N_DAYS:]) if FORCE_REFRESH_LAST_N_DAYS > 0 else set()
    to_fetch: list[str] = []
    for d in dates:
        if d in force_refresh_set:
            to_fetch.append(d)
            continue
        meas_ok = blob_exists(f"{GCS_DATA_PREFIX}/{d}_measurements.json")
        wx_ok = blob_exists(f"{GCS_DATA_PREFIX}/{d}_weather.json")
        if not (meas_ok and wx_ok):
            to_fetch.append(d)
    return to_fetch


# -- Grid preparation (same as backfill) --------------------------------------

def prepare_grid(
    sensors_df: pd.DataFrame,
) -> tuple[set[str], pd.DataFrame, list[tuple[str, str]]]:
    target_ids = set(sensors_df["idsensore"].astype(str).tolist())
    stations = (
        sensors_df[["idstazione", "nomestazione", "lat", "lng"]]
        .drop_duplicates(subset="idstazione")
        .dropna(subset=["lat", "lng"])
        .copy()
    )
    stations["lat_r"] = stations["lat"].astype(float).round(COORD_ROUND_DP).astype(str)
    stations["lng_r"] = stations["lng"].astype(float).round(COORD_ROUND_DP).astype(str)
    grid_cells = list(
        stations[["lat_r", "lng_r"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    return target_ids, stations, grid_cells


# -- Single-day collection ----------------------------------------------------

def collect_day(
    client: RateLimitedClient,
    date_str: str,
    target_ids: set[str],
    stations: pd.DataFrame,
    grid_cells: list[tuple[str, str]],
) -> dict[str, int]:
    # ARPA measurements — always overwrite so force-refresh closes ARPA gaps.
    meas_path = f"{GCS_DATA_PREFIX}/{date_str}_measurements.json"
    if blob_exists(meas_path):
        log.info("Overwriting %s", meas_path)
    all_meas = fetch_arpa_measurements(client, date_str)
    measurements = [m for m in all_meas if m["idsensore"] in target_ids]
    upload_json(measurements, meas_path)
    meas_count: int = len(measurements)

    # Weather per grid cell — always overwrite.
    wx_path = f"{GCS_DATA_PREFIX}/{date_str}_weather.json"
    if blob_exists(wx_path):
        log.info("Overwriting %s", wx_path)

    weather_cache: dict[tuple[str, str], dict[str, Any]] = {}
    failed_cells = 0
    for idx, (lat_r, lng_r) in enumerate(grid_cells):
        try:
            weather_cache[(lat_r, lng_r)] = fetch_weather_archive(
                client, lat_r, lng_r, date_str,
            )
        except MaxRetriesExceeded as exc:
            log.warning("[%s] weather cell (%s,%s) failed: %s",
                        date_str, lat_r, lng_r, exc)
            failed_cells += 1
        if idx < len(grid_cells) - 1:
            time.sleep(INTER_CELL_SLEEP_S)

    weather_records: list[dict[str, Any]] = []
    for _, row in stations.iterrows():
        hourly = weather_cache.get((row["lat_r"], row["lng_r"]), {}).get("hourly", {})
        if hourly:
            weather_records.append({
                "idstazione": row["idstazione"],
                "nomestazione": row["nomestazione"],
                "lat": row["lat"],
                "lng": row["lng"],
                "hourly": hourly,
            })
    upload_json(weather_records, wx_path)

    return {
        "measurements": meas_count,
        "weather_stations": len(weather_records),
        "failed_cells": failed_cells,
    }


# -- Main loop ----------------------------------------------------------------

def _run() -> None:
    window = target_window()
    log.info("Target window: %s ... %s (%d days)",
             window[0], window[-1], len(window))

    to_fetch = dates_to_fetch(window)
    if not to_fetch:
        log.info("All %d window days already present on GCS — nothing to do.",
                 len(window))
        return

    log.info("Fetching %d/%d days (force-refresh last N=%d): %s",
             len(to_fetch), len(window), FORCE_REFRESH_LAST_N_DAYS, to_fetch)

    client = RateLimitedClient()
    log.info("Fetching sensor registry...")
    sensors_df = fetch_sensor_registry(client)
    target_ids, stations, grid_cells = prepare_grid(sensors_df)
    log.info("Grid: %d cells from %d stations, %d target sensors.",
             len(grid_cells), len(stations), len(target_ids))

    for i, d in enumerate(to_fetch, start=1):
        log.info("[%d/%d] Collecting %s ...", i, len(to_fetch), d)
        try:
            stats = collect_day(client, d, target_ids, stations, grid_cells)
            log.info("[%s] OK  meas=%s  weather=%s  failed_cells=%d",
                     d, stats["measurements"], stats["weather_stations"],
                     stats["failed_cells"])
        except CircuitBreakerOpen:
            log.warning("Circuit breaker open — exiting, next scheduled run resumes.")
            sys.exit(0)
        except Exception as exc:
            log.error("[%s] FAILED: %s", d, exc)

        if i < len(to_fetch):
            time.sleep(INTER_DAY_SLEEP_S)

    log.info("Daily refresh done.  HTTP stats: %s", client.stats)


def main() -> None:
    log.info("=" * 60)
    log.info("Daily Refresh  window=%d  end_offset=%d  force_refresh_last_n=%d",
             WINDOW_DAYS, END_OFFSET_DAYS, FORCE_REFRESH_LAST_N_DAYS)
    log.info("Bucket: gs://%s/%s", GCS_BUCKET, GCS_DATA_PREFIX)
    log.info("=" * 60)

    if not acquire_lock():
        log.error("Could not acquire lock — exiting.")
        sys.exit(0)

    try:
        _run()
    finally:
        release_lock()


if __name__ == "__main__":
    main()
