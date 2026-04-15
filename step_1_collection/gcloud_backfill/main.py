"""Entry-point for the GCloud historical backfill.

Orchestrates:
  1. Acquire a distributed lock (prevents overlapping runs).
  2. Load checkpoint from GCS.
  3. Fetch sensor registry (once).
  4. For each remaining date: collect ARPA + weather, upload to GCS.
  5. On rate-limit / transient error: the RateLimitedClient sleeps and
     retries transparently.  On circuit-breaker trip: save checkpoint & exit.
  6. On SIGTERM / SIGINT (Cloud Run shutdown): finish current day, save, exit.

The script is designed to be run as a **Cloud Run Job** triggered hourly by
Cloud Scheduler.  Each run picks up from the last checkpoint and processes
up to DAYS_PER_BATCH days.  On rate-limit / circuit-break it exits cleanly
so the next hourly run resumes automatically.
"""
from __future__ import annotations

import json
import logging
import signal
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
from checkpoint import (
    Checkpoint,
    acquire_lock,
    load_checkpoint,
    release_lock,
    save_checkpoint,
)
from config import (
    BACKFILL_END,
    BACKFILL_START,
    CHECKPOINT_EVERY_N_DAYS,
    COORD_ROUND_DP,
    DAYS_PER_BATCH,
    GCS_BUCKET,
    GCS_DATA_PREFIX,
    HEAL_WINDOW_END_OFFSET,
    HEAL_WINDOW_START_OFFSET,
    INTER_CELL_SLEEP_S,
    INTER_DAY_SLEEP_S,
    OVERWRITE_EXISTING,
)
from rate_limiter import CircuitBreakerOpen, MaxRetriesExceeded, RateLimitedClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# -- Graceful shutdown --------------------------------------------------------

_shutdown_requested: bool = False


def _on_signal(signum: int, _frame: Any) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    log.warning("Signal %d received — finishing current day then exiting.", signum)


signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)


# -- Date helpers -------------------------------------------------------------

def date_range(start: str, end: str) -> list[str]:
    """Inclusive date range as ISO strings."""
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    return [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


# -- GCS upload ---------------------------------------------------------------

_gcs_client: gcs.Client | None = None


def _get_gcs_client() -> gcs.Client:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = gcs.Client()
    return _gcs_client


def upload_json(data: Any, blob_path: str) -> None:
    """Serialize *data* as JSON and upload to ``gs://<bucket>/<blob_path>``.

    In heal mode (``OVERWRITE_EXISTING=true``) an informational log line is
    emitted whenever an existing blob is about to be overwritten, so the
    post-deploy verification can grep for it.  In normal mode GCS overwrites
    silently (same behaviour as before).
    """
    bucket = _get_gcs_client().bucket(GCS_BUCKET)
    blob = bucket.blob(blob_path)
    if OVERWRITE_EXISTING and blob.exists():
        log.info("Overwriting existing blob gs://%s/%s", GCS_BUCKET, blob_path)
    blob.upload_from_string(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        content_type="application/json",
    )


# -- Grid-cell preparation (run once) ----------------------------------------

def prepare_grid(
    sensors_df: pd.DataFrame,
) -> tuple[set[str], pd.DataFrame, list[tuple[str, str]]]:
    """From the sensor registry, derive:

    * *target_ids*  – set of sensor id strings
    * *stations*    – DataFrame with rounded coords
    * *grid_cells*  – deduplicated (lat_r, lng_r) tuples
    """
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
    """Collect ARPA measurements + weather for *date_str* and upload to GCS.

    Returns a small stats dict ``{measurements, weather_stations, failed_cells}``.
    """
    # -- ARPA measurements ----------------------------------------------------
    all_meas = fetch_arpa_measurements(client, date_str)
    measurements = [m for m in all_meas if m["idsensore"] in target_ids]
    upload_json(measurements, f"{GCS_DATA_PREFIX}/{date_str}_measurements.json")

    # -- Weather per grid cell ------------------------------------------------
    weather_cache: dict[tuple[str, str], dict[str, Any]] = {}
    failed_cells = 0

    for idx, (lat_r, lng_r) in enumerate(grid_cells):
        try:
            weather_cache[(lat_r, lng_r)] = fetch_weather_archive(
                client, lat_r, lng_r, date_str,
            )
        except MaxRetriesExceeded as exc:
            log.warning("[%s] weather cell (%s,%s) failed: %s", date_str, lat_r, lng_r, exc)
            failed_cells += 1
        if idx < len(grid_cells) - 1:
            time.sleep(INTER_CELL_SLEEP_S)

    # -- Map stations to their grid-cell weather ------------------------------
    weather_records: list[dict[str, Any]] = []
    for _, row in stations.iterrows():
        key = (row["lat_r"], row["lng_r"])
        hourly = weather_cache.get(key, {}).get("hourly", {})
        if hourly:
            weather_records.append({
                "idstazione": row["idstazione"],
                "nomestazione": row["nomestazione"],
                "lat": row["lat"],
                "lng": row["lng"],
                "hourly": hourly,
            })

    upload_json(weather_records, f"{GCS_DATA_PREFIX}/{date_str}_weather.json")

    return {
        "measurements": len(measurements),
        "weather_stations": len(weather_records),
        "failed_cells": failed_cells,
    }


# -- Main loop ----------------------------------------------------------------

def _heal_window() -> list[str]:
    """Dynamic date window for heal mode: ``[today - START_OFFSET, today - END_OFFSET]``."""
    today = date.today()
    start = today - timedelta(days=HEAL_WINDOW_START_OFFSET)
    end = today - timedelta(days=HEAL_WINDOW_END_OFFSET)
    if end < start:
        raise ValueError(
            f"HEAL_WINDOW_END_OFFSET ({HEAL_WINDOW_END_OFFSET}) must be <= "
            f"HEAL_WINDOW_START_OFFSET ({HEAL_WINDOW_START_OFFSET})."
        )
    return date_range(start.isoformat(), end.isoformat())


def _run() -> None:
    client = RateLimitedClient()
    cp = load_checkpoint()

    if OVERWRITE_EXISTING:
        # Heal mode: ignore BACKFILL_START/END and the checkpoint — every date
        # in the dynamic window is (re)processed and the blob overwritten.
        all_dates = _heal_window()
        remaining = list(all_dates)
        log.info(
            "HEAL mode ON — window [%s..%s] (%d days), ignoring checkpoint.",
            all_dates[0], all_dates[-1], len(all_dates),
        )
    else:
        all_dates = date_range(BACKFILL_START, BACKFILL_END)
        remaining = cp.remaining(all_dates)

    log.info(
        "Total: %d | Done: %d | Remaining: %d | Previously failed: %d",
        len(all_dates), len(cp.completed), len(remaining), len(cp.failed),
    )

    # Nothing left?  Try retrying previously failed dates (only in normal mode).
    if not remaining:
        if not OVERWRITE_EXISTING and cp.failed:
            remaining = sorted(cp.failed.keys())
            log.info("All dates done — retrying %d failed dates.", len(remaining))
        else:
            log.info("Backfill fully complete.  Nothing to do.")
            return

    # Apply batch cap.
    if DAYS_PER_BATCH > 0:
        remaining = remaining[:DAYS_PER_BATCH]
        log.info("Batch capped to %d days.", DAYS_PER_BATCH)

    # Fetch sensor registry once.
    log.info("Fetching sensor registry...")
    sensors_df = fetch_sensor_registry(client)
    target_ids, stations, grid_cells = prepare_grid(sensors_df)
    log.info(
        "Grid: %d cells from %d stations, %d target sensors.",
        len(grid_cells), len(stations), len(target_ids),
    )

    # Process each date.
    days_since_save = 0
    for i, d in enumerate(remaining, start=1):
        if _shutdown_requested:
            log.warning("Shutdown requested — saving checkpoint and exiting.")
            save_checkpoint(cp)
            return

        log.info("[%d/%d] Collecting %s ...", i, len(remaining), d)
        try:
            stats = collect_day(client, d, target_ids, stations, grid_cells)
            cp.mark_completed(d)
            log.info(
                "[%s] OK  meas=%d  weather=%d  failed_cells=%d",
                d, stats["measurements"], stats["weather_stations"], stats["failed_cells"],
            )
        except CircuitBreakerOpen:
            log.warning("Circuit breaker open — saving checkpoint, next run resumes.")
            save_checkpoint(cp)
            sys.exit(0)
        except Exception as exc:
            log.error("[%s] FAILED: %s", d, exc)
            cp.mark_failed(d, str(exc))

        # Periodic checkpoint save.
        days_since_save += 1
        if days_since_save >= CHECKPOINT_EVERY_N_DAYS:
            save_checkpoint(cp)
            days_since_save = 0

        if i < len(remaining):
            time.sleep(INTER_DAY_SLEEP_S)

    # Final save.
    save_checkpoint(cp)
    log.info(
        "Batch done.  Completed: %d | Failed: %d | HTTP stats: %s",
        len(cp.completed), len(cp.failed), client.stats,
    )
    if cp.failed:
        sample = sorted(cp.failed.keys())[:20]
        log.warning("Failed dates (first 20): %s", sample)


# -- Entry-point --------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("Historical Backfill  %s -> %s", BACKFILL_START, BACKFILL_END)
    log.info("Bucket: gs://%s/%s", GCS_BUCKET, GCS_DATA_PREFIX)
    log.info("Batch limit: %s", DAYS_PER_BATCH or "unlimited")
    log.info("=" * 60)

    if not acquire_lock():
        log.error("Could not acquire lock — exiting (another instance running?).")
        sys.exit(0)

    try:
        _run()
    finally:
        release_lock()


if __name__ == "__main__":
    main()
