"""Backfill historical air quality data for a date range.

Fetches ARPA Lombardia measurements and Open-Meteo weather for each day
and saves them as JSON files in data/raw/.

Usage examples
--------------
# Default: 1 year ago → yesterday (Italian time)
python backfill.py

# Custom date range
python backfill.py --start-date 2025-01-01 --end-date 2025-12-31

# Skip days whose files already exist
python backfill.py --skip-existing

# Dry run (print dates without making API calls)
python backfill.py --start-date 2025-03-01 --end-date 2025-03-10 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# Allow running directly from this directory
sys.path.insert(0, str(Path(__file__).parent))

from collector import (
    fetch_arpa_measurements,
    fetch_sensor_registry,
    fetch_station_weather_historical,
    resolve_output_dir,
    save_json,
)

# Round coordinates to deduplicate weather API calls.
# 1 dp ≈ 11 km → reduces ~170 Lombardia stations to ~15–20 unique grid cells.
COORD_ROUND_DP = 1
INTER_CELL_SLEEP = 0.5   # seconds between weather cell requests
INTER_DAY_SLEEP = 2.0    # seconds between days

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-day collection
# ---------------------------------------------------------------------------

def collect_day(
    date_str: str,
    sensors_df: pd.DataFrame,
    output_dir: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    """Collect measurements + weather for one date and persist as JSON.

    sensors_df is fetched once by the caller and reused across days
    to avoid hammering the registry endpoint.
    """
    if skip_existing:
        meas_file = output_dir / f"{date_str}_measurements.json"
        weather_file = output_dir / f"{date_str}_weather.json"
        if meas_file.exists() and weather_file.exists():
            log.info("[%s] already collected — skipping.", date_str)
            return {"date": date_str, "status": "skipped"}

    # --- Measurements ---
    target_sensor_ids = set(sensors_df["idsensore"].astype(str).tolist())
    all_measurements = fetch_arpa_measurements(date_str)
    measurements = [m for m in all_measurements if m["idsensore"] in target_sensor_ids]
    save_json(f"{date_str}_measurements.json", measurements, output_dir)

    # --- Weather (deduplicated by grid cell) ---
    stations = (
        sensors_df[["idstazione", "nomestazione", "lat", "lng"]]
        .drop_duplicates(subset="idstazione")
        .dropna(subset=["lat", "lng"])
        .copy()
    )
    stations["lat_r"] = stations["lat"].astype(float).round(COORD_ROUND_DP).astype(str)
    stations["lng_r"] = stations["lng"].astype(float).round(COORD_ROUND_DP).astype(str)

    unique_cells = stations[["lat_r", "lng_r"]].drop_duplicates()
    log.info(
        "[%s] fetching weather for %d grid cells (%d stations total).",
        date_str, len(unique_cells), len(stations),
    )

    weather_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for i, (_, cell) in enumerate(unique_cells.iterrows()):
        key = (cell["lat_r"], cell["lng_r"])
        try:
            weather_cache[key] = fetch_station_weather_historical(
                cell["lat_r"], cell["lng_r"], date_str
            )
        except requests.RequestException as exc:
            log.warning("[%s] weather fetch failed for cell %s: %s", date_str, key, exc)
        if i < len(unique_cells) - 1:
            time.sleep(INTER_CELL_SLEEP)

    weather_records: list[dict[str, Any]] = []
    failed_stations = 0
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
        else:
            failed_stations += 1

    save_json(f"{date_str}_weather.json", weather_records, output_dir)

    return {
        "date": date_str,
        "status": "ok",
        "measurements": len(measurements),
        "weather_stations": len(weather_records),
        "failed_stations": failed_stations,
    }


# ---------------------------------------------------------------------------
# Date range helpers
# ---------------------------------------------------------------------------

def _default_start() -> str:
    yesterday = pd.Timestamp.now(tz="Europe/Rome") - pd.Timedelta(days=1)
    return (yesterday - pd.Timedelta(days=365)).strftime("%Y-%m-%d")


def _default_end() -> str:
    return (pd.Timestamp.now(tz="Europe/Rome") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def date_range(start: str, end: str) -> list[str]:
    """Return all dates in [start, end] as YYYY-MM-DD strings."""
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    return [
        (start_d + timedelta(days=i)).isoformat()
        for i in range((end_d - start_d).days + 1)
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill historical air quality + weather data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start-date", default=None,
        help="First date to collect (YYYY-MM-DD). Default: 1 year ago.",
    )
    parser.add_argument(
        "--end-date", default=None,
        help="Last date to collect (YYYY-MM-DD). Default: yesterday.",
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="PATH",
        help="Local output directory. Default: data/raw/.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip days whose output files already exist.",
    )
    parser.add_argument(
        "--sleep", type=float, default=INTER_DAY_SLEEP, metavar="SECONDS",
        help="Seconds to sleep between each day's collection.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the dates that would be collected without making any API calls.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    start = args.start_date or _default_start()
    end = args.end_date or _default_end()
    dates = date_range(start, end)
    output_dir = resolve_output_dir(args.output_dir)

    log.info("Backfill: %s → %s (%d days)", start, end, len(dates))
    log.info("Output: %s", output_dir)

    if args.dry_run:
        log.info("Dry-run — no API calls.")
        for d in dates:
            print(d)
        return

    # Fetch sensor registry once and reuse across all days
    log.info("Fetching sensor registry…")
    sensors_df = fetch_sensor_registry()

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for i, d in enumerate(dates, start=1):
        log.info("[%d/%d] Collecting %s…", i, len(dates), d)
        try:
            result = collect_day(d, sensors_df, output_dir, args.skip_existing)
            results.append(result)
            if result["status"] == "ok":
                log.info(
                    "[%s] OK — %d measurements, %d weather stations",
                    d, result["measurements"], result["weather_stations"],
                )
        except Exception as exc:
            log.error("[%s] FAILED: %s", d, exc)
            errors.append(f"{d}: {exc}")
            results.append({"date": d, "status": "error", "error": str(exc)})

        if i < len(dates):
            time.sleep(args.sleep)

    # --- Summary ---
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "error")

    print("\n" + "=" * 50)
    print(f"Backfill complete: {ok} collected, {skipped} skipped, {failed} failed")
    if errors:
        print("Failed dates:")
        for e in errors:
            print(f"  {e}")

    summary_path = output_dir / f"backfill_summary_{start}_{end}.json"
    summary_path.write_text(
        json.dumps({"start": start, "end": end, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
