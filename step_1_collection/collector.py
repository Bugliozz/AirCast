"""Core data collection functions.

Fetches air quality measurements from ARPA Lombardia Socrata API
and weather from the Open-Meteo historical forecast endpoint.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# ARPA Lombardia Socrata API
# ---------------------------------------------------------------------------
ARPA_MEASUREMENTS_URL = "https://www.dati.lombardia.it/resource/nicp-bhqi.json"
ARPA_SENSORS_URL = "https://www.dati.lombardia.it/resource/ib47-atvt.json"
ARPA_PAGE_SIZE = 50_000

# ---------------------------------------------------------------------------
# Open-Meteo historical forecast API (same model as live, works on past dates)
# ---------------------------------------------------------------------------
OPENMETEO_ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPENMETEO_HOURLY_VARS = (
    "temperature_2m,"
    "relative_humidity_2m,"
    "dew_point_2m,"
    "precipitation,"
    "surface_pressure,"
    "cloud_cover,"
    "wind_speed_10m,"
    "wind_direction_10m,"
    "visibility,"
    "shortwave_radiation,"
    "boundary_layer_height"
)

REQUEST_TIMEOUT = 30
WEATHER_MAX_RETRIES = 3
WEATHER_RETRY_BACKOFF = 3.0  # seconds; doubled on each retry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def resolve_output_dir(path: str | None = None) -> Path:
    """Return the output directory as a Path, creating it if needed."""
    if path:
        out = Path(path)
    else:
        # data/raw/ relative to project root (one level up from this file)
        out = Path(__file__).parent.parent / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(filename: str, data: Any, output_dir: Path) -> str:
    """Serialize data to JSON and write to output_dir/filename. Returns the path."""
    target = output_dir / filename
    target.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    n = len(data) if isinstance(data, list) else 1
    log.debug("Saved %s (%d records)", target, n)
    return str(target)


# ---------------------------------------------------------------------------
# ARPA Lombardia
# ---------------------------------------------------------------------------

def fetch_sensor_registry() -> pd.DataFrame:
    """Fetch all sensor metadata from ARPA Lombardia. Returns a DataFrame."""
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        resp = requests.get(
            ARPA_SENSORS_URL,
            params={"$limit": ARPA_PAGE_SIZE, "$offset": offset},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        records.extend(page)
        if len(page) < ARPA_PAGE_SIZE:
            break
        offset += ARPA_PAGE_SIZE

    log.info("Fetched %d sensors from registry.", len(records))
    return pd.DataFrame(records)


def fetch_arpa_measurements(date_str: str) -> list[dict[str, Any]]:
    """Fetch validated ARPA measurements for a given date (YYYY-MM-DD).

    PM10/PM2.5 return one daily value at T00:00:00.
    NO2, O3, CO return hourly values.
    Only records with stato='VA' (validated) are retrieved.
    """
    where_clause = (
        f"data >= '{date_str}T00:00:00.000' "
        f"AND data <= '{date_str}T23:59:59.999' "
        f"AND stato = 'VA'"
    )
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        resp = requests.get(
            ARPA_MEASUREMENTS_URL,
            params={"$where": where_clause, "$limit": ARPA_PAGE_SIZE, "$offset": offset},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        records.extend(page)
        if len(page) < ARPA_PAGE_SIZE:
            break
        offset += ARPA_PAGE_SIZE

    log.info("[%s] Fetched %d measurement records.", date_str, len(records))
    return records


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------

def fetch_station_weather_historical(
    lat: str, lng: str, date_str: str
) -> dict[str, Any]:
    """Fetch hourly weather from Open-Meteo for a single grid cell and date.

    Retries with exponential backoff on transient failures.
    """
    params = {
        "latitude": lat,
        "longitude": lng,
        "hourly": OPENMETEO_HOURLY_VARS,
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "Europe/Rome",
    }
    last_exc: Exception | None = None
    wait = WEATHER_RETRY_BACKOFF
    for attempt in range(1, WEATHER_MAX_RETRIES + 1):
        try:
            resp = requests.get(OPENMETEO_ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < WEATHER_MAX_RETRIES:
                log.debug(
                    "Weather attempt %d/%d failed (%s). Retrying in %.0fs…",
                    attempt, WEATHER_MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
                wait *= 2
    raise last_exc  # type: ignore[misc]
