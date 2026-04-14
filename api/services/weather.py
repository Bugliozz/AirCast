"""Weather service — fetch and aggregate Open-Meteo forecast data.

Provides `fetch_forecast_weather(lat, lng, days)` which calls the Open-Meteo
Forecast API and returns daily-aggregated weather records that match the schema
produced by `step_3_eda.db.load_weather_daily_agg`.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants — reused from step_1_collection/collector.py
# ---------------------------------------------------------------------------
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

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
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_raw(
    lat: float | str,
    lng: float | str,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Call Open-Meteo Forecast API with exponential-backoff retry.

    Parameters
    ----------
    lat, lng:
        Coordinates of the monitoring station.
    start_date, end_date:
        Inclusive date range for the forecast horizon.

    Returns
    -------
    Raw JSON response as returned by Open-Meteo.
    """
    params = {
        "latitude": lat,
        "longitude": lng,
        "hourly": OPENMETEO_HOURLY_VARS,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "Europe/Rome",
    }

    last_exc: Exception | None = None
    wait = WEATHER_RETRY_BACKOFF

    for attempt in range(1, WEATHER_MAX_RETRIES + 1):
        try:
            resp = requests.get(
                OPENMETEO_FORECAST_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < WEATHER_MAX_RETRIES:
                log.debug(
                    "Forecast weather attempt %d/%d failed (%s). Retrying in %.0fs…",
                    attempt, WEATHER_MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
                wait *= 2

    raise last_exc  # type: ignore[misc]


def _aggregate_daily(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate hourly Open-Meteo data to daily records.

    Applies the same aggregation formulas as ``step_3_eda.db.load_weather_daily_agg``:

    - temp_mean / temp_min / temp_max   → AVG/MIN/MAX temperature_2m
    - humidity_mean                     → AVG relative_humidity_2m
    - dewpoint_mean                     → AVG dew_point_2m
    - precip_sum                        → SUM precipitation
    - pressure_mean                     → AVG surface_pressure
    - cloud_cover_mean                  → AVG cloud_cover
    - wind_speed_mean / wind_speed_max  → AVG/MAX wind_speed_10m
    - visibility_mean                   → AVG visibility
    - radiation_mean                    → AVG shortwave_radiation
    - blh_mean / blh_min                → AVG/MIN boundary_layer_height
    - fog_hours                         → COUNT hours where (temp - dewpoint) < 2
    """
    hourly = raw.get("hourly", {})
    timestamps: list[str] = hourly.get("time", [])

    # Build parallel arrays keyed by variable name
    var_arrays: dict[str, list[float | None]] = {
        "temperature_2m":        hourly.get("temperature_2m", []),
        "relative_humidity_2m":  hourly.get("relative_humidity_2m", []),
        "dew_point_2m":          hourly.get("dew_point_2m", []),
        "precipitation":         hourly.get("precipitation", []),
        "surface_pressure":      hourly.get("surface_pressure", []),
        "cloud_cover":           hourly.get("cloud_cover", []),
        "wind_speed_10m":        hourly.get("wind_speed_10m", []),
        "visibility":            hourly.get("visibility", []),
        "shortwave_radiation":   hourly.get("shortwave_radiation", []),
        "boundary_layer_height": hourly.get("boundary_layer_height", []),
    }

    # Group indices by calendar date (YYYY-MM-DD prefix of "YYYY-MM-DDTHH:MM")
    date_to_indices: dict[str, list[int]] = {}
    for i, ts in enumerate(timestamps):
        day_str = ts[:10]
        date_to_indices.setdefault(day_str, []).append(i)

    daily_records: list[dict[str, Any]] = []

    for day_str, indices in sorted(date_to_indices.items()):

        def _vals(col: str) -> list[float]:
            arr = var_arrays.get(col, [])
            return [arr[i] for i in indices if i < len(arr) and arr[i] is not None]

        def _mean(col: str) -> float | None:
            v = _vals(col)
            return sum(v) / len(v) if v else None

        def _min(col: str) -> float | None:
            v = _vals(col)
            return min(v) if v else None

        def _max(col: str) -> float | None:
            v = _vals(col)
            return max(v) if v else None

        def _sum(col: str) -> float | None:
            v = _vals(col)
            return sum(v) if v else None

        # fog_hours: hours where (temp - dewpoint) < 2 °C
        temp_arr = var_arrays.get("temperature_2m", [])
        dewp_arr = var_arrays.get("dew_point_2m", [])
        fog_count = sum(
            1
            for i in indices
            if (
                i < len(temp_arr)
                and i < len(dewp_arr)
                and temp_arr[i] is not None
                and dewp_arr[i] is not None
                and (temp_arr[i] - dewp_arr[i]) < 2.0
            )
        )

        daily_records.append({
            "date":             day_str,
            "temp_mean":        _mean("temperature_2m"),
            "temp_min":         _min("temperature_2m"),
            "temp_max":         _max("temperature_2m"),
            "humidity_mean":    _mean("relative_humidity_2m"),
            "dewpoint_mean":    _mean("dew_point_2m"),
            "precip_sum":       _sum("precipitation"),
            "pressure_mean":    _mean("surface_pressure"),
            "cloud_cover_mean": _mean("cloud_cover"),
            "wind_speed_mean":  _mean("wind_speed_10m"),
            "wind_speed_max":   _max("wind_speed_10m"),
            "visibility_mean":  _mean("visibility"),
            "radiation_mean":   _mean("shortwave_radiation"),
            "blh_mean":         _mean("boundary_layer_height"),
            "blh_min":          _min("boundary_layer_height"),
            "fog_hours":        fog_count,
        })

    return daily_records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_forecast_weather(
    lat: float | str,
    lng: float | str,
    days: int,
) -> list[dict[str, Any]]:
    """Fetch and aggregate weather forecast for the next *days* days.

    Calls the Open-Meteo Forecast API for a future horizon of ``days`` calendar
    days starting from tomorrow, then aggregates the hourly data to daily
    records using the same formulas as ``step_3_eda.db.load_weather_daily_agg``.

    Parameters
    ----------
    lat, lng:
        Geographic coordinates of the monitoring station.
    days:
        Number of forecast days to fetch (1 = tomorrow, 2 = tomorrow + day after).

    Returns
    -------
    List of daily weather dicts, one entry per requested day, ordered by date.
    Each dict has keys: date, temp_mean, temp_min, temp_max, humidity_mean,
    dewpoint_mean, precip_sum, pressure_mean, cloud_cover_mean, wind_speed_mean,
    wind_speed_max, visibility_mean, radiation_mean, blh_mean, blh_min, fog_hours.
    """
    today = date.today()
    start_date = today + timedelta(days=1)
    end_date = today + timedelta(days=days)

    log.debug(
        "Fetching forecast weather lat=%.4f lng=%.4f %s → %s",
        float(lat), float(lng), start_date, end_date,
    )

    raw = _fetch_raw(lat, lng, start_date, end_date)
    records = _aggregate_daily(raw)

    log.debug("Aggregated %d daily weather records.", len(records))
    return records
