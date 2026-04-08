"""ARPA Lombardia and Open-Meteo API wrappers that route through
the rate-limited HTTP client.

These functions mirror the ones in ``step_1_collection/collector.py``
but are self-contained and use :class:`rate_limiter.RateLimitedClient`
for automatic retry / back-off.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from config import (
    ARPA_APP_TOKEN,
    ARPA_MEASUREMENTS_URL,
    ARPA_PAGE_SIZE,
    ARPA_SENSORS_URL,
    OPENMETEO_ARCHIVE_URL,
    OPENMETEO_HOURLY_VARS,
)
from rate_limiter import RateLimitedClient

log = logging.getLogger(__name__)


# -- helpers ------------------------------------------------------------------

def _arpa_headers() -> dict[str, str]:
    """Return Socrata headers, including app-token when configured."""
    headers: dict[str, str] = {}
    if ARPA_APP_TOKEN:
        headers["X-App-Token"] = ARPA_APP_TOKEN
    return headers


# -- ARPA Lombardia -----------------------------------------------------------

def fetch_sensor_registry(client: RateLimitedClient) -> pd.DataFrame:
    """Download the full ARPA sensor catalogue (paginated)."""
    records: list[dict[str, Any]] = []
    offset = 0
    headers = _arpa_headers()

    while True:
        resp = client.get(
            ARPA_SENSORS_URL,
            params={"$limit": ARPA_PAGE_SIZE, "$offset": offset},
            headers=headers,
        )
        page: list[dict[str, Any]] = resp.json()
        if not page:
            break
        records.extend(page)
        if len(page) < ARPA_PAGE_SIZE:
            break
        offset += ARPA_PAGE_SIZE

    log.info("Sensor registry: %d sensors fetched.", len(records))
    return pd.DataFrame(records)


def fetch_arpa_measurements(
    client: RateLimitedClient,
    date_str: str,
) -> list[dict[str, Any]]:
    """Fetch validated ARPA measurements for a single date (YYYY-MM-DD).

    Only records with ``stato='VA'`` (validated) are returned.
    For very old dates the dataset may legitimately contain zero rows.
    """
    where_clause = (
        f"data >= '{date_str}T00:00:00.000' "
        f"AND data <= '{date_str}T23:59:59.999' "
        f"AND stato = 'VA'"
    )
    records: list[dict[str, Any]] = []
    offset = 0
    headers = _arpa_headers()

    while True:
        resp = client.get(
            ARPA_MEASUREMENTS_URL,
            params={
                "$where": where_clause,
                "$limit": ARPA_PAGE_SIZE,
                "$offset": offset,
            },
            headers=headers,
        )
        page: list[dict[str, Any]] = resp.json()
        if not page:
            break
        records.extend(page)
        if len(page) < ARPA_PAGE_SIZE:
            break
        offset += ARPA_PAGE_SIZE

    log.info("[%s] ARPA: %d measurement records.", date_str, len(records))
    return records


# -- Open-Meteo ERA5 Archive --------------------------------------------------

def fetch_weather_archive(
    client: RateLimitedClient,
    lat: str,
    lng: str,
    date_str: str,
) -> dict[str, Any]:
    """Fetch hourly weather from the Open-Meteo ERA5 Archive for one cell/day.

    Uses ``archive-api.open-meteo.com`` which covers 1940-present and is
    required for dates before Jan 2022 (the historical-forecast endpoint
    only starts from that date).
    """
    resp = client.get(
        OPENMETEO_ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lng,
            "hourly": OPENMETEO_HOURLY_VARS,
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "Europe/Rome",
        },
    )
    return resp.json()
