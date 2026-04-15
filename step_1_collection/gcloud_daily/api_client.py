"""ARPA + Open-Meteo wrappers for the daily refresh job.

Mirrors ``gcloud_backfill/api_client.py`` but allows ARPA records without
the ``stato='VA'`` flag so that recent, not-yet-validated days are still
collected.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from config import (
    ARPA_APP_TOKEN,
    ARPA_MEASUREMENTS_URL,
    ARPA_PAGE_SIZE,
    ARPA_REQUIRE_VALIDATED,
    ARPA_SENSORS_URL,
    OPENMETEO_ARCHIVE_URL,
    OPENMETEO_HOURLY_VARS,
)
from rate_limiter import RateLimitedClient

log = logging.getLogger(__name__)


def _arpa_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if ARPA_APP_TOKEN:
        headers["X-App-Token"] = ARPA_APP_TOKEN
    return headers


# -- ARPA ---------------------------------------------------------------------

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
    """Fetch ARPA measurements for a single date.

    By default the ``stato='VA'`` (validated) filter is dropped so that
    recent preliminary data is included — controlled by
    ``ARPA_REQUIRE_VALIDATED`` in ``config.py``.
    """
    where_parts = [
        f"data >= '{date_str}T00:00:00.000'",
        f"data <= '{date_str}T23:59:59.999'",
    ]
    if ARPA_REQUIRE_VALIDATED:
        where_parts.append("stato = 'VA'")
    where_clause = " AND ".join(where_parts)

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

    sanitized = _sanitize_arpa_records(records)
    log.info(
        "[%s] ARPA: %d measurement records (%d with null valore after sanitization).",
        date_str,
        len(sanitized),
        sum(1 for r in sanitized if r.get("valore") is None),
    )
    return sanitized


def _sanitize_arpa_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Null-out sentinel/invalid measurements while preserving the record.

    ARPA marks not-yet-validated readings with ``stato != 'VA'`` and often
    uses ``-9999`` as a sentinel ``valore``. We keep every record (so we
    retain ``stato``, ``idsensore``, ``data`` for traceability) but set
    ``valore`` to ``None`` — serialised as JSON ``null`` — whenever the
    reading is not a trustworthy positive number.
    """
    cleaned: list[dict[str, Any]] = []
    for record in records:
        new_record = dict(record)
        raw_valore = new_record.get("valore")
        try:
            valore = float(raw_valore) if raw_valore is not None else None
        except (TypeError, ValueError):
            valore = None
        stato = new_record.get("stato")
        if valore is None or valore <= 0 or stato != "VA":
            new_record["valore"] = None
        else:
            new_record["valore"] = valore
        cleaned.append(new_record)
    return cleaned


# -- Open-Meteo ---------------------------------------------------------------

def fetch_weather_archive(
    client: RateLimitedClient,
    lat: str,
    lng: str,
    date_str: str,
) -> dict[str, Any]:
    """Fetch hourly weather from the Open-Meteo ERA5 Archive for one cell/day."""
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
