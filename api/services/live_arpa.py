"""Live ARPA data client — Socrata gap-fill for the rolling window.

Returns records in the same shape as the GCS blobs
(``[{idsensore, data, valore, stato, ...}]``) so the existing aggregator in
``recent_data.py`` can consume them without changes.

Two feeds are exposed:

* :func:`fetch_socrata_day` — validated daily archive (``nicp-bhqi``,
  ``stato='VA'``), used to gap-fill any missing day in the rolling window.
* :func:`fetch_nrt_today` — Near Real Time provisional hourly feed
  (``ykhg-b8rs``), used to expose the current day (T₀) PM10 so the lag-1
  feature reflects today rather than yesterday.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

import requests

log = logging.getLogger(__name__)

_SOCRATA_URL = "https://www.dati.lombardia.it/resource/nicp-bhqi.json"
_NRT_MEASUREMENTS_URL = "https://www.dati.lombardia.it/resource/ykhg-b8rs.json"

# Sentinel values that ARPA uses to mark missing/invalid data.
_SENTINEL_VALUES: frozenset[float] = frozenset({-9999.0, -999.0, 9999.0})

# Status tag for NRT records — they have no ``stato`` field in the feed
# (data is provisional / non-validated), so we inject this marker so
# downstream consumers can distinguish them from validated ``VA`` rows.
NRT_STATUS = "NRT"


def _sanitize_arpa_records(
    records: list[dict[str, Any]],
    *,
    allowed_statuses: frozenset[str] = frozenset({"VA"}),
) -> list[dict[str, Any]]:
    """Null-out sentinel/invalid ``valore`` values, preserving the record.

    Records whose ``stato`` is not in ``allowed_statuses`` also have their
    ``valore`` nulled out — this lets the same helper handle both the
    validated archive (``{'VA'}``) and the NRT feed (``{'NRT'}``).
    """
    cleaned: list[dict[str, Any]] = []
    for record in records:
        new_record = dict(record)
        raw_valore = new_record.get("valore")
        try:
            valore: float | None = float(raw_valore) if raw_valore is not None else None
        except (TypeError, ValueError):
            valore = None

        stato = new_record.get("stato")
        if (
            valore is None
            or valore <= 0
            or valore in _SENTINEL_VALUES
            or stato not in allowed_statuses
        ):
            new_record["valore"] = None
        else:
            new_record["valore"] = valore
        cleaned.append(new_record)
    return cleaned


def fetch_socrata_day(
    iso_date: str,
    target_sensor_ids: set[str],
    *,
    timeout_s: float = 10.0,
) -> list[dict[str, Any]]:
    """Fetch validated ARPA measurements for a single day from Socrata.

    Returns ``[]`` on any HTTP error or timeout (swallowed, warning logged).
    """
    where_clause = (
        f"data >= '{iso_date}T00:00:00.000' "
        f"AND data <= '{iso_date}T23:59:59.999' "
        f"AND stato='VA'"
    )
    params: dict[str, Any] = {
        "$where": where_clause,
        "$limit": 50000,
    }

    try:
        resp = requests.get(_SOCRATA_URL, params=params, timeout=timeout_s)
        resp.raise_for_status()
        records: list[dict[str, Any]] = resp.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("fetch_socrata_day(%s) failed: %s", iso_date, exc)
        return []

    if target_sensor_ids:
        records = [r for r in records if str(r.get("idsensore", "")) in target_sensor_ids]

    sanitized = _sanitize_arpa_records(records, allowed_statuses=frozenset({"VA"}))
    log.info(
        "Socrata day=%s: %d records (after sensor filter+sanitize), %d with null valore.",
        iso_date,
        len(sanitized),
        sum(1 for r in sanitized if r.get("valore") is None),
    )
    return sanitized


def fetch_nrt_today(
    target_sensor_ids: set[str],
    *,
    timeout_s: float = 10.0,
) -> list[dict[str, Any]]:
    """Fetch today's hourly PM10 (and co-pollutant) measurements from NRT.

    The NRT feed (``ykhg-b8rs``) publishes provisional, non-validated hourly
    readings for the current day — the validated archive (``nicp-bhqi``)
    lags by at least one day.  This is the only feed that exposes T₀
    observations, so it is used to lift the lag-1 feature from T−1 (yesterday)
    to T₀ (today) for stations that have NRT coverage.

    NRT records have no ``stato`` field; this function tags each one with
    ``stato='NRT'`` so downstream consumers can distinguish provisional data
    from validated archive rows (``stato='VA'``).

    Returns ``[]`` on any HTTP error or timeout (swallowed, warning logged).
    """
    today = date.today().isoformat()
    where_clause = (
        f"data >= '{today}T00:00:00.000' "
        f"AND data <= '{today}T23:59:59.999'"
    )
    params: dict[str, Any] = {
        "$where": where_clause,
        "$limit": 50000,
    }

    try:
        resp = requests.get(_NRT_MEASUREMENTS_URL, params=params, timeout=timeout_s)
        resp.raise_for_status()
        records: list[dict[str, Any]] = resp.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("fetch_nrt_today(%s) failed: %s", today, exc)
        return []

    tagged: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("idsensore", "")) not in target_sensor_ids:
            continue
        new_record = dict(record)
        new_record["stato"] = NRT_STATUS
        tagged.append(new_record)

    sanitized = _sanitize_arpa_records(
        tagged, allowed_statuses=frozenset({NRT_STATUS})
    )
    log.info(
        "NRT today=%s: %d records (after sensor filter+sanitize), %d with null valore.",
        today,
        len(sanitized),
        sum(1 for r in sanitized if r.get("valore") is None),
    )
    return sanitized
