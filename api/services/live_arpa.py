"""Live ARPA data client — Socrata gap-fill for the rolling window.

Returns records in the same shape as the GCS blobs
(``[{idsensore, data, valore, stato, ...}]``) so the existing aggregator in
``recent_data.py`` can consume them without changes.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

log = logging.getLogger(__name__)

_SOCRATA_URL = "https://www.dati.lombardia.it/resource/nicp-bhqi.json"

# Sentinel values that ARPA uses to mark missing/invalid data.
_SENTINEL_VALUES: frozenset[float] = frozenset({-9999.0, -999.0, 9999.0})


def _sanitize_arpa_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Null-out sentinel/invalid ``valore`` values, preserving the record."""
    cleaned: list[dict[str, Any]] = []
    for record in records:
        new_record = dict(record)
        raw_valore = new_record.get("valore")
        try:
            valore: float | None = float(raw_valore) if raw_valore is not None else None
        except (TypeError, ValueError):
            valore = None

        stato = new_record.get("stato")
        if valore is None or valore <= 0 or valore in _SENTINEL_VALUES or stato != "VA":
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

    sanitized = _sanitize_arpa_records(records)
    log.info(
        "Socrata day=%s: %d records (after sensor filter+sanitize), %d with null valore.",
        iso_date,
        len(sanitized),
        sum(1 for r in sanitized if r.get("valore") is None),
    )
    return sanitized
