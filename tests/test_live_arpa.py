"""Unit tests for api.services.live_arpa.fetch_socrata_day.

Uses ``requests-mock`` to intercept HTTP calls — no real network traffic.
"""
from __future__ import annotations

import pytest
import requests

_ISO_DATE = "2026-04-10"
_SENSOR_IDS = {"6918", "6919"}
_SOCRATA_URL = "https://www.dati.lombardia.it/resource/nicp-bhqi.json"


def _make_socrata_record(
    idsensore: str = "6918",
    data: str = f"{_ISO_DATE}T12:00:00.000",
    valore: str = "35.5",
    stato: str = "VA",
) -> dict:
    return {"idsensore": idsensore, "data": data, "valore": valore, "stato": stato}


class TestFetchSocrataDay:
    def test_returns_valid_records_with_correct_shape(self, requests_mock):
        from api.services.live_arpa import fetch_socrata_day

        requests_mock.get(_SOCRATA_URL, json=[_make_socrata_record(valore="35.5", stato="VA")])
        result = fetch_socrata_day(_ISO_DATE, _SENSOR_IDS)

        assert len(result) == 1
        rec = result[0]
        assert rec["idsensore"] == "6918"
        assert rec["stato"] == "VA"
        assert rec["valore"] == pytest.approx(35.5)

    def test_filters_by_sensor_ids(self, requests_mock):
        from api.services.live_arpa import fetch_socrata_day

        raw = [_make_socrata_record(idsensore="6918"), _make_socrata_record(idsensore="9999")]
        requests_mock.get(_SOCRATA_URL, json=raw)

        result = fetch_socrata_day(_ISO_DATE, {"6918"})

        assert all(r["idsensore"] == "6918" for r in result)
        assert len(result) == 1

    def test_sentinel_minus9999_becomes_null(self, requests_mock):
        from api.services.live_arpa import fetch_socrata_day

        requests_mock.get(_SOCRATA_URL, json=[_make_socrata_record(valore="-9999", stato="VA")])
        result = fetch_socrata_day(_ISO_DATE, _SENSOR_IDS)

        assert len(result) == 1
        assert result[0]["valore"] is None

    def test_non_va_stato_becomes_null(self, requests_mock):
        from api.services.live_arpa import fetch_socrata_day

        requests_mock.get(_SOCRATA_URL, json=[_make_socrata_record(valore="22.0", stato="NV")])
        result = fetch_socrata_day(_ISO_DATE, _SENSOR_IDS)

        assert len(result) == 1
        assert result[0]["valore"] is None

    def test_non_positive_valore_becomes_null(self, requests_mock):
        from api.services.live_arpa import fetch_socrata_day

        requests_mock.get(_SOCRATA_URL, json=[_make_socrata_record(valore="0.0", stato="VA")])
        result = fetch_socrata_day(_ISO_DATE, _SENSOR_IDS)

        assert result[0]["valore"] is None

    def test_http_500_returns_empty_list(self, requests_mock):
        from api.services.live_arpa import fetch_socrata_day

        requests_mock.get(_SOCRATA_URL, status_code=500)
        assert fetch_socrata_day(_ISO_DATE, _SENSOR_IDS) == []

    def test_timeout_returns_empty_list(self, requests_mock):
        from api.services.live_arpa import fetch_socrata_day

        requests_mock.get(_SOCRATA_URL, exc=requests.exceptions.Timeout)
        assert fetch_socrata_day(_ISO_DATE, _SENSOR_IDS) == []

    def test_empty_target_sensor_ids_returns_all_records(self, requests_mock):
        from api.services.live_arpa import fetch_socrata_day

        raw = [_make_socrata_record(idsensore="6918"), _make_socrata_record(idsensore="7000")]
        requests_mock.get(_SOCRATA_URL, json=raw)

        assert len(fetch_socrata_day(_ISO_DATE, set())) == 2

    def test_no_records_returns_empty_list(self, requests_mock):
        from api.services.live_arpa import fetch_socrata_day

        requests_mock.get(_SOCRATA_URL, json=[])
        assert fetch_socrata_day(_ISO_DATE, _SENSOR_IDS) == []
