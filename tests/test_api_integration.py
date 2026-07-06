"""Integration tests for the FastAPI app's core endpoints.

These tests instantiate ``TestClient`` without the ``with`` context manager,
so the app's ``lifespan`` (model loading, GCS prefetch) never runs — keeping
the tests fast and independent of trained artifacts or network access.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import StationOut
from api.services import history


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestStationsEndpoint:
    def test_stations_returns_expected_fields(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_stations = [
            StationOut(
                idstazione="501",
                nomestazione="Milano - Test",
                comune="Milano",
                lat=45.4642,
                lon=9.1900,
            ),
            StationOut(
                idstazione="502",
                nomestazione="Bergamo - Test",
                comune="Bergamo",
                lat=45.6983,
                lon=9.6773,
            ),
        ]
        monkeypatch.setattr(history, "get_stations", lambda: fake_stations)

        response = client.get("/stations")
        assert response.status_code == 200

        body = response.json()
        assert isinstance(body, list)
        assert len(body) == 2
        for station in body:
            assert "idstazione" in station
            assert "lat" in station
            assert "lon" in station
