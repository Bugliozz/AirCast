"""Regression tests for PM10 lag handling with trailing NaNs."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from api.services import predictor


class _DummyRegressionModel:
    """Minimal regression stub exposing the feature contract."""

    def __init__(self) -> None:
        self.feature_names_in_ = np.array(
            ["pm10_lag1", "pm10_lag2", "pm10_roll7", "pm10_roll3", "pm10_diff"],
            dtype=object,
        )
        self.last_X: pd.DataFrame | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.last_X = X.copy()
        return np.array([42.0])


class _DummyClassificationModel:
    """Minimal classifier stub for predictor.predict()."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([0])


def _make_lag_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data_giorno": pd.date_range("2026-04-01", periods=8, freq="D"),
            "provincia": ["MI"] * 8,
            "quota": [120.0] * 8,
            "lat": [45.50] * 8,
            "lng": [9.20] * 8,
            "dist_industrial_km": [3.4] * 8,
            "n_industrial_zones_15km": [2.0] * 8,
            "pm10": [18.0, 21.0, np.nan, 25.0, 30.0, 28.0, 35.0, np.nan],
            "no2_mean": [20.0] * 8,
            "o3_mean": [35.0] * 8,
            "no2_max": [28.0] * 8,
            "o3_max": [48.0] * 8,
            "pressure_mean": [1012.0, 1011.0, 1010.0, 1009.0, 1013.0, 1014.0, 1015.0, 1016.0],
            "wind_speed_mean": [2.5, 2.2, 2.1, 1.9, 1.8, 1.7, 1.6, 1.5],
            "blh_mean": [350.0] * 8,
            "temp_mean": [12.0] * 8,
        }
    )


def _make_weather() -> dict[str, float]:
    return {
        "temp_mean": 14.0,
        "humidity_mean": 60.0,
        "dewpoint_mean": 8.0,
        "precip_sum": 0.0,
        "pressure_mean": 1018.0,
        "cloud_cover_mean": 40.0,
        "wind_speed_mean": 2.0,
        "wind_speed_max": 4.0,
        "radiation_mean": 150.0,
        "blh_mean": 400.0,
        "blh_min": 180.0,
        "fog_hours": 0.0,
    }


def test_build_features_forward_fills_short_pm10_gaps(monkeypatch) -> None:
    regression_model = _DummyRegressionModel()
    monkeypatch.setattr(predictor, "_regression_model", regression_model)

    features = predictor.build_features(
        station_id="S1",
        target_date=date(2026, 4, 16),
        weather_daily=_make_weather(),
        lag_data=_make_lag_data(),
    )

    row = features.iloc[0]
    assert row["pm10_lag1"] == 35.0
    assert row["pm10_lag2"] == 35.0
    assert np.isclose(row["pm10_roll7"], 27.857142857142858)
    assert np.isclose(row["pm10_roll3"], 32.666666666666664)
    assert row["pm10_diff"] == 0.0


def test_build_features_caps_forward_fill_at_two_days(monkeypatch) -> None:
    regression_model = _DummyRegressionModel()
    monkeypatch.setattr(predictor, "_regression_model", regression_model)

    lag_data = _make_lag_data()
    lag_data["pm10"] = [18.0, 21.0, 24.0, 25.0, 30.0, np.nan, np.nan, np.nan]

    features = predictor.build_features(
        station_id="S1",
        target_date=date(2026, 4, 16),
        weather_daily=_make_weather(),
        lag_data=lag_data,
    )

    row = features.iloc[0]
    assert row["pm10_lag1"] == 30.0
    assert row["pm10_lag2"] == 30.0
    assert row["pm10_roll3"] == 30.0
    assert np.isclose(row["pm10_roll7"], 26.666666666666668)


def test_predict_handles_trailing_nan_pm10_without_raising(monkeypatch) -> None:
    regression_model = _DummyRegressionModel()

    monkeypatch.setattr(predictor, "_regression_model", regression_model)
    monkeypatch.setattr(predictor, "_classification_model", _DummyClassificationModel())
    monkeypatch.setattr(predictor, "_feature_store", pd.DataFrame({"idstazione": ["S1"]}))
    monkeypatch.setattr(
        predictor,
        "get_station_metadata",
        lambda station_id: {
            "idstazione": station_id,
            "nomestazione": "Milano Verziere",
            "lat": 45.50,
            "lng": 9.20,
        },
    )
    monkeypatch.setattr(
        predictor,
        "compute_data_quality",
        lambda station_id: {"data_quality": "partial", "valid_days_last_7": 5},
    )
    monkeypatch.setattr(predictor, "get_lag_features", lambda station_id, conn=None: _make_lag_data())
    monkeypatch.setattr(
        predictor,
        "fetch_forecast_weather",
        lambda lat, lng, days: [_make_weather()],
    )

    result = predictor.predict("S1", days=1)

    assert result["data_quality"] == "partial"
    assert result["valid_days_last_7"] == 5
    assert result["predictions"][0]["pm10_predicted"] == 42.0
    assert regression_model.last_X is not None
    assert regression_model.last_X.iloc[0]["pm10_lag1"] == 35.0
    assert regression_model.last_X.iloc[0]["pm10_lag2"] == 35.0
