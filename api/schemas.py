"""Pydantic schemas for the REST API request/response models."""

from __future__ import annotations

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel

DataQuality = Literal["ok", "partial", "stale"]


class StationOut(BaseModel):
    """Metadata for a monitoring station."""

    idstazione: str
    nomestazione: str
    comune: str
    lat: float
    lon: float
    data_quality: DataQuality = "ok"
    valid_days_last_7: int = 7
    nrt_available: bool = False


class WeatherUsed(BaseModel):
    """Daily weather summary used as input features for the prediction."""

    temp_mean: Optional[float] = None
    wind_speed_mean: Optional[float] = None
    boundary_layer_height_mean: Optional[float] = None


class ForecastDrivers(BaseModel):
    """Most relevant model input features exposed for forecast explainability."""

    temp_mean: Optional[float] = None
    pm10_lag1: Optional[float] = None
    pm10_lag2: Optional[float] = None
    pm10_roll3: Optional[float] = None
    pm10_roll7: Optional[float] = None
    pm10_diff: Optional[float] = None
    pressure_mean: Optional[float] = None
    pressure_mean_lag1: Optional[float] = None
    wind_speed_mean: Optional[float] = None
    wind_speed_max: Optional[float] = None
    wind_speed_roll3: Optional[float] = None
    blh_mean: Optional[float] = None
    blh_min: Optional[float] = None
    precip_sum: Optional[float] = None
    stagnation_index: Optional[float] = None
    stagnation_flag: Optional[bool] = None
    no2_mean: Optional[float] = None
    o3_mean: Optional[float] = None
    heating_season: Optional[int] = None
    stagione: Optional[str] = None
    provincia: Optional[str] = None
    dist_industrial_km: Optional[float] = None


class ForecastItem(BaseModel):
    """Single-day forecast for one station."""

    date: date
    pm10_predicted: float
    alert_class: str          # "verde" | "giallo" | "arancio" | "rosso"
    alert_index: int          # 0–3 (ordinal severity)
    alert_source: Optional[str] = None
    weather_used: WeatherUsed
    model_drivers: Optional[ForecastDrivers] = None


class ForecastResponse(BaseModel):
    """Full forecast response for a station (1 or 2 days ahead)."""

    station_id: str
    station_name: str
    predictions: List[ForecastItem]
    data_quality: DataQuality = "ok"
    valid_days_last_7: int = 7
    nrt_available: bool = False


class HistoryRecord(BaseModel):
    """One historical PM10 measurement for a station."""

    date: date
    pm10: Optional[float] = None
    alert_class: Optional[str] = None  # "verde" | "giallo" | "arancio" | "rosso"


class HistoryResponse(BaseModel):
    """Historical PM10 series for a station."""

    station_id: str
    records: List[HistoryRecord]
