"""Pydantic schemas for the REST API request/response models."""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel


class StationOut(BaseModel):
    """Metadata for a monitoring station."""

    idstazione: str
    nomestazione: str
    comune: str
    lat: float
    lon: float


class WeatherUsed(BaseModel):
    """Daily weather summary used as input features for the prediction."""

    temp_mean: Optional[float] = None
    wind_speed_mean: Optional[float] = None
    boundary_layer_height_mean: Optional[float] = None


class ForecastItem(BaseModel):
    """Single-day forecast for one station."""

    date: date
    pm10_predicted: float
    alert_class: str          # "verde" | "giallo" | "arancio" | "rosso"
    alert_index: int          # 0–3 (ordinal severity)
    weather_used: WeatherUsed


class ForecastResponse(BaseModel):
    """Full forecast response for a station (1 or 2 days ahead)."""

    station_id: str
    station_name: str
    predictions: List[ForecastItem]


class HistoryRecord(BaseModel):
    """One historical PM10 measurement for a station."""

    date: date
    pm10: Optional[float] = None
    alert_class: Optional[str] = None  # "verde" | "giallo" | "arancio" | "rosso"


class HistoryResponse(BaseModel):
    """Historical PM10 series for a station."""

    station_id: str
    records: List[HistoryRecord]
