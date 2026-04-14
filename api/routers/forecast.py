"""Router for PM10 forecast endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, status

from api.schemas import ForecastResponse
from api.services import predictor

log = logging.getLogger(__name__)

router = APIRouter(tags=["forecast"])


@router.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    station_id: str = Query(..., description="Monitoring station identifier."),
    days: int = Query(1, description="Forecast horizon in days (1 or 2)."),
) -> ForecastResponse:
    """Return the PM10 + alert-class forecast for a station (1 or 2 days ahead)."""
    if days not in (1, 2):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query parameter 'days' must be 1 or 2.",
        )

    try:
        result = predictor.predict(station_id, days)
    except predictor.StationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except predictor.ModelNotFoundError as exc:
        log.exception("Prediction models are not loaded: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction models are not available.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        log.exception("Forecast generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Forecast generation failed.",
        ) from exc

    return ForecastResponse(**result)
