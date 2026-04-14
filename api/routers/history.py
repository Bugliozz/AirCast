"""Router for historical PM10 series endpoints."""

from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, HTTPException, Query, status

from api.schemas import HistoryResponse
from api.services import history as history_service

log = logging.getLogger(__name__)

router = APIRouter(tags=["history"])


@router.get("/history", response_model=HistoryResponse)
def get_history(
    station_id: str = Query(..., description="Station identifier (idstazione)."),
    from_date: date = Query(..., description="Inclusive start date (YYYY-MM-DD)."),
    to_date: date = Query(..., description="Inclusive end date (YYYY-MM-DD)."),
) -> HistoryResponse:
    """Return daily PM10 history for a station in the given date range."""
    if from_date > to_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="from_date must be on or before to_date.",
        )

    try:
        records = history_service.get_history(station_id, from_date, to_date)
    except Exception as exc:
        log.exception("Failed to load history for station %s: %s", station_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="History lookup failed.",
        ) from exc

    return HistoryResponse(station_id=station_id, records=records)
