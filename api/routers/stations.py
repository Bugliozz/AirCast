"""Router for station-related endpoints."""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from api.schemas import StationOut
from api.services import history

log = logging.getLogger(__name__)

router = APIRouter(tags=["stations"])


@router.get("/stations", response_model=List[StationOut])
def list_stations() -> List[StationOut]:
    """Return all monitoring stations with their coordinates and municipality."""
    try:
        return history.get_stations()
    except Exception as exc:
        log.exception("Failed to load station registry: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Station registry unavailable.",
        ) from exc
