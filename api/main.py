"""FastAPI application entry point for the PM10 forecasting API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict

from fastapi import FastAPI

from api.routers.forecast import router as forecast_router
from api.routers.history import router as history_router
from api.routers.stations import router as stations_router
from api.services import predictor

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    predictor.load_models()
    log.info("API startup complete: models and station registry loaded.")
    yield


app = FastAPI(
    title="PM10 Air Quality Forecast API",
    description="Forecast daily PM10 and alert class for Lombardy monitoring stations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(stations_router)
app.include_router(forecast_router)
app.include_router(history_router)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
