"""FastAPI application entry point for the PM10 forecasting API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import AsyncIterator, Dict

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.routers.clusters import router as clusters_router
from api.routers.forecast import router as forecast_router
from api.routers.history import router as history_router
from api.routers.stations import router as stations_router
from api.services import clusters as clusters_service
from api.services import history as history_service
from api.services import map_view, predictor, recent_data

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    predictor.load_models()
    recent_data.prefetch()
    recent_data.start_background_refresh()
    log.info("API startup complete: models, station registry, and recent-data cache loaded.")
    try:
        yield
    finally:
        await recent_data.stop_background_refresh()


app = FastAPI(
    title="PM10 Air Quality Forecast API",
    description="Forecast daily PM10 and alert class for Lombardy monitoring stations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(stations_router)
app.include_router(forecast_router)
app.include_router(history_router)
app.include_router(clusters_router)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, name="home")
def home(request: Request) -> HTMLResponse:
    map_html = map_view.render_alert_map_html()
    return templates.TemplateResponse(
        request,
        "map.html",
        {"map_html": map_html},
    )


@app.get("/forecast-ui", response_class=HTMLResponse, name="forecast_ui")
def forecast_ui_get(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "forecast.html",
        {
            "stations": history_service.get_stations(),
            "result": None,
            "error": None,
            "selected_station": None,
            "selected_days": 1,
            "cluster_info": None,
        },
    )


@app.post("/forecast-ui", response_class=HTMLResponse)
def forecast_ui_post(
    request: Request,
    station_id: str = Form(...),
    days: int = Form(1),
) -> HTMLResponse:
    result = None
    error = None
    if days not in (1, 2):
        error = "The forecast horizon must be 1 or 2 days."
    else:
        try:
            result = predictor.predict(station_id, days=days)
        except Exception as exc:  # noqa: BLE001 - surface to user
            log.exception("Forecast UI failure for station %s", station_id)
            error = f"Forecast unavailable: {exc}"

    return templates.TemplateResponse(
        request,
        "forecast.html",
        {
            "stations": history_service.get_stations(),
            "result": result,
            "error": error,
            "selected_station": station_id,
            "selected_days": days,
            "cluster_info": clusters_service.get_station_cluster_info(station_id),
        },
    )


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


@app.get("/history-ui", response_class=HTMLResponse, name="history_ui")
def history_ui_get(request: Request) -> HTMLResponse:
    latest = history_service.get_latest_date() or date.today()
    default_from = latest - timedelta(days=30)
    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "stations": history_service.get_stations(),
            "chart_data": None,
            "records": None,
            "error": None,
            "selected_station": None,
            "from_date": default_from.isoformat(),
            "to_date": latest.isoformat(),
            "cluster_info": None,
        },
    )


@app.post("/history-ui", response_class=HTMLResponse)
def history_ui_post(
    request: Request,
    station_id: str = Form(...),
    from_date: str = Form(...),
    to_date: str = Form(...),
) -> HTMLResponse:
    chart_data = None
    records = None
    error = None
    try:
        d_from = _parse_iso_date(from_date)
        d_to = _parse_iso_date(to_date)
        if d_from > d_to:
            raise ValueError("The start date must be on or before the end date.")

        records = history_service.get_history(station_id, d_from, d_to)
        if not records:
            error = "No historical data is available for the selected period."
        else:
            stations = {s.idstazione: s for s in history_service.get_stations()}
            station = stations.get(station_id)
            title = (
                f"PM10 - {station.nomestazione} ({station.comune})"
                if station else f"PM10 - station {station_id}"
            )
            chart_data = {
                "title": title,
                "dates": [r.date.isoformat() for r in records],
                "pm10": [r.pm10 for r in records],
            }
    except ValueError as exc:
        error = f"Invalid input: {exc}"
    except Exception as exc:  # noqa: BLE001
        log.exception("History UI failure for station %s", station_id)
        error = f"History unavailable: {exc}"

    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "stations": history_service.get_stations(),
            "chart_data": chart_data,
            "records": records,
            "error": error,
            "selected_station": station_id,
            "from_date": from_date,
            "to_date": to_date,
            "cluster_info": clusters_service.get_station_cluster_info(station_id),
        },
    )
