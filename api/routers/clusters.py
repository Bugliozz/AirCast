"""Router for station clustering analysis page and JSON endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.services import clusters as clusters_service
from api.services import history as history_service

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["clusters"])


@router.get("/clusters", response_class=HTMLResponse, name="clusters")
def clusters_page(request: Request) -> HTMLResponse:
    stations = history_service.get_stations()
    cluster_data = clusters_service.get_all_cluster_data(stations)
    return templates.TemplateResponse(
        request,
        "clusters.html",
        {"cluster_data": cluster_data, "active_page": "clusters"},
    )


@router.get("/api/clusters", response_model=None)
def api_clusters() -> list[dict[str, Any]]:
    """JSON endpoint — cluster profiles with station lists."""
    stations = history_service.get_stations()
    data = clusters_service.get_all_cluster_data(stations)
    result = []
    for cluster in data:
        c = {k: v for k, v in cluster.items() if k != "stations"}
        c["stations"] = [
            {
                "idstazione": s.idstazione,
                "nomestazione": s.nomestazione,
                "comune": s.comune,
                "lat": s.lat,
                "lon": s.lon,
            }
            for s in cluster["stations"]
        ]
        result.append(c)
    return result
