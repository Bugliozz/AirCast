"""Server-side Folium map of Lombardy alert forecasts for tomorrow.

The map aggregates a tomorrow prediction for every known station and renders a
colored marker per station. Because producing many forecasts takes several
seconds, the rendered HTML is cached in-memory for 15 minutes.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import folium

from api.services import history, predictor

log = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 15 * 60
_LOMBARDY_CENTER: Tuple[float, float] = (45.65, 9.75)
_DEFAULT_ZOOM = 8

_ALERT_COLORS = {
    "verde": "#2ecc71",
    "giallo": "#f1c40f",
    "arancione": "#e67e22",
    "arancio": "#e67e22",
    "rosso": "#e74c3c",
}

_ALERT_LABELS = {
    "verde": "green",
    "giallo": "yellow",
    "arancione": "orange",
    "arancio": "orange",
    "rosso": "red",
}

_cache_html: Optional[str] = None
_cache_ts: float = 0.0


def _marker_color(alert_class: Optional[str]) -> str:
    if not alert_class:
        return "#6b7c93"
    return _ALERT_COLORS.get(alert_class.lower(), "#6b7c93")


def _alert_label(alert_class: str) -> str:
    return _ALERT_LABELS.get(alert_class.lower(), alert_class)


def _build_map_html() -> str:
    stations = history.get_stations()
    fmap = folium.Map(
        location=list(_LOMBARDY_CENTER),
        zoom_start=_DEFAULT_ZOOM,
        tiles="OpenStreetMap",
    )

    for station in stations:
        alert_class: Optional[str] = None
        pm10_value: Optional[float] = None
        try:
            result = predictor.predict(station.idstazione, days=1)
            first = result["predictions"][0]
            alert_class = first.get("alert_class")
            pm10_value = first.get("pm10_predicted")
        except Exception as exc:  # noqa: BLE001 - skip stations that fail
            log.warning("Forecast failed for station %s: %s", station.idstazione, exc)

        popup_lines = [
            f"<strong>{station.nomestazione}</strong>",
            f"<br/>{station.comune}",
        ]
        if pm10_value is not None:
            popup_lines.append(f"<br/>PM10 tomorrow: <b>{pm10_value:.1f}</b> &mu;g/m&sup3;")
        if alert_class:
            popup_lines.append(f"<br/>Alert: <b>{_alert_label(alert_class)}</b>")

        dq = getattr(station, "data_quality", "ok")
        if dq == "partial":
            border_color, border_weight = "#6b7c93", 3
            popup_lines.append(
                f"<br/><em>Partial data: {station.valid_days_last_7}/7 valid days</em>"
            )
        elif dq == "stale":
            border_color, border_weight = "#e74c3c", 3
            popup_lines.append(
                f"<br/><em>Insufficient data: {station.valid_days_last_7}/7 valid days</em>"
            )
        else:
            border_color, border_weight = _marker_color(alert_class), 1

        folium.CircleMarker(
            location=[station.lat, station.lon],
            radius=7,
            color=border_color,
            fill=True,
            fill_color=_marker_color(alert_class),
            fill_opacity=0.85,
            weight=border_weight,
            popup=folium.Popup("".join(popup_lines), max_width=260),
            tooltip=station.nomestazione,
        ).add_to(fmap)

    return fmap.get_root().render()


def render_alert_map_html(force_refresh: bool = False) -> str:
    """Return cached Folium map HTML rebuilt every ``_CACHE_TTL_SECONDS``."""
    global _cache_html, _cache_ts
    now = time.time()
    if not force_refresh and _cache_html is not None and (now - _cache_ts) < _CACHE_TTL_SECONDS:
        return _cache_html

    log.info("Rebuilding alert map (cache expired or forced).")
    _cache_html = _build_map_html()
    _cache_ts = now
    return _cache_html
