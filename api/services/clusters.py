"""Load and cache station clustering artifacts for the API."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from api.config import CLUSTER_INTERPRETATION_PATH, CLUSTERS_CSV_PATH

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Return (station_id → cluster, cluster profiles). Cached on first call."""
    try:
        df = pd.read_csv(CLUSTERS_CSV_PATH, dtype={"idstazione": str})
        station_map: dict[str, int] = dict(zip(df["idstazione"], df["cluster"]))
    except Exception:
        log.exception("Could not load station_clusters.csv — cluster info unavailable.")
        station_map = {}

    try:
        profiles: list[dict[str, Any]] = json.loads(
            Path(CLUSTER_INTERPRETATION_PATH).read_text(encoding="utf-8")
        )
    except Exception:
        log.exception("Could not load cluster_interpretation.json.")
        profiles = []

    return station_map, profiles


def _cluster_label(profile: dict[str, Any], all_profiles: list[dict[str, Any]]) -> tuple[str, str]:
    """Return (human label, Bootstrap badge class) for a cluster profile."""
    max_pm10 = max(p["pm10_mean"] for p in all_profiles)
    if profile["pm10_mean"] == max_pm10:
        return "High PM10 criticality", "danger"
    return "Low PM10 criticality", "success"


def get_station_cluster_info(station_id: str) -> Optional[dict[str, Any]]:
    """Return cluster badge info for a single station, or None if unknown."""
    station_map, profiles = _load()
    cluster = station_map.get(str(station_id))
    if cluster is None or not profiles:
        return None

    profile = next((p for p in profiles if p["cluster"] == cluster), None)
    if profile is None:
        return None

    label, badge_class = _cluster_label(profile, profiles)
    return {
        "cluster": cluster,
        "label": label,
        "badge_class": badge_class,
        "pm10_mean": round(profile["pm10_mean"], 1),
        "pct_critical_days": round(profile["pct_critical_days"] * 100, 1),
    }


def get_all_cluster_data(stations: list[Any]) -> list[dict[str, Any]]:
    """Return full cluster summary for the /clusters page.

    Parameters
    ----------
    stations:
        List of StationOut objects from history_service.get_stations().
    """
    station_map, profiles = _load()
    if not profiles:
        return []

    registry_map = {s.idstazione: s for s in stations}
    sorted_profiles = sorted(profiles, key=lambda p: p["pm10_mean"], reverse=True)

    result = []
    for profile in sorted_profiles:
        cluster_id = profile["cluster"]
        label, badge_class = _cluster_label(profile, profiles)
        stations_in_cluster = sorted(
            [
                registry_map[sid]
                for sid, cid in station_map.items()
                if cid == cluster_id and sid in registry_map
            ],
            key=lambda s: s.nomestazione,
        )
        result.append(
            {
                "cluster": cluster_id,
                "label": label,
                "badge_class": badge_class,
                "n_stations": len(stations_in_cluster),
                "pm10_mean": round(profile["pm10_mean"], 1),
                "pct_critical_days": round(profile["pct_critical_days"] * 100, 1),
                "seasonality_ratio": round(profile["seasonality_ratio"], 2),
                "quota_mean": round(profile["quota"], 0),
                "wind_speed_mean": round(profile["wind_speed_mean"], 1),
                "no2_mean": round(profile["no2_mean"], 1),
                "stations": stations_in_cluster,
            }
        )

    return result
