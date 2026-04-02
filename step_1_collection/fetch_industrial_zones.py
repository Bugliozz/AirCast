"""Fetch industrial zone polygons from OpenStreetMap for Lombardia.

Downloads landuse=industrial polygons via the Overpass API and saves
them as a GeoJSON file for downstream spatial analysis.

Usage
-----
python step_1_collection/fetch_industrial_zones.py [--force]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import requests
from shapely.geometry import MultiPolygon, Polygon, mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OVERPASS_QUERY = """
[out:json][timeout:180];
area["name"="Lombardia"]["admin_level"="4"]->.lombardia;
(
  way["landuse"="industrial"](area.lombardia);
  relation["landuse"="industrial"](area.lombardia);
);
out geom;
"""

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def _way_to_polygon(element: dict) -> Polygon | None:
    """Convert an Overpass way element (with geometry) to a Shapely Polygon."""
    geom = element.get("geometry", [])
    if len(geom) < 4:
        return None
    coords = [(node["lon"], node["lat"]) for node in geom]
    return Polygon(coords)


def _relation_to_multipolygon(element: dict) -> MultiPolygon | None:
    """Convert an Overpass relation element to a Shapely MultiPolygon.

    Handles simple cases where members are outer ways with inline geometry.
    """
    outers: list[Polygon] = []
    for member in element.get("members", []):
        if member.get("role") != "outer" or member.get("type") != "way":
            continue
        geom = member.get("geometry", [])
        if len(geom) < 4:
            continue
        coords = [(node["lon"], node["lat"]) for node in geom]
        outers.append(Polygon(coords))

    if not outers:
        return None
    return MultiPolygon(outers) if len(outers) > 1 else MultiPolygon([outers[0]])


def _overpass_to_geojson(data: dict) -> dict:
    """Convert Overpass JSON response to a GeoJSON FeatureCollection."""
    features: list[dict] = []
    skipped = 0

    for element in data.get("elements", []):
        etype = element.get("type")
        tags = element.get("tags", {})
        osm_id = element.get("id")

        if etype == "way":
            geom = _way_to_polygon(element)
        elif etype == "relation":
            geom = _relation_to_multipolygon(element)
        else:
            skipped += 1
            continue

        if geom is None or geom.is_empty:
            skipped += 1
            continue

        features.append({
            "type": "Feature",
            "properties": {
                "osm_id": osm_id,
                "osm_type": etype,
                "name": tags.get("name", ""),
            },
            "geometry": mapping(geom),
        })

    if skipped:
        log.info("Skipped %d elements (invalid geometry or unsupported type).", skipped)

    return {"type": "FeatureCollection", "features": features}


def fetch_industrial_zones(output_dir: Path, force: bool = False) -> Path:
    """Fetch landuse=industrial polygons from OSM Overpass API for Lombardia.

    Returns path to the cached GeoJSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "industrial_zones.geojson"

    if output_path.exists() and not force:
        log.info("Cache found: %s — skipping download (use --force to re-fetch).", output_path)
        return output_path

    log.info("Querying Overpass API for industrial zones in Lombardia…")
    resp = requests.post(
        OVERPASS_URL,
        data={"data": OVERPASS_QUERY},
        timeout=200,
    )
    resp.raise_for_status()
    raw = resp.json()

    n_elements = len(raw.get("elements", []))
    log.info("Overpass returned %d elements.", n_elements)

    geojson = _overpass_to_geojson(raw)
    log.info("Converted to %d GeoJSON features.", len(geojson["features"]))

    output_path.write_text(
        json.dumps(geojson, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Saved: %s", output_path)
    return output_path


def main() -> None:
    force = "--force" in sys.argv
    fetch_industrial_zones(RAW_DIR, force=force)


if __name__ == "__main__":
    main()
