"""Spatial feature engineering: station proximity to industrial zones.

Uses GeoPandas to compute, for each ARPA station:
- dist_industrial_km:       distance to the nearest industrial zone border
- n_industrial_zones_15km:  count of industrial zones within a 15 km radius
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pymysql

log = logging.getLogger(__name__)

# UTM zone 32N — metric CRS for northern Italy
CRS_METRIC = "EPSG:32632"
CRS_WGS84 = "EPSG:4326"


def load_stations_geodf(conn: pymysql.Connection) -> gpd.GeoDataFrame:
    """Load stations from MySQL and return as a GeoDataFrame in EPSG:32632."""
    df = pd.read_sql("SELECT idstazione, lat, lng FROM stations WHERE lat IS NOT NULL AND lng IS NOT NULL", conn)
    if df.empty:
        raise ValueError("No stations with valid coordinates found in MySQL.")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lng"], df["lat"]),
        crs=CRS_WGS84,
    )
    return gdf.to_crs(CRS_METRIC)


def load_industrial_zones(geojson_path: Path) -> gpd.GeoDataFrame:
    """Load industrial zones GeoJSON and reproject to EPSG:32632."""
    if not geojson_path.exists():
        raise FileNotFoundError(
            f"{geojson_path} not found. Run fetch_industrial_zones.py first."
        )

    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise ValueError("Industrial zones GeoJSON contains no features.")

    gdf = gdf.set_crs(CRS_WGS84, allow_override=True)
    return gdf.to_crs(CRS_METRIC)


def compute_industrial_proximity(
    stations_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    radius_km: float = 15.0,
) -> pd.DataFrame:
    """Compute distance to nearest industrial zone and count within radius.

    Both GeoDataFrames must be in a metric CRS (EPSG:32632).

    Returns DataFrame with columns:
        idstazione, dist_industrial_km, n_industrial_zones_15km
    """
    # --- Nearest distance ---
    nearest = gpd.sjoin_nearest(
        stations_gdf[["idstazione", "geometry"]],
        zones_gdf[["geometry"]],
        how="left",
        distance_col="dist_m",
    )
    # sjoin_nearest may duplicate rows if equidistant; keep closest
    nearest = nearest.sort_values("dist_m").drop_duplicates(subset="idstazione", keep="first")
    nearest["dist_industrial_km"] = nearest["dist_m"] / 1_000

    # --- Count within radius ---
    radius_m = radius_km * 1_000
    buffered = stations_gdf[["idstazione", "geometry"]].copy()
    buffered["geometry"] = buffered.geometry.buffer(radius_m)

    within = gpd.sjoin(buffered, zones_gdf[["geometry"]], how="left", predicate="intersects")
    counts = (
        within.groupby("idstazione")["index_right"]
        .apply(lambda s: s.notna().sum())
        .rename("n_industrial_zones_15km")
        .reset_index()
    )

    # --- Merge ---
    result = (
        nearest[["idstazione", "dist_industrial_km"]]
        .merge(counts, on="idstazione", how="left")
    )
    result["n_industrial_zones_15km"] = result["n_industrial_zones_15km"].fillna(0).astype(int)

    log.info(
        "Industrial proximity: %d stations, median dist %.1f km, max zones within %d km: %d.",
        len(result),
        result["dist_industrial_km"].median(),
        radius_km,
        result["n_industrial_zones_15km"].max(),
    )
    return result
