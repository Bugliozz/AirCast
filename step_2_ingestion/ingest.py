"""Step 2 — ETL: load raw JSON files into MySQL + compute spatial features.

Usage
-----
# Start the stack first:
#   cd step_2_ingestion && docker compose up -d
#   (wait ~30 s for MySQL to be ready)

# Then run from project root:
python -m step_2_ingestion.ingest

# Or override defaults with env vars:
MYSQL_HOST=localhost MYSQL_PASSWORD=airpass python step_2_ingestion/ingest.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pymysql
import requests

# ──────────────────────────────────────────────────────────────────────────────
# Config  (all overridable via environment variables)
# ──────────────────────────────────────────────────────────────────────────────

MYSQL_HOST     = os.getenv("MYSQL_HOST",     "localhost")
MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB       = os.getenv("MYSQL_DB",       "airquality")
MYSQL_USER     = os.getenv("MYSQL_USER",     "airuser")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "airpass")

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

ARPA_SENSORS_URL = "https://www.dati.lombardia.it/resource/ib47-atvt.json"
ARPA_PAGE_SIZE   = 50_000
REQUEST_TIMEOUT  = 30

MYSQL_BATCH_SIZE = 5_000   # rows per executemany call

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

def fetch_sensor_registry() -> pd.DataFrame:
    """Fetch all sensor metadata from ARPA Lombardia. Caches result to data/raw/."""
    cache_path = RAW_DIR / "sensors_registry.json"
    if cache_path.exists():
        log.info("Loading sensor registry from cache: %s", cache_path)
        return pd.DataFrame(json.loads(cache_path.read_text(encoding="utf-8")))

    log.info("Fetching sensor registry from ARPA API…")
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        resp = requests.get(
            ARPA_SENSORS_URL,
            params={"$limit": ARPA_PAGE_SIZE, "$offset": offset},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        records.extend(page)
        if len(page) < ARPA_PAGE_SIZE:
            break
        offset += ARPA_PAGE_SIZE

    log.info("Fetched %d sensor records.", len(records))
    cache_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# MySQL helpers
# ──────────────────────────────────────────────────────────────────────────────

def connect_mysql(retries: int = 10, delay: float = 3.0) -> pymysql.Connection:
    """Connect to MySQL, retrying until the container is ready."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            conn = pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                db=MYSQL_DB,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                charset="utf8mb4",
                autocommit=False,
            )
            log.info("MySQL connected.")
            return conn
        except pymysql.Error as exc:
            last_exc = exc
            log.warning("MySQL not ready (attempt %d/%d): %s", attempt, retries, exc)
            time.sleep(delay)
    raise RuntimeError("Could not connect to MySQL after retries") from last_exc


def _executemany_batched(
    cursor: pymysql.cursors.Cursor,
    sql: str,
    rows: list[tuple],
) -> int:
    """Run executemany in chunks. Returns total affected rows."""
    total = 0
    for i in range(0, len(rows), MYSQL_BATCH_SIZE):
        batch = rows[i : i + MYSQL_BATCH_SIZE]
        cursor.executemany(sql, batch)
        total += cursor.rowcount
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Load stations & sensors
# ──────────────────────────────────────────────────────────────────────────────

def _safe(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column if present, else a series of None."""
    return df[col] if col in df.columns else pd.Series([None] * len(df), index=df.index)


def _convert_nan_to_none(row: tuple) -> tuple:
    """Convert numpy NaN/pandas NA values to None for MySQL compatibility."""
    return tuple(None if pd.isna(v) else v for v in row)


def upsert_stations(conn: pymysql.Connection, sensors_df: pd.DataFrame) -> int:
    """Upsert one row per unique station into `stations`."""
    stations = (
        sensors_df[["idstazione"]]
        .assign(
            nomestazione = _safe(sensors_df, "nomestazione"),
            provincia    = _safe(sensors_df, "provincia"),
            comune       = _safe(sensors_df, "comune"),
            quota        = pd.to_numeric(_safe(sensors_df, "quota"), errors="coerce"),
            lat          = pd.to_numeric(_safe(sensors_df, "lat"),   errors="coerce"),
            lng          = pd.to_numeric(_safe(sensors_df, "lng"),   errors="coerce"),
        )
        .drop_duplicates(subset="idstazione")
    )

    rows = [_convert_nan_to_none(tuple(row)) for row in stations.itertuples(index=False, name=None)]
    sql = """
        INSERT INTO stations (idstazione, nomestazione, provincia, comune, quota, lat, lng)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            nomestazione = VALUES(nomestazione),
            provincia    = VALUES(provincia),
            comune       = VALUES(comune),
            quota        = VALUES(quota),
            lat          = VALUES(lat),
            lng          = VALUES(lng)
    """
    with conn.cursor() as cur:
        n = _executemany_batched(cur, sql, rows)
    conn.commit()
    log.info("Upserted %d station rows.", len(rows))
    return len(rows)


def upsert_sensors(conn: pymysql.Connection, sensors_df: pd.DataFrame) -> int:
    """Upsert one row per sensor into `sensors`."""
    sensors = sensors_df.assign(
        datastart = pd.to_datetime(_safe(sensors_df, "datastart"), errors="coerce").dt.date,
        datastop  = pd.to_datetime(_safe(sensors_df, "datastop"),  errors="coerce").dt.date,
        tiposensore = _safe(sensors_df, "nometiposensore"),  # Rename: nometiposensore → tiposensore
    )[["idsensore", "idstazione", "tiposensore", "unitamisura", "datastart", "datastop"]]

    # Ensure all required columns exist
    for col in ["idstazione", "tiposensore", "unitamisura", "datastart", "datastop"]:
        if col not in sensors.columns:
            sensors[col] = None

    rows = [_convert_nan_to_none(tuple(row)) for row in sensors.itertuples(index=False, name=None)]
    sql = """
        INSERT INTO sensors (idsensore, idstazione, tiposensore, unitamisura, datastart, datastop)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            idstazione  = VALUES(idstazione),
            tiposensore = VALUES(tiposensore),
            unitamisura = VALUES(unitamisura),
            datastart   = VALUES(datastart),
            datastop    = VALUES(datastop)
    """
    with conn.cursor() as cur:
        inserted = _executemany_batched(cur, sql, rows)
    conn.commit()
    log.info("Upserted %d sensor rows.", inserted)
    return inserted


# ──────────────────────────────────────────────────────────────────────────────
# Load measurements
# ──────────────────────────────────────────────────────────────────────────────

def ingest_measurements(conn: pymysql.Connection) -> int:
    """Insert all *_measurements.json files into `measurements`. Idempotent."""
    files = sorted(RAW_DIR.glob("*_measurements.json"))
    if not files:
        log.warning("No measurement files found in %s", RAW_DIR)
        return 0

    total = 0
    sql = """
        INSERT IGNORE INTO measurements (idsensore, data, valore, stato)
        VALUES (%s, %s, %s, %s)
    """
    for path in files:
        records = json.loads(path.read_text(encoding="utf-8"))
        rows: list[tuple] = []
        for r in records:
            try:
                rows.append((
                    str(r["idsensore"]),
                    r["data"].replace("T", " ").split(".")[0],   # → "YYYY-MM-DD HH:MM:SS"
                    float(r["valore"]) if r.get("valore") not in (None, "") else None,
                    r.get("stato"),
                ))
            except (KeyError, ValueError) as exc:
                log.warning("Skipping malformed measurement record: %s", exc)
                continue

        with conn.cursor() as cur:
            inserted = _executemany_batched(cur, sql, rows)
        conn.commit()
        total += inserted
        log.info("[%s] %d/%d rows inserted.", path.name, inserted, len(rows))

    log.info("Measurements total: %d rows inserted.", total)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Load weather
# ──────────────────────────────────────────────────────────────────────────────

WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "surface_pressure", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "visibility",
    "shortwave_radiation", "boundary_layer_height",
]


def ingest_weather(conn: pymysql.Connection) -> int:
    """Flatten hourly weather JSON and insert into `weather_hourly`. Idempotent."""
    files = sorted(RAW_DIR.glob("*_weather.json"))
    if not files:
        log.warning("No weather files found in %s", RAW_DIR)
        return 0

    total = 0
    sql = f"""
        INSERT IGNORE INTO weather_hourly
            (idstazione, dt, {", ".join(WEATHER_VARS)})
        VALUES
            (%s, %s, {", ".join(["%s"] * len(WEATHER_VARS))})
    """

    for path in files:
        stations = json.loads(path.read_text(encoding="utf-8"))
        rows: list[tuple] = []

        for station in stations:
            idstazione = str(station["idstazione"])
            hourly = station.get("hourly", {})
            times = hourly.get("time", [])

            for i, ts in enumerate(times):
                dt = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
                vals = tuple(
                    hourly.get(var, [None] * len(times))[i]
                    for var in WEATHER_VARS
                )
                rows.append((idstazione, dt, *vals))

        with conn.cursor() as cur:
            inserted = _executemany_batched(cur, sql, rows)
        conn.commit()
        total += inserted
        log.info("[%s] %d/%d rows inserted.", path.name, inserted, len(rows))

    log.info("Weather total: %d rows inserted.", total)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Step 2: Ingestion start ===")
    log.info("Raw data directory: %s", RAW_DIR)

    # ── 1. Sensor registry ────────────────────────────────────────────────────
    sensors_df = fetch_sensor_registry()
    log.info("Registry: %d sensors across %d stations.",
             len(sensors_df),
             sensors_df["idstazione"].nunique() if "idstazione" in sensors_df.columns else "?")

    # ── 2. MySQL ─────────────────────────────────────────────────────────────
    conn = connect_mysql()
    try:
        upsert_stations(conn, sensors_df)
        upsert_sensors(conn, sensors_df)
        ingest_measurements(conn)
        ingest_weather(conn)

        # ── 3. Industrial proximity (spatial features) ───────────────────
        geojson_path = RAW_DIR / "industrial_zones.geojson"
        if geojson_path.exists():
            try:
                from .spatial import (
                    compute_industrial_proximity,
                    load_industrial_zones,
                    load_stations_geodf,
                )

                stations_gdf = load_stations_geodf(conn)
                zones_gdf = load_industrial_zones(geojson_path)
                industrial_df = compute_industrial_proximity(stations_gdf, zones_gdf)
                industrial_df.to_parquet(RAW_DIR / "industrial_proximity.parquet", index=False)
                log.info("Industrial proximity: %d stations computed.", len(industrial_df))
            except Exception as exc:
                log.warning("Industrial proximity computation skipped: %s", exc)
        else:
            log.warning("No industrial_zones.geojson found — run fetch_industrial_zones.py first.")
    finally:
        conn.close()

    log.info("=== Step 2: Ingestion complete ===")


if __name__ == "__main__":
    main()
