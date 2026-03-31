"""Step 2 — ETL: load raw JSON files into MySQL and Neo4j.

Usage
-----
# Start the stack first:
#   cd step_2_ingestion && docker compose up -d
#   (wait ~30 s for MySQL to be ready)

# Then run from project root:
python step_2_ingestion/ingest.py

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
from neo4j import GraphDatabase

# ──────────────────────────────────────────────────────────────────────────────
# Config  (all overridable via environment variables)
# ──────────────────────────────────────────────────────────────────────────────

MYSQL_HOST     = os.getenv("MYSQL_HOST",     "localhost")
MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB       = os.getenv("MYSQL_DB",       "airquality")
MYSQL_USER     = os.getenv("MYSQL_USER",     "airuser")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "airpass")

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jpass")

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


def upsert_stations(conn: pymysql.Connection, sensors_df: pd.DataFrame) -> int:
    """Upsert one row per unique station into `stations`."""
    cols = ["idstazione", "nomestazione", "provincia", "comune", "zona", "quota", "lat", "lng"]
    station_cols = [c for c in cols if c in sensors_df.columns or c in ("idstazione",)]

    stations = (
        sensors_df[["idstazione"]]
        .assign(
            nomestazione = _safe(sensors_df, "nomestazione"),
            provincia    = _safe(sensors_df, "provincia"),
            comune       = _safe(sensors_df, "comune"),
            zona         = _safe(sensors_df, "zona"),
            quota        = pd.to_numeric(_safe(sensors_df, "quota"), errors="coerce"),
            lat          = pd.to_numeric(_safe(sensors_df, "lat"),   errors="coerce"),
            lng          = pd.to_numeric(_safe(sensors_df, "lng"),   errors="coerce"),
        )
        .drop_duplicates(subset="idstazione")
        .where(pd.notna, None)
    )

    rows = [tuple(row) for row in stations.itertuples(index=False, name=None)]
    sql = """
        INSERT INTO stations (idstazione, nomestazione, provincia, comune, zona, quota, lat, lng)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            nomestazione = VALUES(nomestazione),
            provincia    = VALUES(provincia),
            comune       = VALUES(comune),
            zona         = VALUES(zona),
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
    )[["idsensore", "idstazione", "tiposensore", "unitamisura", "datastart", "datastop"]] \
     .where(pd.notna, None)

    # Rename columns that might be missing
    for col in ["idstazione", "tiposensore", "unitamisura", "datastart", "datastop"]:
        if col not in sensors.columns:
            sensors[col] = None

    rows = [tuple(row) for row in sensors.itertuples(index=False, name=None)]
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
# Neo4j graph
# ──────────────────────────────────────────────────────────────────────────────

def build_neo4j_graph(sensors_df: pd.DataFrame) -> None:
    """Create Station, Province, and Zone nodes with relationships."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    stations = (
        sensors_df[["idstazione"]]
        .assign(
            nomestazione = _safe(sensors_df, "nomestazione"),
            provincia    = _safe(sensors_df, "provincia"),
            zona         = _safe(sensors_df, "zona"),
            quota        = pd.to_numeric(_safe(sensors_df, "quota"),   errors="coerce"),
            lat          = pd.to_numeric(_safe(sensors_df, "lat"),     errors="coerce"),
            lng          = pd.to_numeric(_safe(sensors_df, "lng"),     errors="coerce"),
        )
        .drop_duplicates(subset="idstazione")
        .where(pd.notna, None)
        .to_dict("records")
    )

    with driver.session() as session:
        # Constraints (idempotent)
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Station)  REQUIRE s.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Province) REQUIRE p.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (z:Zone)     REQUIRE z.type IS UNIQUE")

        # Stations
        session.run(
            """
            UNWIND $rows AS r
            MERGE (s:Station {id: r.idstazione})
            SET   s.name     = r.nomestazione,
                  s.lat      = r.lat,
                  s.lng      = r.lng,
                  s.quota    = r.quota
            """,
            rows=stations,
        )
        log.info("Neo4j: upserted %d Station nodes.", len(stations))

        # Provinces + LOCATED_IN relationships
        session.run(
            """
            UNWIND $rows AS r
            WHERE r.provincia IS NOT NULL
            MERGE (p:Province {name: r.provincia})
            WITH p, r
            MATCH (s:Station {id: r.idstazione})
            MERGE (s)-[:LOCATED_IN]->(p)
            """,
            rows=stations,
        )
        log.info("Neo4j: upserted Province nodes and LOCATED_IN edges.")

        # Zones + ZONE_TYPE relationships
        session.run(
            """
            UNWIND $rows AS r
            WHERE r.zona IS NOT NULL
            MERGE (z:Zone {type: r.zona})
            WITH z, r
            MATCH (s:Station {id: r.idstazione})
            MERGE (s)-[:ZONE_TYPE]->(z)
            """,
            rows=stations,
        )
        log.info("Neo4j: upserted Zone nodes and ZONE_TYPE edges.")

        # Industrial spatial influence relationships
        # Creates an edge: (Industriale)-[:INFLUENZA_SU {distanza_km}]->(Station) if distance <= 15 km
        result = session.run(
            """
            MATCH (s1:Station)
            MATCH (s2:Station {zona: 'Industriale'})
            WHERE s1.id <> s2.id AND s1.lat IS NOT NULL AND s2.lat IS NOT NULL
            WITH s1, s2, point.distance(point({latitude: s1.lat, longitude: s1.lng}), point({latitude: s2.lat, longitude: s2.lng})) / 1000.0 AS dist_km
            WHERE dist_km <= 15.0
            MERGE (s2)-[r:INFLUENZA_SU]->(s1)
            SET r.distanza_km = round(dist_km * 10) / 10.0
            RETURN count(r) as influenze
            """
        )
        record = result.single()
        influenze_count = record["influenze"] if record else 0
        log.info("Neo4j: created %d INFLUENZA_SU spatial edges from Industrial zones.", influenze_count)

    driver.close()


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
    finally:
        conn.close()

    # ── 3. Neo4j ─────────────────────────────────────────────────────────────
    try:
        build_neo4j_graph(sensors_df)
    except Exception as exc:
        log.warning("Neo4j ingestion skipped: %s", exc)

    log.info("=== Step 2: Ingestion complete ===")


if __name__ == "__main__":
    main()
