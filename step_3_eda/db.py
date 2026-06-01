"""Step 3 — Database access layer.

Provides MySQL connection and query helpers that return Pandas DataFrames.
Reuses the same env-var / retry pattern as step_2_ingestion/ingest.py.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import pymysql

from step_3_eda.config import (
    MYSQL_DB,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Connection
# ──────────────────────────────────────────────────────────────────────────────

def connect_mysql(retries: int = 10, delay: float = 3.0) -> pymysql.Connection:
    """Connect to MySQL with retry logic.

    Raises RuntimeError with a helpful message if the container is down.
    """
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

    raise RuntimeError(
        "Cannot connect to MySQL. "
        "Ensure Docker is running: cd step_2_ingestion && docker compose up -d"
    ) from last_exc


# ──────────────────────────────────────────────────────────────────────────────
# Query helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_stations(conn: pymysql.Connection) -> pd.DataFrame:
    """Load the full stations registry."""
    return pd.read_sql("SELECT * FROM stations", conn)


def load_sensors(conn: pymysql.Connection) -> pd.DataFrame:
    """Load the full sensors registry."""
    return pd.read_sql("SELECT * FROM sensors", conn)


def load_pm10_daily(conn: pymysql.Connection) -> pd.DataFrame:
    """Load daily PM10 values joined with station metadata."""
    sql = """
        SELECT
            s.idstazione,
            st.nomestazione,
            st.provincia,
            st.comune,
            st.quota,
            st.lat,
            st.lng,
            DATE(m.data)  AS data_giorno,
            AVG(m.valore) AS pm10
        FROM measurements m
        JOIN sensors  s  ON m.idsensore  = s.idsensore
        JOIN stations st ON s.idstazione = st.idstazione
        WHERE s.tiposensore LIKE '%PM10%'
          AND m.valore IS NOT NULL
          AND m.valore > 0
        GROUP BY s.idstazione, st.nomestazione, st.provincia, st.comune, st.quota, st.lat, st.lng, DATE(m.data)
        ORDER BY s.idstazione, data_giorno
    """
    df = pd.read_sql(sql, conn, parse_dates=["data_giorno"])
    log.info("PM10 daily: %d rows, %d stations.", len(df), df["idstazione"].nunique())
    return df


def load_weather_daily_agg(conn: pymysql.Connection) -> pd.DataFrame:
    """Load weather data aggregated to daily level per station."""
    sql = """
        SELECT
            idstazione,
            DATE(dt) AS data_giorno,
            AVG(temperature_2m)        AS temp_mean,
            MIN(temperature_2m)        AS temp_min,
            MAX(temperature_2m)        AS temp_max,
            AVG(relative_humidity_2m)  AS humidity_mean,
            AVG(dew_point_2m)          AS dewpoint_mean,
            SUM(precipitation)         AS precip_sum,
            AVG(surface_pressure)      AS pressure_mean,
            AVG(cloud_cover)           AS cloud_cover_mean,
            AVG(wind_speed_10m)        AS wind_speed_mean,
            MAX(wind_speed_10m)        AS wind_speed_max,
            AVG(visibility)            AS visibility_mean,
            AVG(shortwave_radiation)   AS radiation_mean,
            AVG(boundary_layer_height) AS blh_mean,
            MIN(boundary_layer_height) AS blh_min,
            SUM(CASE WHEN (temperature_2m - dew_point_2m) < 2
                     THEN 1 ELSE 0 END) AS fog_hours
        FROM weather_hourly
        GROUP BY idstazione, DATE(dt)
        ORDER BY idstazione, data_giorno
    """
    df = pd.read_sql(sql, conn, parse_dates=["data_giorno"])
    log.info("Weather daily agg: %d rows.", len(df))
    return df


def load_pollutants_daily_agg(conn: pymysql.Connection) -> pd.DataFrame:
    """Load NO2/O3/CO/PM2.5 aggregated to daily means and maxima per station."""
    sql = """
        SELECT
            s.idstazione,
            s.tiposensore,
            DATE(m.data) AS data_giorno,
            AVG(m.valore) AS valore_mean,
            MAX(m.valore) AS valore_max
        FROM measurements m
        JOIN sensors s ON m.idsensore = s.idsensore
        WHERE (   s.tiposensore LIKE '%Azoto%'
               OR s.tiposensore LIKE '%Ozono%'
               OR s.tiposensore LIKE '%Carbonio%'
               OR s.tiposensore LIKE '%PM2%')
          AND m.valore IS NOT NULL
        GROUP BY s.idstazione, s.tiposensore, DATE(m.data)
        ORDER BY s.idstazione, data_giorno
    """
    df = pd.read_sql(sql, conn, parse_dates=["data_giorno"])
    log.info("Pollutants daily agg: %d rows.", len(df))
    return df


def load_no2_hourly(conn: pymysql.Connection) -> pd.DataFrame:
    """Load raw hourly NO2 measurements (for the hourly-profile plot)."""
    sql = """
        SELECT
            s.idstazione,
            m.data      AS dt,
            m.valore    AS no2
        FROM measurements m
        JOIN sensors s ON m.idsensore = s.idsensore
        WHERE LOWER(TRIM(s.tiposensore)) = 'biossido di azoto'
          AND m.valore IS NOT NULL
          AND m.valore > 0
        ORDER BY s.idstazione, m.data
    """
    df = pd.read_sql(sql, conn, parse_dates=["dt"])
    log.info("NO2 hourly: %d rows.", len(df))
    return df
