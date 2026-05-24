"""Rebuild daily_dataset_clean.parquet from local raw JSON files.

Bypasses MySQL entirely: reads all *_measurements.json + *_weather.json from
data/raw/, optionally downloads the GCS gap first, replicates the db.py
aggregation logic, then runs the full build_dataset feature-engineering
pipeline.

Usage
-----
# Download gap from GCS (default: day after last parquet date → today-15d)
python -m scripts.rebuild_parquet_from_raw

# Supply an explicit window
python -m scripts.rebuild_parquet_from_raw --start 2026-04-02 --end 2026-05-07

# Skip the GCS download (raw files already in data/raw/)
python -m scripts.rebuild_parquet_from_raw --skip-download

Env vars
--------
GCS_BUCKET       default: exam-project-backfill
SAFE_CUTOFF_DAYS default: 15  (data more recent than this is skipped)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PARQUET_DIR = ROOT / "step_3_eda"

GCS_BUCKET = os.environ.get("GCS_BUCKET", "exam-project-backfill")
GCS_DATA_PREFIX = "data/raw"
SAFE_CUTOFF_DAYS = int(os.environ.get("SAFE_CUTOFF_DAYS", "15"))

WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "surface_pressure", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "visibility",
    "shortwave_radiation", "boundary_layer_height",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rebuild_parquet")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", help="First date to download from GCS (YYYY-MM-DD).")
    p.add_argument("--end", help="Last date to download from GCS, inclusive (YYYY-MM-DD).")
    p.add_argument("--skip-download", action="store_true", help="Skip GCS download; assume data/raw/ is up to date.")
    return p.parse_args()


# ── GCS download ──────────────────────────────────────────────────────────────

def _default_gap_window() -> tuple[date, date]:
    parquet_path = PARQUET_DIR / "daily_dataset_clean.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path, columns=["data_giorno"])
        last = pd.to_datetime(df["data_giorno"]).max().date()
        start = last + timedelta(days=1)
    else:
        start = date(2024, 1, 1)
    end = date.today() - timedelta(days=SAFE_CUTOFF_DAYS)
    return start, end


def download_gap_from_gcs(start: date, end: date) -> None:
    from google.cloud import storage as gcs

    bucket = gcs.Client().bucket(GCS_BUCKET)
    window = [
        (start + timedelta(days=i)).isoformat()
        for i in range((end - start).days + 1)
    ]
    log.info("GCS download: %s → %s (%d days)", window[0], window[-1], len(window))
    pulled = missing = 0
    for d in window:
        for suffix in ("measurements", "weather"):
            dest = RAW_DIR / f"{d}_{suffix}.json"
            if dest.exists():
                continue
            blob = bucket.blob(f"{GCS_DATA_PREFIX}/{d}_{suffix}.json")
            if blob.exists():
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(dest.as_posix())
                pulled += 1
                log.info("  + %s", dest.name)
            else:
                missing += 1
                log.warning("  - %s not on GCS", dest.name)
    log.info("GCS done: %d pulled, %d missing.", pulled, missing)


# ── Sensor registry ───────────────────────────────────────────────────────────

def load_sensor_registry() -> pd.DataFrame:
    cache = RAW_DIR / "sensors_registry.json"
    if cache.exists():
        reg = pd.DataFrame(json.loads(cache.read_text(encoding="utf-8")))
        log.info("Sensor registry: %d records from cache.", len(reg))
    else:
        import requests
        log.info("Fetching sensor registry from ARPA API...")
        records: list[dict] = []
        offset = 0
        while True:
            resp = requests.get(
                "https://www.dati.lombardia.it/resource/ib47-atvt.json",
                params={"$limit": 50_000, "$offset": offset},
                timeout=30,
            )
            resp.raise_for_status()
            page = resp.json()
            if not page:
                break
            records.extend(page)
            if len(page) < 50_000:
                break
            offset += 50_000
        cache.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        reg = pd.DataFrame(records)
        log.info("Sensor registry: %d records fetched.", len(reg))

    reg["idsensore"] = reg["idsensore"].astype(str)
    reg["idstazione"] = reg["idstazione"].astype(str)
    for col in ("quota", "lat", "lng"):
        reg[col] = pd.to_numeric(reg.get(col), errors="coerce")
    return reg


# ── Measurements ──────────────────────────────────────────────────────────────

def load_all_measurements(registry: pd.DataFrame) -> pd.DataFrame:
    """Read all *_measurements.json → single DataFrame with sensor type + station."""
    # Build lookup: idsensore → {idstazione, tiposensore}
    sensor_map = (
        registry[["idsensore", "idstazione", "nometiposensore"]]
        .rename(columns={"nometiposensore": "tiposensore"})
        .set_index("idsensore")
    )

    files = sorted(RAW_DIR.glob("*_measurements.json"))
    log.info("Reading %d measurement files...", len(files))

    chunks: list[pd.DataFrame] = []
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not raw:
            continue
        df = pd.DataFrame(raw)
        df["idsensore"] = df["idsensore"].astype(str)
        df["valore"] = pd.to_numeric(df.get("valore"), errors="coerce")
        # Keep only validated positive readings
        mask = df["valore"].notna() & (df["valore"] > 0)
        if "stato" in df.columns:
            mask &= df["stato"] == "VA"
        df = df[mask]
        if df.empty:
            continue
        df["data_giorno"] = pd.to_datetime(df["data"].str[:10])
        chunks.append(df[["idsensore", "data_giorno", "valore"]])

    all_m = pd.concat(chunks, ignore_index=True)
    log.info("Measurement records after filter: %d", len(all_m))

    # Join sensor type + station
    all_m = all_m.join(sensor_map, on="idsensore", how="inner")
    return all_m


# ── PM10 daily ────────────────────────────────────────────────────────────────

def build_pm10_daily(measurements: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    pm10_mask = measurements["tiposensore"].str.upper().str.contains("PM10", na=False)
    pm10 = measurements[pm10_mask].copy()

    pm10_daily = (
        pm10.groupby(["idstazione", "data_giorno"])["valore"]
        .mean()
        .reset_index()
        .rename(columns={"valore": "pm10"})
    )

    station_meta = (
        registry[["idstazione", "nomestazione", "provincia", "comune", "quota", "lat", "lng"]]
        .drop_duplicates("idstazione")
    )
    pm10_daily = pm10_daily.merge(station_meta, on="idstazione", how="left")
    log.info("PM10 daily: %d rows, %d stations.", len(pm10_daily), pm10_daily["idstazione"].nunique())
    return pm10_daily


# ── Pollutants daily ──────────────────────────────────────────────────────────

def build_pollutants_daily(measurements: pd.DataFrame) -> pd.DataFrame:
    tipo_lower = measurements["tiposensore"].str.lower().fillna("")
    is_pollutant = (
        tipo_lower.str.contains("azoto")
        | tipo_lower.str.contains("ozono")
        | tipo_lower.str.contains("carbonio")
        | tipo_lower.str.contains("pm2")
    )
    is_pm10 = tipo_lower.str.contains("pm10")
    pollutants = measurements[is_pollutant & ~is_pm10].copy()

    daily = (
        pollutants.groupby(["idstazione", "tiposensore", "data_giorno"])["valore"]
        .agg(valore_mean="mean", valore_max="max")
        .reset_index()
    )
    log.info("Pollutants daily agg: %d rows.", len(daily))
    return daily


# ── Weather daily ─────────────────────────────────────────────────────────────

def load_all_weather() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*_weather.json"))
    log.info("Reading %d weather files...", len(files))

    chunks: list[pd.DataFrame] = []
    for path in files:
        stations = json.loads(path.read_text(encoding="utf-8"))
        for station in stations:
            idstazione = str(station["idstazione"])
            hourly = station.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                continue
            n = len(times)
            row: dict = {"idstazione": [idstazione] * n, "dt": times}
            for var in WEATHER_VARS:
                row[var] = hourly.get(var, [None] * n)
            chunks.append(pd.DataFrame(row))

    all_hourly = pd.concat(chunks, ignore_index=True)
    all_hourly["dt"] = pd.to_datetime(all_hourly["dt"])
    all_hourly["data_giorno"] = all_hourly["dt"].dt.normalize()
    log.info("Hourly weather records: %d", len(all_hourly))

    weather = (
        all_hourly.groupby(["idstazione", "data_giorno"])
        .agg(
            temp_mean=("temperature_2m",      "mean"),
            temp_min=("temperature_2m",       "min"),
            temp_max=("temperature_2m",       "max"),
            humidity_mean=("relative_humidity_2m", "mean"),
            dewpoint_mean=("dew_point_2m",    "mean"),
            precip_sum=("precipitation",      "sum"),
            pressure_mean=("surface_pressure","mean"),
            cloud_cover_mean=("cloud_cover",  "mean"),
            wind_speed_mean=("wind_speed_10m","mean"),
            wind_speed_max=("wind_speed_10m", "max"),
            visibility_mean=("visibility",    "mean"),
            radiation_mean=("shortwave_radiation", "mean"),
            blh_mean=("boundary_layer_height","mean"),
            blh_min=("boundary_layer_height", "min"),
        )
        .reset_index()
    )

    # Fog hours: T - Td < 2 °C
    if "temperature_2m" in all_hourly.columns and "dew_point_2m" in all_hourly.columns:
        fog = (all_hourly["temperature_2m"] - all_hourly["dew_point_2m"]) < 2
        fog_hours = (
            fog.groupby([all_hourly["idstazione"], all_hourly["data_giorno"]])
            .sum()
            .reset_index()
            .rename(columns={0: "fog_hours"})
        )
        weather = weather.merge(fog_hours, on=["idstazione", "data_giorno"], how="left")

    log.info("Weather daily agg: %d rows.", len(weather))
    return weather


# ── Orchestration ─────────────────────────────────────────────────────────────

def build_and_save() -> None:
    from step_3_eda.build_dataset import (
        _add_alert_class,
        _add_lag_and_rolling_features,
        _add_stagnation_flag,
        _add_stagnation_index,
        _add_temporal_features,
        _drop_redundant_features,
        _fill_date_gaps,
        _pivot_pollutants,
        clean_dataset,
        save_dataset,
    )

    # ── Load ──────────────────────────────────────────────────────────────────
    registry = load_sensor_registry()
    measurements = load_all_measurements(registry)

    pm10 = build_pm10_daily(measurements, registry)
    pollutants_raw = build_pollutants_daily(measurements)
    pollutants = _pivot_pollutants(pollutants_raw)
    weather = load_all_weather()

    # ── Join ──────────────────────────────────────────────────────────────────
    df = _fill_date_gaps(pm10)

    weather["data_giorno"] = pd.to_datetime(weather["data_giorno"])
    df = df.merge(weather, on=["idstazione", "data_giorno"], how="left")

    if not pollutants.empty:
        pollutants["data_giorno"] = pd.to_datetime(pollutants["data_giorno"])
        df = df.merge(pollutants, on=["idstazione", "data_giorno"], how="left")

    industrial_path = RAW_DIR / "industrial_proximity.parquet"
    if industrial_path.exists():
        industrial = pd.read_parquet(industrial_path)
        industrial["idstazione"] = industrial["idstazione"].astype(str)
        df = df.merge(industrial, on="idstazione", how="left")
        log.info("Industrial proximity merged.")
    else:
        log.warning("industrial_proximity.parquet not found — feature skipped.")

    log.info("Joined dataset: %d rows, %d columns.", *df.shape)

    # ── Feature engineering ───────────────────────────────────────────────────
    df = _add_temporal_features(df)
    df = _add_lag_and_rolling_features(df)
    df = _add_alert_class(df)
    df = _add_stagnation_flag(df)
    df = _add_stagnation_index(df)
    df = _drop_redundant_features(df)

    # ── Save raw ──────────────────────────────────────────────────────────────
    save_dataset(df, PARQUET_DIR, "daily_dataset.parquet")

    # ── Clean + save ──────────────────────────────────────────────────────────
    df_clean = clean_dataset(df)
    save_dataset(df_clean, PARQUET_DIR, "daily_dataset_clean.parquet")

    log.info(
        "=== Parquet rebuilt: %d rows, %d stations, %s → %s ===",
        len(df_clean),
        df_clean["idstazione"].nunique(),
        df_clean["data_giorno"].min().date(),
        df_clean["data_giorno"].max().date(),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if not args.skip_download:
        if args.start:
            from datetime import datetime
            start = datetime.strptime(args.start, "%Y-%m-%d").date()
        else:
            start, _ = _default_gap_window()

        if args.end:
            from datetime import datetime
            end = datetime.strptime(args.end, "%Y-%m-%d").date()
        else:
            _, end = _default_gap_window()

        if start > end:
            log.info("No gap to download (start=%s > end=%s). Proceeding with existing files.", start, end)
        else:
            download_gap_from_gcs(start, end)
    else:
        log.info("--skip-download: using existing files in data/raw/")

    build_and_save()


if __name__ == "__main__":
    main()
