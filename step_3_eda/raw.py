"""Load selected EDA inputs directly from local raw JSON files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw")
SENSORS_REGISTRY_FILE = "sensors_registry.json"


def load_no2_hourly_from_raw(raw_dir: Path | str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load hourly nitrogen-oxide measurements from local raw JSON files.

    The clean EDA parquet is daily-granularity, while plot 13 needs hourly
    timestamps.  This loader mirrors ``load_no2_hourly`` without requiring the
    MySQL staging database: measurement files provide ``idsensore`` + timestamp,
    and ``sensors_registry.json`` provides sensor type and station mapping.
    """
    raw_path = Path(raw_dir)
    registry_path = raw_path / SENSORS_REGISTRY_FILE
    if not registry_path.exists():
        raise FileNotFoundError(f"Sensor registry not found: {registry_path}")

    registry = pd.DataFrame(json.loads(registry_path.read_text(encoding="utf-8")))
    required_registry_cols = {"idsensore", "idstazione", "nometiposensore"}
    missing_cols = required_registry_cols - set(registry.columns)
    if missing_cols:
        raise ValueError(
            f"{registry_path} is missing required columns: {sorted(missing_cols)}"
        )

    no2_sensors = registry[
        registry["nometiposensore"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("biossido di azoto")
    ][["idsensore", "idstazione"]].copy()
    no2_sensors["idsensore"] = no2_sensors["idsensore"].astype(str)
    no2_sensors["idstazione"] = no2_sensors["idstazione"].astype(str)
    sensor_to_station = dict(
        zip(no2_sensors["idsensore"], no2_sensors["idstazione"])
    )
    target_sensor_ids = set(sensor_to_station)
    if not target_sensor_ids:
        raise ValueError("No NO2 sensors found in sensors_registry.json.")

    frames: list[pd.DataFrame] = []
    measurement_files = sorted(raw_path.glob("*_measurements.json"))
    for path in measurement_files:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not records:
            continue

        day_df = pd.DataFrame(records)
        if not {"idsensore", "data", "valore"}.issubset(day_df.columns):
            log.warning("Skipping malformed measurement file: %s", path)
            continue

        day_df["idsensore"] = day_df["idsensore"].astype(str)
        day_df = day_df[day_df["idsensore"].isin(target_sensor_ids)]
        if day_df.empty:
            continue

        day_df["no2"] = pd.to_numeric(day_df["valore"], errors="coerce")
        day_df = day_df[day_df["no2"].notna() & (day_df["no2"] > 0)]
        if day_df.empty:
            continue

        day_df["idstazione"] = day_df["idsensore"].map(sensor_to_station)
        day_df["dt"] = pd.to_datetime(day_df["data"], errors="coerce")
        day_df = day_df.dropna(subset=["dt", "idstazione"])
        frames.append(day_df[["idstazione", "dt", "no2"]])

    if not frames:
        return pd.DataFrame(columns=["idstazione", "dt", "no2"])

    out = pd.concat(frames, ignore_index=True).sort_values(["idstazione", "dt"])
    log.info("NO2 hourly from raw JSON: %d rows.", len(out))
    return out.reset_index(drop=True)
