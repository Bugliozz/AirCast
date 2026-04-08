"""Configuration for the GCloud historical backfill.

All tuneable knobs live here.  Override via environment variables where noted.
"""
from __future__ import annotations

import os

# -- GCS ----------------------------------------------------------------------
GCS_BUCKET: str = os.environ.get("GCS_BUCKET", "exam-project-backfill")
GCS_DATA_PREFIX: str = "data/raw"
GCS_CHECKPOINT_BLOB: str = "checkpoint/backfill_state.json"
GCS_LOCK_BLOB: str = "checkpoint/backfill.lock"

# -- ARPA Lombardia (Socrata) -------------------------------------------------
ARPA_MEASUREMENTS_URL: str = (
    "https://www.dati.lombardia.it/resource/nicp-bhqi.json"
)
ARPA_SENSORS_URL: str = (
    "https://www.dati.lombardia.it/resource/ib47-atvt.json"
)
# Optional Socrata app-token (raises rate limits significantly).
ARPA_APP_TOKEN: str = os.environ.get("ARPA_APP_TOKEN", "")
ARPA_PAGE_SIZE: int = 50_000

# -- Open-Meteo ERA5 Archive --------------------------------------------------
# The *archive* endpoint covers 1940-present (~ 5-day delay).
# The historical-forecast endpoint only starts from Jan 2022, so we MUST use
# the archive for 2016-2021.
OPENMETEO_ARCHIVE_URL: str = (
    "https://archive-api.open-meteo.com/v1/archive"
)
# 'visibility' is NOT available in ERA5; all others match the live collector.
OPENMETEO_HOURLY_VARS: str = ",".join([
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "boundary_layer_height",
])

# -- HTTP / rate-limiting -----------------------------------------------------
REQUEST_TIMEOUT: int = 60
MAX_RETRIES: int = 10
INITIAL_BACKOFF_S: float = 5.0
MAX_BACKOFF_S: float = 600.0        # 10 min ceiling
BACKOFF_MULTIPLIER: float = 2.0
JITTER_FACTOR: float = 0.5          # +/- 50 %

# -- Circuit breaker -----------------------------------------------------------
MAX_CONSECUTIVE_FAILURES: int = 15
CIRCUIT_BREAK_SLEEP_S: int = 0      # exit immediately, next hourly run resumes
MAX_CIRCUIT_BREAKS: int = 1         # exit on first trip, checkpoint preserves progress

# -- Batch / pacing -----------------------------------------------------------
# Max days to process per hourly run (0 = unlimited).
DAYS_PER_BATCH: int = int(os.environ.get("DAYS_PER_BATCH", "20"))
INTER_DAY_SLEEP_S: float = 3.0
INTER_CELL_SLEEP_S: float = 0.3
COORD_ROUND_DP: int = 1
CHECKPOINT_EVERY_N_DAYS: int = 5

# -- Date range (inclusive) ----------------------------------------------------
BACKFILL_START: str = os.environ.get("BACKFILL_START", "2024-01-01")
BACKFILL_END: str = os.environ.get("BACKFILL_END", "2025-04-07")

# -- Lock ----------------------------------------------------------------------
LOCK_TIMEOUT_MINUTES: int = 180     # stale-lock threshold
