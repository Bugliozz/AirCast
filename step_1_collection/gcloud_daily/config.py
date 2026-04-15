"""Configuration for the GCloud daily refresh job.

This job keeps the GCS bucket current for the rolling window of days that
the API needs to compute ``pm10_roll7`` / ``pm10_roll3`` features at
inference time.  It is intentionally separate from the historical
``gcloud_backfill`` module so the two can be deployed and scheduled
independently.
"""
from __future__ import annotations

import os

# -- GCS (shared with gcloud_backfill) ----------------------------------------
GCS_BUCKET: str = os.environ.get("GCS_BUCKET", "exam-project-backfill")
GCS_DATA_PREFIX: str = "data/raw"
GCS_LOCK_BLOB: str = "checkpoint/daily.lock"

# -- ARPA Lombardia (Socrata) -------------------------------------------------
ARPA_MEASUREMENTS_URL: str = (
    "https://www.dati.lombardia.it/resource/nicp-bhqi.json"
)
ARPA_SENSORS_URL: str = (
    "https://www.dati.lombardia.it/resource/ib47-atvt.json"
)
ARPA_APP_TOKEN: str = os.environ.get("ARPA_APP_TOKEN", "")
ARPA_PAGE_SIZE: int = 50_000

# Recent days are not yet flagged ``stato='VA'`` (validated): accept any
# record.  Set to ``True`` to keep the backfill-style validated-only filter.
ARPA_REQUIRE_VALIDATED: bool = os.environ.get(
    "ARPA_REQUIRE_VALIDATED", "false",
).lower() == "true"

# -- Open-Meteo ---------------------------------------------------------------
# ERA5 archive lags ~5 days in its *validated* form but still serves the
# latest days in preliminary form — adequate for lag/rolling features.
OPENMETEO_ARCHIVE_URL: str = (
    "https://archive-api.open-meteo.com/v1/archive"
)
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

# -- HTTP / rate-limiting (same profile as backfill) --------------------------
REQUEST_TIMEOUT: int = 60
MAX_RETRIES: int = 10
INITIAL_BACKOFF_S: float = 5.0
MAX_BACKOFF_S: float = 600.0
BACKOFF_MULTIPLIER: float = 2.0
JITTER_FACTOR: float = 0.5

# -- Circuit breaker ----------------------------------------------------------
MAX_CONSECUTIVE_FAILURES: int = 15
CIRCUIT_BREAK_SLEEP_S: int = 0
MAX_CIRCUIT_BREAKS: int = 1

# -- Pacing -------------------------------------------------------------------
INTER_DAY_SLEEP_S: float = 2.0
INTER_CELL_SLEEP_S: float = 0.3
COORD_ROUND_DP: int = 1

# -- Rolling window ------------------------------------------------------------
# Number of recent days to keep in sync on GCS.  Covers ``pm10_roll7`` at
# inference time (plus one day of safety margin for the ``.shift(1)``).
WINDOW_DAYS: int = int(os.environ.get("WINDOW_DAYS", "7"))

# End of the window, expressed as "today minus N days".  Defaults to 1 so
# we never target an incomplete calendar day.
END_OFFSET_DAYS: int = int(os.environ.get("END_OFFSET_DAYS", "1"))

# Number of tail days in the window to force-refresh (overwrite even if the
# blob already exists).  Covers ARPA preliminary-data consolidation, which
# typically happens within 1-3 days of the measurement date.
FORCE_REFRESH_LAST_N_DAYS: int = int(os.environ.get("FORCE_REFRESH_LAST_N_DAYS", "3"))

# -- Lock ----------------------------------------------------------------------
LOCK_TIMEOUT_MINUTES: int = 60
