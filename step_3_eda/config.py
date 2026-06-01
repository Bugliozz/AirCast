"""Step 3 — EDA configuration constants."""

from __future__ import annotations

import os

# ──────────────────────────────────────────────────────────────────────────────
# MySQL (same env-var pattern as step_2_ingestion)
# ──────────────────────────────────────────────────────────────────────────────
MYSQL_HOST     = os.getenv("MYSQL_HOST",     "localhost")
MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB       = os.getenv("MYSQL_DB",       "airquality")
MYSQL_USER     = os.getenv("MYSQL_USER",     "airuser")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "airpass")

# ──────────────────────────────────────────────────────────────────────────────
# PM10 alert-class thresholds  (EU 2008/50/CE + WHO 2021)
#
# NOTE: these 4 classes are daily severity bins derived from EU/WHO limit
# values (20 = WHO AQG 2021; 35 = intermediate level; 50 = daily EU limit
# not to be exceeded more than 35 times/year). They DO NOT coincide with the
# "operational alerts" of regional anti-smog plans (e.g. Lombardy, DGR
# 449/2018) which trigger after N consecutive days > 50 µg/m³. The instantaneous
# choice makes the target balanced and suitable for supervised classification.
# ──────────────────────────────────────────────────────────────────────────────
PM10_THRESHOLDS: list[float] = [0, 20, 35, 50, float("inf")]
PM10_LABELS: list[str] = ["verde", "giallo", "arancio", "rosso"]

# ──────────────────────────────────────────────────────────────────────────────
# Missing-value threshold
# ──────────────────────────────────────────────────────────────────────────────
MISSING_THRESHOLD = 0.50

# ──────────────────────────────────────────────────────────────────────────────
# Stagnation flag — fixed meteorological thresholds (no data-dependent stats)
# ──────────────────────────────────────────────────────────────────────────────
STAGNATION_PRESSURE_THRESHOLD = 1013.25  # hPa, standard atmosphere
STAGNATION_WIND_THRESHOLD = 1.5          # m/s, Beaufort calm / light air
STAGNATION_BLH_THRESHOLD = 500           # m, low boundary layer height

# Continuous stagnation index — minimum clip values (physically meaningful)
# These avoid division by zero while preserving meteorological interpretability.
# Max stagnation_index = 1 / (0.1 * 10 * 1) = 1.0 (dimensionless, bounded)
STAGNATION_INDEX_WIND_MIN = 0.1   # m/s — absolute calm threshold
STAGNATION_INDEX_BLH_MIN = 10.0   # m — shallowest realistic mixing layer

# ──────────────────────────────────────────────────────────────────────────────
# Plot settings
# ──────────────────────────────────────────────────────────────────────────────
PLOT_DPI = 150
FIGSIZE_WIDE: tuple[int, int] = (14, 6)
FIGSIZE_SQUARE: tuple[int, int] = (10, 8)

ALERT_COLORS: dict[str, str] = {
    "verde":   "#4CAF50",
    "giallo":  "#FFC107",
    "arancio": "#FF9800",
    "rosso":   "#F44336",
}

# ──────────────────────────────────────────────────────────────────────────────
# Season mapping  (meteorological seasons)
# ──────────────────────────────────────────────────────────────────────────────
MONTH_TO_SEASON: dict[int, str] = {
    12: "inverno", 1: "inverno", 2: "inverno",
     3: "primavera", 4: "primavera", 5: "primavera",
     6: "estate", 7: "estate", 8: "estate",
     9: "autunno", 10: "autunno", 11: "autunno",
}
SEASON_ORDER: list[str] = ["inverno", "primavera", "estate", "autunno"]
