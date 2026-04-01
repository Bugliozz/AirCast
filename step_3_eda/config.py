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
# ──────────────────────────────────────────────────────────────────────────────
PM10_THRESHOLDS: list[float] = [0, 20, 35, 50, float("inf")]
PM10_LABELS: list[str] = ["verde", "giallo", "arancio", "rosso"]

# ──────────────────────────────────────────────────────────────────────────────
# Missing-value threshold
# ──────────────────────────────────────────────────────────────────────────────
MISSING_THRESHOLD = 0.50

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
