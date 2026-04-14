"""Quick smoke test for the serving path.

Verifies:
  1. GCS recent-data fetch + daily aggregation works.
  2. Predictor loads models + parquet (used only for static metadata).
  3. End-to-end predict() returns a response for a real station id.

Usage:
  python scripts/smoke_test_predict.py [STATION_ID]
"""
from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from api.services import predictor
from api.services.recent_data import fetch_recent_window


def main() -> int:
    predictor.load_models()

    window = fetch_recent_window()
    print(f"\n--- Recent window ---\nrows: {len(window)}  "
          f"stations: {window['idstazione'].nunique() if not window.empty else 0}")
    if window.empty:
        print("ERROR: no recent data fetched from GCS.")
        return 2
    print(window.head(3).to_string())

    if len(sys.argv) > 1:
        station_id = sys.argv[1]
    else:
        fs_ids = set(predictor._feature_store["idstazione"].astype(str))
        window_pm10 = window.dropna(subset=["pm10"])
        candidates = [s for s in window_pm10["idstazione"].astype(str).unique() if s in fs_ids]
        if not candidates:
            print("ERROR: no station intersects parquet + recent PM10.")
            return 3
        station_id = candidates[0]
    print(f"\n--- Predicting for station {station_id} ---")
    result = predictor.predict(station_id, days=2)
    for p in result["predictions"]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
