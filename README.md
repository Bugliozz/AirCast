# AirPulita — PM10 Forecasting for Lombardy

[![Tests](https://github.com/Bugliozz/Data-Science-Exam-Project/actions/workflows/test.yml/badge.svg)](https://github.com/Bugliozz/Data-Science-Exam-Project/actions/workflows/test.yml)
[![Docker Publish](https://github.com/Bugliozz/Data-Science-Exam-Project/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/Bugliozz/Data-Science-Exam-Project/actions/workflows/docker-publish.yml)

AirPulita is an end-to-end data science pipeline that forecasts next-day PM10 concentration and 4-class air-quality alert (`verde`/`giallo`/`arancio`/`rosso`) for the 67 ARPA monitoring stations in Lombardy, Italy. It combines ARPA Lombardia air-quality measurements, Open-Meteo hourly weather and OpenStreetMap industrial-zone proximity (2024-01-01 → 2026-05-07, ~54.6k station-days) to train regression, classification and clustering models, and serves the results through a FastAPI backend with an integrated Jinja2 web UI, containerised and deployed on Google Cloud Run.

---

## Project Structure

```
DataScience_ExamProject/
├── step_1_collection/       # ARPA + Open-Meteo + OSM data collection (CLI + Cloud Run jobs)
│   ├── gcloud_backfill/     # Cloud Run Job: historical backfill + weekly "heal" of preliminary data
│   └── gcloud_daily/        # Cloud Run Job: daily rolling-window refresh
├── step_2_ingestion/        # ETL: JSON blobs → MySQL (training store only)
├── step_3_eda/              # Feature engineering, EDA, RMT diagnostics → daily_dataset_clean.parquet
├── step_4_regression/       # PM10 regression: ElasticNet, RandomForest, XGBoost
├── step_5_classification/   # 4-class alert classification: base classifiers, ordinal, hybrid, calibration
├── step_6_clustering/       # Unsupervised station clustering: KMeans, Agglomerative, DBSCAN
├── api/                     # FastAPI app — REST endpoints + Jinja2 frontend
│   ├── routers/             # stations, forecast, history, clusters
│   ├── services/            # predictor, recent_data (GCS), history, map_view, live_arpa
│   ├── templates/           # map / forecast / history / clusters HTML
│   └── static/              # CSS, assets
├── shared/                  # Cross-step utilities (temporal split, preprocessing, log transform)
├── scripts/                 # Windows .bat pipeline runners + smoke tests
├── tests/                   # pytest unit + integration suite (62 tests)
├── data/raw/                # Immutable raw JSON archive (measurements, weather, sensor registry)
├── summary.ipynb            # End-to-end notebook walkthrough (EDA → models → live demo)
├── technical_report.md      # Full technical write-up
├── Dockerfile                # Single-container build served on Cloud Run
└── requirements.txt
```

---

## Pipeline Overview

```
[ARPA Lombardia API]  [Open-Meteo API]  [OpenStreetMap]
        │                    │                 │
        └────────────────────┴─────────────────┘
                             │
                    Step 1: Collection
                  (JSON blobs → data/raw/)
                             │
                    Step 2: Ingestion
              (JSON → MySQL: measurements,
               weather_hourly, stations, sensors)
               + industrial_proximity.parquet
                             │
                    Step 3: EDA + Feature Engineering
                  (MySQL → daily_dataset_clean.parquet)
                  (RMT spectral diagnostic)
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         Step 4:         Step 5:       Step 6:
       Regression     Classification  Clustering
          │               │              │
     regression_      classification_  clustering_
     metrics.json     metrics.json     metrics.json
          │               │
          └───────────────┘
                    │
               FastAPI App
         (predictor.py loads .joblib)
         (recent_data.py reads GCS)
                    │
           [Google Cloud Run]
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for the containerised serving image, and/or local MySQL during training)
- (optional) A GCP service account with `roles/storage.objectViewer` on the GCS bucket, to enable the live "last 7 days" data window used at inference time

### Run with Docker (serving only)

The image bakes in the pre-trained model artifacts and the cleaned parquet, so it serves forecasts without needing MySQL:

```bash
docker build -t airpulita-pm10 .
docker run -p 8080:8080 -e GCS_BUCKET=exam-project-backfill airpulita-pm10
```

- Web UI (map / forecast / history / clusters): http://localhost:8080/
- Swagger UI: http://localhost:8080/docs

> The API and the web frontend are the **same** FastAPI app on the same port — there is no separate webapp container. `step_2_ingestion/compose.yaml` only spins up MySQL (+ Metabase) for local training and is not used in production.

### Run locally (full pipeline, from raw data to a running API)

```bash
# 1. Local MySQL (training store only)
docker compose -f step_2_ingestion/compose.yaml up -d mysql

# 2. Ingest raw JSON → MySQL
python -m step_2_ingestion.ingest

# 3. Feature engineering + EDA → daily_dataset_clean.parquet
python -m step_3_eda.eda

# 4. Train + evaluate the regression models
python -m step_4_regression.train
python -m step_4_regression.evaluate

# 5. Train + evaluate the classification models (+ probability calibration)
python -m step_5_classification.train
python -m step_5_classification.evaluate
python -m step_5_classification.calibrate

# 6. (optional, parallel — depends only on the parquet) Station clustering
python -m step_6_clustering.profiles
python -m step_6_clustering.cluster
python -m step_6_clustering.evaluate

# 7. Run the API + web UI
uvicorn api.main:app --reload
```

- Web UI: http://localhost:8000/
- Swagger UI: http://localhost:8000/docs

On Windows, `scripts\run_pipeline.bat` runs steps 2–5 end-to-end, and `scripts\run_api.bat` runs step 7 with venv/dependency/artifact checks.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check → `{"status": "ok"}` |
| GET | `/stations` | List all 67 monitoring stations (id, name, municipality, coordinates, data quality) |
| GET | `/forecast?station_id&days` | PM10 + alert-class forecast, 1 or 2 days ahead |
| GET | `/history?station_id&from_date&to_date` | Historical daily PM10 series for a station |
| GET | `/api/clusters` | Station clustering profiles, JSON |
| GET | `/` | Web UI — Lombardy map coloured by predicted alert class |
| GET/POST | `/forecast-ui` | Web UI — single-station forecast form |
| GET/POST | `/history-ui` | Web UI — historical PM10 chart |
| GET | `/clusters` | Web UI — station clustering page |

Full interactive documentation (request/response schemas) is available at `/docs` once the app is running.

---

## Data Sources

| Source | Type | Coverage |
|---|---|---|
| [ARPA Lombardia](https://www.dati.lombardia.it/resource/nicp-bhqi.json) (Socrata) | Air quality measurements (PM10, PM2.5, NO2, O3, CO) | 2024-01-01 → 2026-05-07 |
| ARPA Sensor Registry (`ib47-atvt`) | Station/sensor metadata (coordinates, province, altitude) | Static |
| [Open-Meteo](https://open-meteo.com/) Historical Forecast API | Hourly meteorology (temperature, wind, boundary layer height, precipitation, ...) | 2024-01-01 → 2026-05-07 |
| [OpenStreetMap](https://www.openstreetmap.org/) (Overpass API) | Industrial zone polygons → industrial-proximity feature | Static |

---

## Models

| Task | Model | Primary metric | Test score |
|---|---|---|---|
| Regression — next-day PM10 (µg/m³) | XGBoost | R² / RMSE / MAE | 0.696 / 9.24 / 6.26 |
| Classification — next-day alert class | Hybrid (XGBoost regressor → XGBoost classifier) | F1-macro / severe-error-rate / recall(`rosso`) | 0.636 / 2.3% / 0.68 |
| Clustering — 67 stations | KMeans (k=2) | Silhouette | 0.467 |

The classification model is selected under hard production constraints (`severe_error_rate ≤ 2.5%`, `recall_rosso ≥ 0.65`) evaluated across 7 candidate strategies — see `step_5_classification/artifacts/final_model_selection.json` and `technical_report.md` for the full comparison.

---

## License

This project was developed for an academic Data Science exam. No open-source license is granted — all rights reserved by the author.
