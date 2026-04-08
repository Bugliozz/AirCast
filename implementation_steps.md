# Piano di Implementazione — Step 5–10

Checklist operativa per tutti i passi ancora da implementare.
Per la descrizione tecnica dettagliata di ogni step, vedi [`technical_doc.md`](technical_doc.md).

---

## Step 5 — Classificazione Allerta (`step_5_classification/`)

### 5.1 Scaffold directory

- [ ] Crea `step_5_classification/__init__.py` (vuoto, marca il package Python)
- [ ] Crea directory `step_5_classification/artifacts/plots/`

### 5.2 `config.py`

- [ ] Definisci `LABEL_MAP = {'verde': 0, 'giallo': 1, 'arancio': 2, 'rosso': 3}`
- [ ] Definisci `TARGET_COL = "classe_allerta"` e `DROP_COLS = ["idstazione", "nomestazione", "comune", "pm10"]`
- [ ] Definisci `N_CV_SPLITS`, `CV_SCORING = "f1_macro"`, `RANDOM_STATE`, `RANDOM_SEARCH_N_ITER`, `RANDOM_SEARCH_N_JOBS`
- [ ] Definisci `ARTIFACTS_DIR`, `PLOTS_DIR`, `METRICS_FILE`
- [ ] Definisci `LOGISTIC_PARAM_GRID` con `C ∈ [0.01, 0.1, 1.0, 10.0]`
- [ ] Definisci `RANDOM_FOREST_PARAM_DIST` (`n_estimators`, `max_depth`, `min_samples_leaf`)
- [ ] Definisci `XGBOOST_PARAM_DIST` (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `min_child_weight`, `colsample_bytree`)

### 5.3 `train.py`

- [ ] Carica `daily_dataset_clean.parquet`
- [ ] `temporal_train_test_split` con `TARGET_COL` e `DROP_COLS`
- [ ] Applica `.map(LABEL_MAP)` su `y_train` e `y_test` (non `OrdinalEncoder`)
- [ ] `impute_missing(X_train)` → ottieni `train_medians`; poi `impute_missing(X_test, medians=train_medians)`
- [ ] `make_temporal_cv_splits` con `data_giorno` reintrodotta su copia di `X_train`
- [ ] Build + fit pipeline **Logistic Regression**: `build_preprocessor(model_type="linear")` → `LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, class_weight="balanced")` → `GridSearchCV(param_grid=LOGISTIC_PARAM_GRID, scoring="f1_macro")`
- [ ] Build + fit pipeline **Random Forest Classifier**: `build_preprocessor(model_type="tree")` → `RandomForestClassifier(class_weight="balanced")` → `RandomizedSearchCV(n_iter=50, scoring="f1_macro")`
- [ ] Build + fit pipeline **XGBoost Classifier**: `build_preprocessor(model_type="tree")` → `XGBClassifier(objective="multi:softprob", num_class=4)` → `RandomizedSearchCV(n_iter=50, scoring="f1_macro")` con `classifier__sample_weight=compute_sample_weight("balanced", y_train)`
- [ ] Selezione best model per `best_score_` (f1_macro CV) tra i tre
- [ ] Salva `logistic_regression_best.joblib`, `random_forest_best.joblib`, `xgboost_best.joblib`, `best_model.joblib`

### 5.4 `evaluate.py`

- [ ] Carica i tre modelli `.joblib` salvati
- [ ] `model.predict(X_test)` per ogni modello (ritorna indici 0–3)
- [ ] `classification_report(output_dict=True)` con `target_names=["verde","giallo","arancio","rosso"]` per ogni modello
- [ ] Calcola `severe_error_rate = (np.abs(y_pred - y_test) >= 2).mean()` per ogni modello
- [ ] Plot confusion matrix heatmap (`seaborn.heatmap` annotata con conteggi) per ogni modello → salva in `artifacts/plots/`
- [ ] `permutation_importance(best_model, X_test, y_test, scoring="f1_macro", n_repeats=10)` → plot feature importance → salva in `artifacts/plots/`
- [ ] Salva `classification_metrics.json` con f1_macro, per_class, severe_error_rate per ogni modello + `"best_model"` + `"n_test_samples"`

### 5.5 Calibrazione probabilità (opzionale)

- [ ] `CalibratedClassifierCV(best_model, method="sigmoid", cv="prefit").fit(X_test, y_test)`
- [ ] Salva `best_model_calibrated.joblib`
- [ ] Reliability diagram (calibration curve one-vs-rest) prima/dopo per ogni classe

### 5.6 Verifica

- [ ] `python -m step_5_classification.train` completa senza errori
- [ ] `python -m step_5_classification.evaluate` completa senza errori
- [ ] `classification_metrics.json` presente con valori plausibili (f1_macro > 0)
- [ ] Tutti i plot di confusion matrix salvati in `artifacts/plots/`

---

## Step 6 — REST API (`api/`)

### 6.1 Scaffold directory

- [ ] Crea `api/__init__.py`
- [ ] Crea `api/routers/__init__.py`
- [ ] Crea `api/services/__init__.py`

### 6.2 `api/schemas.py`

- [ ] Pydantic model `StationOut` (idstazione, nomestazione, comune, lat, lon)
- [ ] Pydantic model `WeatherUsed` (temp_mean, wind_speed_mean, boundary_layer_height_mean)
- [ ] Pydantic model `ForecastItem` (date, pm10_predicted, alert_class, alert_index, weather_used)
- [ ] Pydantic model `ForecastResponse` (station_id, station_name, predictions: list[ForecastItem])
- [ ] Pydantic model `HistoryRecord` (date, pm10, alert_class)
- [ ] Pydantic model `HistoryResponse` (station_id, records: list[HistoryRecord])

### 6.3 `api/services/weather.py`

- [ ] Funzione `fetch_forecast_weather(lat, lng, days)` → chiama Open-Meteo Forecast API (stesso endpoint di `step_1_collection/collector.py` ma con horizon futuro)
- [ ] Aggregazione giornaliera: stesse aggregazioni di `build_dataset.py` (mean/sum/min/max per variabile)
- [ ] Retry logic: 3 tentativi con backoff esponenziale (riusa costanti da `step_1_collection/collector.py`)

### 6.4 `api/services/predictor.py`

- [ ] Carica modelli regression + classification da `artifacts/` **una volta sola al startup** (variabili modulo-level)
- [ ] Funzione `get_lag_features(station_id, conn)` → query MySQL ultime 8 righe storiche PM10 per la stazione
- [ ] Funzione `build_features(station_id, target_date, weather_daily, lag_data)` → costruisce `X_future` con tutte le feature (meteo, lag, rolling, temporali, spaziali)
- [ ] Funzione `predict(station_id, days)` → orchestra weather fetch + lag fetch + build_features + predict regression + predict classification → ritorna dict predizione
- [ ] Gestione `ModelNotFoundError` se i `.joblib` non sono presenti

### 6.5 `api/routers/stations.py`

- [ ] `GET /stations` → query MySQL tabella `stations` → ritorna `list[StationOut]`

### 6.6 `api/routers/forecast.py`

- [ ] `GET /forecast` con query params `station_id: str` (required), `days: int = 1`
- [ ] Validazione: `days` deve essere 1 o 2 (HTTPException 422 altrimenti)
- [ ] Chiama `predictor.predict(station_id, days)` → ritorna `ForecastResponse`

### 6.7 `api/routers/history.py`

- [ ] `GET /history` con query params `station_id: str`, `from_date: date`, `to_date: date`
- [ ] Query MySQL tabella `measurements` per intervallo date e stazione
- [ ] Ritorna `HistoryResponse`

### 6.8 `api/main.py`

- [ ] App FastAPI con `title`, `description`, `version`
- [ ] `GET /health` → `{"status": "ok"}`
- [ ] Include routers: `stations`, `forecast`, `history` (con prefix `/` senza versioning)
- [ ] Startup event: carica modelli tramite `predictor.py`, verifica connessione MySQL
- [ ] Connessione DB via variabili env `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASS`

### 6.9 Verifica API

- [ ] `uvicorn api.main:app --reload` avvia senza errori
- [ ] `GET /health` → `{"status": "ok"}`
- [ ] `GET /stations` → lista non vuota con schema corretto
- [ ] `GET /forecast?station_id=<id>&days=1` → predizione con pm10_predicted e alert_class validi
- [ ] `GET /forecast?station_id=<id>&days=2` → predizione giorno +2 (lag1 = predizione giorno +1)
- [ ] `GET /history?station_id=<id>&from_date=2026-01-01&to_date=2026-03-31` → dati storici
- [ ] Swagger UI accessibile su `http://localhost:8000/docs`

---

## Step 7 — Web Interface (`webapp/`)

### 7.1 Scaffold directory

- [ ] Crea `webapp/utils/__init__.py`

### 7.2 `webapp/utils/api_client.py`

- [ ] Classe `APIClient` con `base_url` configurabile da env `API_BASE_URL`
- [ ] Metodo `get_stations()` → `GET /stations`
- [ ] Metodo `get_forecast(station_id, days)` → `GET /forecast`
- [ ] Metodo `get_history(station_id, from_date, to_date)` → `GET /history`
- [ ] Gestione errori HTTP con messaggi user-friendly

### 7.3 `webapp/pages/1_map.py` — Mappa allerte

- [ ] Carica tutte le stazioni via `api_client.get_stations()`
- [ ] Fetch previsione giorno +1 per tutte le stazioni (bulk call a `/forecast`)
- [ ] Mappa Lombardia con `folium` integrata via `st.components.v1.html`
- [ ] Marker per ogni stazione: colore = classe allerta prevista (verde/giallo/arancio/rosso)
- [ ] Popup su click: nome stazione, comune, PM10 previsto, classe allerta
- [ ] Pulsante "Aggiorna previsioni" → ricarica le previsioni

### 7.4 `webapp/pages/2_forecast.py` — Previsione stazione

- [ ] Dropdown: lista stazioni (da `get_stations()`)
- [ ] Slider: orizzonte 1 o 2 giorni
- [ ] Mostra PM10 previsto come numero grande + badge colorato con classe allerta
- [ ] Tabella delle feature meteo usate per la predizione (temp_mean, wind_speed_mean, blh_mean)
- [ ] Disclaimer: "Orizzonte 2 giorni: la lag feature è stimata dal giorno precedente"

### 7.5 `webapp/pages/3_history.py` — Serie storica

- [ ] Dropdown: seleziona stazione
- [ ] Date picker: from/to (default: ultimi 30 giorni)
- [ ] Grafico lineare PM10 nel tempo (`plotly` o `st.line_chart`)
- [ ] Bande orizzontali colorate per soglie allerta (verde <20, giallo <35, arancio <50, rosso ≥50)

### 7.6 `webapp/app.py` — Entry point

- [ ] Configurazione pagina Streamlit (titolo, layout wide, icona)
- [ ] Navigazione multi-pagina tra le 3 pagine

### 7.7 Verifica webapp

- [ ] `streamlit run webapp/app.py` avvia senza errori
- [ ] Mappa carica con marker per ogni stazione
- [ ] Pagina previsione mostra PM10 e classe allerta con colore corretto
- [ ] Pagina storico mostra grafico con bande colorate

---

## Step 8 — Docker & Deploy

### 8.1 `Dockerfile.api`

- [ ] Crea `Dockerfile.api` a root
- [ ] `FROM python:3.11-slim`, `WORKDIR /app`
- [ ] `COPY requirements.txt .` + `RUN pip install --no-cache-dir -r requirements.txt`
- [ ] `COPY api/ ./api/`, `COPY shared/ ./shared/`, `COPY step_4_regression/ ./step_4_regression/`, `COPY step_5_classification/ ./step_5_classification/`
- [ ] `EXPOSE 8000` + `CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]`

### 8.2 `Dockerfile.webapp`

- [ ] Crea `Dockerfile.webapp` a root
- [ ] `FROM python:3.11-slim`, `WORKDIR /app`
- [ ] `COPY requirements.txt .` + install
- [ ] `COPY webapp/ ./webapp/`
- [ ] `EXPOSE 8501` + `CMD ["streamlit", "run", "webapp/app.py", "--server.port=8501", "--server.address=0.0.0.0"]`

### 8.3 `compose.yaml` (root — sostituisce `step_2_ingestion/compose.yaml`)

- [ ] Crea `compose.yaml` a root del progetto
- [ ] Servizio `mysql`: image mysql:8.0, env vars, volume persistente, healthcheck (10 retry)
- [ ] Servizio `api`: build da `Dockerfile.api`, `depends_on: mysql (condition: service_healthy)`, mount artifacts read-only
- [ ] Servizio `webapp`: build da `Dockerfile.webapp`, `depends_on: api`, env `API_BASE_URL: http://api:8000`
- [ ] Volume named `mysql_data`

### 8.4 Verifica Docker

- [ ] `docker compose up --build` completa senza errori
- [ ] Tutti e 3 i servizi in stato `running` / `healthy`
- [ ] `http://localhost:8000/docs` accessibile (Swagger UI)
- [ ] `http://localhost:8501` accessibile (Streamlit)
- [ ] End-to-end: previsione dalla UI → risposta corretta

---

## Step 9 — GitHub Actions CI/CD

### 9.1 Workflow CI (`test.yml`)

- [ ] Crea `.github/workflows/test.yml`
- [ ] Trigger: `push` su `master`/`main` + `pull_request`
- [ ] Job `test`: `ubuntu-latest`, setup Python 3.11, `pip install -r requirements.txt`, `pytest tests/ -v`

### 9.2 Workflow Docker Publish (`docker-publish.yml`)

- [ ] Crea `.github/workflows/docker-publish.yml`
- [ ] Trigger: `push` su `master`/`main` + tag `v*.*.*`
- [ ] Step: login GHCR con `GITHUB_TOKEN`
- [ ] Step: build e push immagine `api` → `ghcr.io/${{ github.repository }}/api:latest`
- [ ] Step: build e push immagine `webapp` → `ghcr.io/${{ github.repository }}/webapp:latest`

### 9.3 Test suite di base (`tests/`)

- [ ] Crea `tests/__init__.py`
- [ ] Test unit: `temporal_train_test_split` — verifica che nessun giorno sia in train e test contemporaneamente
- [ ] Test unit: `make_temporal_cv_splits` — verifica 5 fold, nessun overlap intra-fold
- [ ] Test unit: `impute_missing` — verifica che le mediane del test set NON vengano calcolate sui dati test
- [ ] Test integration: `GET /health` → 200 `{"status": "ok"}`
- [ ] Test integration: `GET /stations` → lista con campi `idstazione`, `lat`, `lon` presenti

### 9.4 Verifica

- [ ] `pytest tests/ -v` passa localmente senza errori
- [ ] Pipeline CI verde su GitHub dopo push

---

## Step 10 — `summary.ipynb`

- [ ] Crea `summary.ipynb` alla root del progetto
- [ ] **Sezione 1 — Introduzione**: cella markdown con obiettivo, dataset usati, pipeline diagram (ASCII da `technical_doc.md` sezione 2)
- [ ] **Sezione 2 — Raccolta Dati**: cella codice con esempio fetch ARPA + Open-Meteo, preview JSON risultante
- [ ] **Sezione 3 — Feature Engineering**: tabelle feature principali (temporali, lag/rolling, meteo derivate, spaziali) + motivazioni encoding ciclico
- [ ] **Sezione 4 — EDA**: carica `daily_dataset.parquet`, mostra i 13 plot da `step_3_eda/plots/` con `IPython.display`
- [ ] **Sezione 5 — Regressione PM10**: carica `step_4_regression/artifacts/regression_metrics.json`, tabella comparativa R²/RMSE/MAE, feature importance XGBoost
- [ ] **Sezione 6 — Classificazione Allerta**: carica `step_5_classification/artifacts/classification_metrics.json`, tabella F1-macro per modello, confusion matrix best model
- [ ] **Sezione 7 — Demo previsione**: chiama `GET /forecast?station_id=501&days=1` via `requests`, mostra output JSON formattato
- [ ] **Sezione 8 — Conclusioni**: tabella limiti del modello (catturabili vs non catturabili), possibili miglioramenti futuri
- [ ] Verifica: notebook eseguibile da cima a fondo senza errori (`Run All`) con API attiva per la sezione demo

---

## Step 11 — `README.md`

- [ ] Crea `README.md` alla root del progetto
- [ ] **Titolo + descrizione** (2–3 righe): cosa fa, quali dati usa, cosa predice
- [ ] **Project Structure**: albero directory semplificato (solo livello 1–2)
- [ ] **Pipeline Overview**: diagramma ASCII da `technical_doc.md` sezione 2
- [ ] **Quick Start — Prerequisites**: Docker & Docker Compose, Python 3.11+
- [ ] **Quick Start — Run with Docker**: `docker compose up --build`, URL API (`http://localhost:8000/docs`) e webapp (`http://localhost:8501`)
- [ ] **Quick Start — Run locally**: comandi step-by-step (MySQL → ingest → build_dataset → train regression → train classification → API → webapp)
- [ ] **API Reference**: tabella endpoint principali (metodo, path, descrizione) + link a `/docs`
- [ ] **Data Sources**: ARPA Lombardia, Open-Meteo, OpenStreetMap
- [ ] **Models**: tabella breve (modello, task, metrica primaria, score ottenuto dal test set)
- [ ] **License**: sezione con licenza del progetto

---

## Ordine di Esecuzione

```
Step 5 (classification)
    ↓
Step 6 (API) + Step 7 (webapp)  ← sviluppabili in parallelo
    ↓
Step 8 (Docker)
    ↓
Step 9 (CI/CD) + Step 10 (notebook) + Step 11 (README)  ← parallelizzabili
```

> Step 10 (notebook) richiede che Step 5, 6 e 8 siano completati (per i `.json` di metriche e l'API attiva).  
> Step 11 (README) richiede che tutti i comandi di esecuzione siano verificati.
