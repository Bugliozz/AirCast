# Piano di Implementazione — Step 5–10

Checklist operativa per tutti i passi ancora da implementare.
Per la descrizione tecnica dettagliata di ogni step, vedi [`technical_doc.md`](technical_doc.md).

---

## Step 5 — Classificazione Allerta (`step_5_classification/`)

### 5.1 Scaffold directory

- [x] Crea `step_5_classification/__init__.py` (vuoto, marca il package Python)
- [x] Crea directory `step_5_classification/artifacts/plots/`

### 5.2 `config.py`

- [x] Definisci `LABEL_MAP = {'verde': 0, 'giallo': 1, 'arancio': 2, 'rosso': 3}`
- [x] Definisci `TARGET_COL = "classe_allerta"` e `DROP_COLS = ["idstazione", "nomestazione", "comune", "pm10"]`
- [x] Definisci `N_CV_SPLITS`, `CV_SCORING = "f1_macro"`, `RANDOM_STATE`, `RANDOM_SEARCH_N_ITER`, `RANDOM_SEARCH_N_JOBS`
- [x] Definisci `ARTIFACTS_DIR`, `PLOTS_DIR`, `METRICS_FILE`
- [x] Definisci `LOGISTIC_PARAM_GRID` con `C ∈ [0.01, 0.1, 1.0, 10.0]`
- [x] Definisci `RANDOM_FOREST_PARAM_DIST` (`n_estimators`, `max_depth`, `min_samples_leaf`)
- [x] Definisci `XGBOOST_PARAM_DIST` (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `min_child_weight`, `colsample_bytree`)

### 5.3 `train.py`

- [x] Carica `daily_dataset_clean.parquet`
- [x] `temporal_train_test_split` con `TARGET_COL` e `DROP_COLS`
- [x] Applica `.map(LABEL_MAP)` su `y_train` e `y_test` (non `OrdinalEncoder`)
- [x] `impute_missing(X_train)` → ottieni `train_medians`; poi `impute_missing(X_test, medians=train_medians)`
- [x] `make_temporal_cv_splits` con `data_giorno` reintrodotta su copia di `X_train`
- [x] Build + fit pipeline **Logistic Regression**: `build_preprocessor(model_type="linear")` → `LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, class_weight="balanced")` → `GridSearchCV(param_grid=LOGISTIC_PARAM_GRID, scoring="f1_macro")`
- [x] Build + fit pipeline **Random Forest Classifier**: `build_preprocessor(model_type="tree")` → `RandomForestClassifier(class_weight="balanced")` → `RandomizedSearchCV(n_iter=50, scoring="f1_macro")`
- [x] Build + fit pipeline **XGBoost Classifier**: `build_preprocessor(model_type="tree")` → `XGBClassifier(objective="multi:softprob", num_class=4)` → `RandomizedSearchCV(n_iter=50, scoring="f1_macro")` con `classifier__sample_weight=compute_sample_weight("balanced", y_train)`
- [x] Selezione best model per `best_score_` (f1_macro CV) tra i tre
- [x] Salva `logistic_regression_best.joblib`, `random_forest_best.joblib`, `xgboost_best.joblib`, `best_model.joblib`

### 5.4 `evaluate.py`

- [x] Carica i tre modelli `.joblib` salvati
- [x] `model.predict(X_test)` per ogni modello (ritorna indici 0–3)
- [x] `classification_report(output_dict=True)` con `target_names=["verde","giallo","arancio","rosso"]` per ogni modello
- [x] Calcola `severe_error_rate = (np.abs(y_pred - y_test) >= 2).mean()` per ogni modello
- [x] Plot confusion matrix heatmap (`seaborn.heatmap` annotata con conteggi) per ogni modello → salva in `artifacts/plots/`
- [x] `permutation_importance(best_model, X_test, y_test, scoring="f1_macro", n_repeats=10)` → plot feature importance → salva in `artifacts/plots/`
- [x] Salva `classification_metrics.json` con f1_macro, per_class, severe_error_rate per ogni modello + `"best_model"` + `"n_test_samples"`

### 5.5 Calibrazione probabilità (opzionale)

- [x] Inizializza `tscv = make_temporal_cv_splits(X_train_con_date, n_splits=5)`
- [x] `CalibratedClassifierCV(best_model, method="isotonic", cv=tscv).fit(X_train, y_train)` (ATTENZIONE: non fittare mai sul test set)
- [x] Salva `best_model_calibrated.joblib`
- [x] Reliability diagram (calibration curve one-vs-rest) prima/dopo per ogni classe

### 5.6 Verifica

- [x] `python -m step_5_classification.train` completa senza errori
- [x] `python -m step_5_classification.evaluate` completa senza errori
- [x] `classification_metrics.json` presente con valori plausibili (f1_macro > 0)
- [x] Tutti i plot di confusion matrix salvati in `artifacts/plots/`

### 5.7 Allineamento scelte progettuali

- [x] Conferma da EDA: `pm10` ha distribuzione fortemente asimmetrica a destra (`skew ~ 1.71`, `kurtosis ~ 5.64`), con code alte e soglie di allerta che tagliano una variabile continua non gaussiana
- [x] Conferma da EDA: il problema non e' lineare; i driver piu' forti sono lag/rolling di PM10 (`pm10_lag1 ~ 0.79`, `pm10_roll3 ~ 0.71`, `pm10_roll7 ~ 0.61`) e variabili meteo/stagionali con effetti non monotoni
- [x] Conferma da EDA: vento, BLH e precipitazione mostrano relazione inversa con PM10; stagione invernale e province/stazioni piu' critiche spostano la distribuzione verso valori piu' alti
- [x] Evidenza modelli regressivi: `XGBoost` e' il best model attuale (`R2 = 0.721`, `RMSE = 9.03`, `MAE = 6.08`), davanti a `RandomForest` (`R2 = 0.693`, `RMSE = 9.47`) e molto sopra `ElasticNet` (`R2 = 0.351`, `RMSE = 13.78`)
- [x] Evidenza modelli regressivi: `ElasticNet` resta baseline utile per interpretabilita' e confronto, ma non e' la scelta migliore per questo dataset perche' non cattura bene soglie, interazioni e picchi
- [x] Evidenza modelli classificativi: `XGBoost` e' il best model attuale per `f1_macro`, ma con margine minimo su `RandomForest` (`0.6413` vs `0.6392`), quindi il vantaggio non e' netto
- [x] Evidenza modelli classificativi: la confusion matrix mostra che gli errori sono soprattutto tra classi adiacenti (`verde/giallo/arancio/rosso`), segnale coerente con un target ordinale piu' che con un multiclass "piatto"
- [x] Evidenza feature importance: `pm10_roll7` domina sia regressione sia classificazione; seguono `o3_mean`, `pressure_mean`, `provincia`, feature di vento e alcune lag meteo
- [x] Evidenza da confronto extra: la pipeline `XGBoost regressione -> soglie classi` produce in verifica rapida `f1_macro ~ 0.6434` e `severe_error_rate ~ 0.0157`, quindi e' competitiva con la classificazione separata e va considerata come baseline strutturale
- [x] Nota decisionale: per uso operativo di allerta non basta ottimizzare solo `f1_macro`; vanno monitorati anche `recall` della classe `rosso` e `severe_error_rate`
- [ ] Esperimento aggiuntivo: implementa benchmark `CatBoostRegressor` e `CatBoostClassifier` con confronto diretto contro `XGBoost` e `RandomForest`
- [ ] Esperimento aggiuntivo: prova gestione nativa delle categoriche (`provincia`, `stagione`) e valuta reintroduzione di `idstazione` come feature categoriale, dato che l'inferenza finale avviene sulle stesse stazioni gia' viste in training
- [ ] Esperimento aggiuntivo: formalizza e salva nei metrics artifact il confronto `regressione -> soglie` vs classificazione separata, includendo `f1_macro`, `severe_error_rate`, `recall_rosso` e confusion matrix
- [ ] Esperimento aggiuntivo: valuta un approccio ordinale o cost-sensitive per la classificazione allerta, coerente con la distanza semantica tra errori adiacenti e errori gravi (`rosso -> verde`)
- [ ] Esperimento aggiuntivo: definisci un criterio finale di scelta modello production-ready che combini `f1_macro`, `recall_rosso` e penalizzazione degli errori con distanza >= 2
- [ ] Documentazione tecnica: riporta in `technical_doc.md`, `summary.ipynb` e `README.md` che la scelta attuale migliore e' `XGBoost`, ma che la classificazione resta area aperta a miglioramento tramite benchmark ordinali / CatBoost / regressione con soglie

---

## Step 5.8 — Daily refresh su GCloud (`step_1_collection/gcloud_daily/`)

Affianca il backfill storico esistente (`gcloud_backfill/`, che resta deployato per estensioni manuali del range) con un secondo Cloud Run Job che mantiene fresca la finestra rolling richiesta dalle lag/rolling feature dell'API.

### 5.8.1 Modulo `step_1_collection/gcloud_daily/`

- [x] `config.py`: variabili env `GCS_BUCKET`, `WINDOW_DAYS=7`, `END_OFFSET_DAYS=1`, `ARPA_REQUIRE_VALIDATED=false` (include giorni preliminari)
- [x] `api_client.py`: variante ARPA senza filtro `stato='VA'` per i giorni recenti non ancora validati
- [x] `rate_limiter.py`: copiato da `gcloud_backfill/` (stesso profilo retry/backoff/circuit-breaker)
- [x] `lock.py`: lock GCS separato (`checkpoint/daily.lock`) — nessun checkpoint perché la finestra e' corta e l'idempotenza e' gia' data dal check file-exists
- [x] `main.py`: costruisce finestra `[today-7, today-1]`, per ogni data verifica presenza di `{date}_measurements.json` e `{date}_weather.json` su GCS, scarica solo i mancanti, esce subito se tutto presente
- [x] `requirements.txt`, `Dockerfile`: allineati a `gcloud_backfill/`
- [x] `deploy.sh`: crea Cloud Run Job `backfill-daily` + Cloud Scheduler `trigger-daily` con cron `0 3 * * *` (Europe/Rome), riusa stesso bucket / repo Artifact Registry / service account del backfill

### 5.8.2 Serving path indipendente dal training (refactor)

La finestra rolling **non** viene piu' materializzata localmente ne' ingerita in MySQL: il serving path la legge direttamente da GCS al volo. Questo elimina il rischio che chiamate API sporadiche creino buchi nello storico di training.

- [x] `api/services/recent_data.py`: scarica i 7 JSON da `gs://<bucket>/data/raw/` in memoria, aggrega orario→daily (PM10 + pollutants + meteo) riusando le stesse regole di `step_3_eda/db.py`, cache TTL 15 min
- [x] `api/services/predictor.py`: `get_lag_features()` ora chiama `fetch_recent_for_station()` (GCS); il parquet `daily_dataset_clean.parquet` resta caricato all'avvio solo come **anagrafica statica** (lat/lng, provincia, industrial proximity)
- [x] `step_2_ingestion/ingest.py`: `DATA_DIRS` ridotto a `[RAW_DIR]`, `data/recent/` non viene piu' letta
- [x] `scripts/smoke_test_predict.py`: end-to-end check (fetch GCS + predict day+1/+2) — verificato con stazione 1264, PM10=42.25/39.27
- [x] `scripts/sync_gcs.py` + `scripts/run_sync_gcs.bat`: lasciati nel repo come utility di backfill manuale, ma **non** fanno piu' parte del flusso runtime

### 5.8.3 Deploy e verifica

- [x] `export GCP_PROJECT_ID=<id>` + `bash step_1_collection/gcloud_daily/deploy.sh`
- [x] `gcloud run jobs execute backfill-daily --region=europe-west1` → termina con "Daily refresh done" e i 7 JSON giornalieri sono presenti su `gs://exam-project-backfill/data/raw/`
- [x] `gcloud scheduler jobs describe trigger-daily --location=europe-west1` mostra schedule `0 3 * * *`
- [x] Serving path verificato: `python -m scripts.smoke_test_predict` scarica la finestra 7gg da GCS, aggrega daily, produce predizioni day+1/+2 senza toccare MySQL ne' `data/recent/`
- [x] Training path isolato: `step_2_ingestion/ingest.py` non legge piu' `data/recent/` → lo storico in MySQL dipende solo dal backfill storico (`data/raw/`)

---

## Step 6 — REST API (`api/`)

### 6.1 Scaffold directory

- [x] Crea `api/__init__.py`
- [x] Crea `api/routers/__init__.py`
- [x] Crea `api/services/__init__.py`

### 6.2 `api/schemas.py`

- [x] Pydantic model `StationOut` (idstazione, nomestazione, comune, lat, lon)
- [x] Pydantic model `WeatherUsed` (temp_mean, wind_speed_mean, boundary_layer_height_mean)
- [x] Pydantic model `ForecastItem` (date, pm10_predicted, alert_class, alert_index, weather_used)
- [x] Pydantic model `ForecastResponse` (station_id, station_name, predictions: list[ForecastItem])
- [x] Pydantic model `HistoryRecord` (date, pm10, alert_class)
- [x] Pydantic model `HistoryResponse` (station_id, records: list[HistoryRecord])

### 6.3 `api/services/weather.py`

- [x] Funzione `fetch_forecast_weather(lat, lng, days)` → chiama Open-Meteo Forecast API (stesso endpoint di `step_1_collection/collector.py` ma con horizon futuro)
- [x] Aggregazione giornaliera: stesse aggregazioni di `build_dataset.py` (mean/sum/min/max per variabile)
- [x] Retry logic: 3 tentativi con backoff esponenziale (riusa costanti da `step_1_collection/collector.py`)

### 6.4 `api/services/predictor.py`

- [x] Carica modelli regression + classification da `artifacts/` **una volta sola al startup** (variabili modulo-level)
- [x] Funzione `get_lag_features(station_id)` → scarica 7gg da GCS via `api/services/recent_data.py`, aggrega orario→daily al volo (no MySQL, no parquet) e aggiunge i metadati statici dal parquet
- [x] Funzione `build_features(station_id, target_date, weather_daily, lag_data)` → costruisce `X_future` con tutte le feature (meteo, lag, rolling, temporali, spaziali)
- [x] Funzione `predict(station_id, days)` → orchestra weather fetch + lag fetch + build_features + predict regression + predict classification → ritorna dict predizione
- [x] Gestione `ModelNotFoundError` se i `.joblib` non sono presenti

### 6.5 `api/routers/stations.py`

- [x] `GET /stations` → query MySQL tabella `stations` → ritorna `list[StationOut]`

### 6.6 `api/routers/forecast.py`

- [x] `GET /forecast` con query params `station_id: str` (required), `days: int = 1`
- [x] Validazione: `days` deve essere 1 o 2 (HTTPException 422 altrimenti)
- [x] Chiama `predictor.predict(station_id, days)` → ritorna `ForecastResponse`

### 6.7 `api/routers/history.py`

- [x] `GET /history` con query params `station_id: str`, `from_date: date`, `to_date: date`
- [x] Query MySQL tabella `measurements` per intervallo date e stazione
- [x] Ritorna `HistoryResponse`

### 6.8 `api/main.py`

- [x] App FastAPI con `title`, `description`, `version`
- [x] `GET /health` → `{"status": "ok"}`
- [x] Include routers: `stations`, `forecast`, `history` (con prefix `/` senza versioning)
- [x] Startup event: carica modelli tramite `predictor.py`, verifica connessione MySQL
- [x] Connessione DB via variabili env `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASS`

### 6.9 Verifica API

- [x] `uvicorn api.main:app --reload` avvia senza errori
- [x] `GET /health` → `{"status": "ok"}`
- [x] `GET /stations` → lista non vuota con schema corretto
- [x] `GET /forecast?station_id=<id>&days=1` → predizione con pm10_predicted e alert_class validi
- [x] `GET /forecast?station_id=<id>&days=2` → predizione giorno +2 (lag1 = predizione giorno +1)
- [x] `GET /history?station_id=<id>&from_date=2026-01-01&to_date=2026-03-31` → dati storici
- [x] Swagger UI accessibile su `http://localhost:8000/docs`

---

## Step 7 — Frontend integrato in FastAPI (`api/templates/`, `api/static/`)

> **Allineamento feedback prof (Lab 4):** niente Streamlit separato. Il frontend viene servito dalla **stessa app FastAPI** via `Jinja2Templates` + `StaticFiles`, così backend + frontend stanno in un singolo container deployato su Cloud Run (un solo `gcloud run deploy`).

### 7.1 Rimozione dipendenza MySQL dal container runtime

Obiettivo: il container Cloud Run non deve parlare con MySQL. MySQL resta SOLO come store locale per la pipeline di training/ingestion (`step_2_ingestion/`).

- [x] `api/services/history.py`: carica `daily_dataset_clean.parquet` all'avvio (già caricato da `predictor.py` come anagrafica — riusare stesso DataFrame modulo-level per non duplicarlo in memoria)
- [x] Funzione `get_history(station_id, from_date, to_date)` → filtra il DataFrame in memoria per `idstazione` e `data_giorno` nel range, ritorna `list[HistoryRecord]` con `pm10` e `alert_class` ricavata dalle soglie
- [x] Funzione `get_stations()` → `df.groupby("idstazione").first()[["nomestazione","comune","lat","lng"]]` → `list[StationOut]`
- [x] Refactor `api/routers/stations.py`: usa `history.get_stations()` invece di MySQL
- [x] Refactor `api/routers/history.py`: usa `history.get_history(...)` invece di MySQL
- [x] Elimina `api/db.py` e rimuovi `pymysql` da `requirements.txt` del container API (resta in requirements training)
- [x] `api/main.py`: rimuovi lifespan hook di connessione MySQL

### 7.2 Templates & static

- [ ] Crea `api/templates/` e `api/static/` (CSS minimale + logo)
- [ ] `api/main.py`: `app.mount("/static", StaticFiles(directory="api/static"))` + `templates = Jinja2Templates(directory="api/templates")`
- [ ] Aggiungi `jinja2` a `requirements.txt`

### 7.3 Pagine UI (Jinja2 templates)

- [ ] `templates/base.html`: layout comune (Bootstrap via CDN, navbar con link Mappa / Previsione / Storico)
- [ ] `templates/map.html` — Mappa allerte Lombardia
  - Route `GET /` (home) → renderizza mappa Folium embeddata come HTML
  - Server-side: carica stazioni + fetch previsione day+1 per tutte (cache 15min), genera mappa Folium con marker colorati per classe allerta, passa `map_html` al template
- [ ] `templates/forecast.html` — Previsione singola stazione
  - Route `GET /forecast-ui` → form con dropdown stazioni + radio 1/2 giorni
  - Route `POST /forecast-ui` → chiama `predictor.predict(station_id, days)`, mostra PM10 grande + badge classe allerta + tabella meteo usate + disclaimer day+2
- [ ] `templates/history.html` — Serie storica
  - Route `GET /history-ui` → form: dropdown stazione + date picker from/to (default ultimi 30gg)
  - Route `POST /history-ui` → chiama `history.get_history(...)`, renderizza grafico Plotly inline con bande soglie allerta

### 7.4 Verifica frontend

- [ ] `uvicorn api.main:app --reload` avvia senza errori, senza MySQL
- [ ] `GET /` → mappa Lombardia con marker colorati
- [ ] `GET /forecast-ui` + submit → previsione con badge allerta
- [ ] `GET /history-ui` + submit → grafico storico con bande
- [ ] Endpoint JSON esistenti (`/stations`, `/forecast`, `/history`) continuano a funzionare

---

## Step 8 — Deploy Cloud Run

> **Allineamento feedback prof:** deploy su GCP Cloud Run (come Lab 4). Niente `docker compose` per produzione: compose resta solo per dev locale della pipeline di training (MySQL + ingestion).

### 8.1 `Dockerfile` (root)

- [ ] Crea `Dockerfile` a root seguendo pattern Lab 4 slide 6
- [ ] `FROM python:3.11-slim`, `WORKDIR /app`
- [ ] `COPY requirements.txt .` + `RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt`
- [ ] `COPY api/ ./api/`, `COPY shared/ ./shared/`, `COPY step_4_regression/ ./step_4_regression/`, `COPY step_5_classification/ ./step_5_classification/`
- [ ] `COPY artifacts/ ./artifacts/` (modelli `.joblib` + `daily_dataset_clean.parquet`)
- [ ] `ENV PORT=8080` + `EXPOSE 8080`
- [ ] `CMD ["sh","-c","uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]`

### 8.2 Configurazione env per Cloud Run

- [ ] `api/config.py`: legge da env `GCS_BUCKET`, `ARTIFACTS_DIR`, `PARQUET_PATH` con default sensati
- [ ] Service account Cloud Run: ruolo `roles/storage.objectViewer` sul bucket `exam-project-backfill` (per fetch 7gg recenti)

### 8.3 Deploy

- [ ] `gcloud run deploy pm10-forecast --source . --region=europe-west1 --max-instances=2 --allow-unauthenticated --memory=1Gi --set-env-vars GCS_BUCKET=exam-project-backfill`
- [ ] Annota URL pubblico restituito (es. `https://pm10-forecast-xxx.a.run.app`)

### 8.4 `compose.yaml` (solo dev locale, NON per deploy)

- [ ] Tieni `step_2_ingestion/compose.yaml` esistente per MySQL locale + pipeline training
- [ ] Aggiungi commento in `README.md`: "compose.yaml serve solo per training pipeline locale; il deploy produzione è su Cloud Run"

### 8.5 Verifica deploy

- [ ] `https://<service-url>/health` → `{"status":"ok"}`
- [ ] `https://<service-url>/` → mappa Lombardia caricata
- [ ] `https://<service-url>/forecast?station_id=<id>&days=1` → JSON previsione
- [ ] `https://<service-url>/docs` → Swagger UI
- [ ] Log Cloud Run: nessun errore di connessione MySQL, fetch GCS OK

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
