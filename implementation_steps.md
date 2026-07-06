# Piano di Implementazione — Step 5–11

Checklist operativa per tutti i passi ancora da implementare.
Per la descrizione tecnica dettagliata di ogni step, vedi [`technical_doc.md`](technical_doc.md).

---

## Step 3.5 - Random Matrix Theory Diagnostic (`step_3_eda/`)

Integrazione non invasiva nello Step 3: diagnostica spettrale della matrice di correlazione delle feature prima della modellazione ML.

- [x] Crea `step_3_eda/rmt.py` con selezione feature numeriche, standardizzazione, autovalori/autovettori, bounds Marchenko-Pastur
- [x] Escludi target/identificatori (`pm10`, `classe_allerta`, `idstazione`, `nomestazione`, `comune`, `data_giorno`) e ordinali calendario grezzi (`mese`, `giorno_settimana`)
- [x] Aggiungi null empirico `grouped_circular_shift` per stazione, piu' adatto a serie temporali ambientali rispetto allo shuffle iid
- [x] Salva artifact tabellari: `rmt_summary.json`, `rmt_eigenvalues.csv`, `rmt_component_loadings.csv`, `rmt_feature_means.csv`, `rmt_feature_stds.csv`
- [x] Aggiungi plot RMT: `14_rmt_eigenvalue_spectrum.png`, `15_rmt_top_eigenvector_loadings.png`, `16_rmt_empirical_null_comparison.png`
- [x] Integra la diagnostica in `python -m step_3_eda.eda` senza modificare training, classificazione, regressione o API
- [x] Verifica standalone: `python -m step_3_eda.rmt --input step_3_eda/daily_dataset_clean.parquet`

Risultato corrente: 41 feature numeriche selezionate, 9 componenti sopra il bulk Marchenko-Pastur (`lambda+ = 1.056`), ~74.0% varianza spettrale spiegata dalle componenti fuori bulk.

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
- [x] Esperimento aggiuntivo: benchmark CatBoost valutato e rimosso — non aggiunge valore rispetto a XGBoost sul dataset corrente
- [x] Esperimento aggiuntivo: gestione categoriche (`provincia`, `stagione`) tramite `OrdinalEncoder` in `build_preprocessor()` per tutti i modelli ad albero
- [x] Esperimento aggiuntivo: confronto `regressione -> soglie` vs classificazione separata formalizzato in `classification_metrics.json` con chiave `regression_to_class_xgboost` (f1_macro=0.640, severe_error_rate=1.57%, recall_rosso=0.447)
- [x] Esperimento aggiuntivo: approccio ordinale implementato — `OrdinalClassifier` (Frank & Hall 2001) wrappa XGBoost in 3 classificatori binari P(y>k) con per-threshold balanced weights; vedi `step_5_classification/ordinal_classifier.py` e `train_ordinal.py`
- [x] Esperimento aggiuntivo: criterio production-ready definito in `STRATEGIA_CLASSIFICAZIONE_IBRIDA.md` §8 — vincoli hard C1 (severe_error_rate ≤ 2.5%) e C2 (recall_rosso ≥ 0.65); `xgboost_ordinal` e' il primo modello a superarli entrambi
- [x] Documentazione tecnica: aggiornato `technical_doc.md` §17bis con percorso completo (regressor baseline → Fase 1 → Fase 2 → Fase 3 → selezione finale `xgboost_ordinal`, production_ready=true)

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

- [x] Crea `api/templates/` e `api/static/` (CSS minimale + logo)
- [x] `api/main.py`: `app.mount("/static", StaticFiles(directory="api/static"))` + `templates = Jinja2Templates(directory="api/templates")`
- [x] Aggiungi `jinja2` a `requirements.txt`

### 7.3 Pagine UI (Jinja2 templates)

- [x] `templates/base.html`: layout comune (Bootstrap via CDN, navbar con link Mappa / Previsione / Storico)
- [x] `templates/map.html` — Mappa allerte Lombardia
  - Route `GET /` (home) → renderizza mappa Folium embeddata come HTML
  - Server-side: carica stazioni + fetch previsione day+1 per tutte (cache 15min), genera mappa Folium con marker colorati per classe allerta, passa `map_html` al template
- [x] `templates/forecast.html` — Previsione singola stazione
  - Route `GET /forecast-ui` → form con dropdown stazioni + radio 1/2 giorni
  - Route `POST /forecast-ui` → chiama `predictor.predict(station_id, days)`, mostra PM10 grande + badge classe allerta + tabella meteo usate + disclaimer day+2
- [x] `templates/history.html` — Serie storica
  - Route `GET /history-ui` → form: dropdown stazione + date picker from/to (default ultimi 30gg)
  - Route `POST /history-ui` → chiama `history.get_history(...)`, renderizza grafico Plotly inline con bande soglie allerta

### 7.4 Verifica frontend

- [x] `uvicorn api.main:app --reload` avvia senza errori, senza MySQL
- [x] `GET /` → mappa Lombardia con marker colorati
- [x] `GET /forecast-ui` + submit → previsione con badge allerta
- [x] `GET /history-ui` + submit → grafico storico con bande
- [x] Endpoint JSON esistenti (`/stations`, `/forecast`, `/history`) continuano a funzionare

---

## Step 7.5 — Data validation hygiene (stato ARPA + freshness UI)

> **Contesto.** ARPA marca i record recenti come `stato='NA'` con `valore=-9999`; la validazione (`stato='VA'`) arriva ~12 giorni lavorativi dopo. Il daily job scarica i blob una sola volta (skip-if-exists in `gcloud_daily/main.py:137`) e non li riscrive mai: un blob salvato in forma preliminare resta sporco per sempre. Oggi il filtro `valore <= 0` esiste solo per PM10 (`api/services/recent_data.py:124`, `api/services/history.py:28`, `step_3_eda/db.py:90`) e non copre NO2/O3/CO/PM2.5. Soluzione a 5 livelli: sanificazione in ingestion, heal job settimanale, filtro al serving + data_quality, warning in UI, fix startup della finestra.

### 7.5.1 Sanificazione in ingestion (`step_1_collection/gcloud_daily/`)

Obiettivo: il blob su GCS è auto-descrittivo e pulito. Valori sentinel (`-9999`) rimossi prima dell'upload, senza però droppare l'intero record (teniamo `stato`, `idsensore`, `data` per tracciabilità).

- [x] `step_1_collection/gcloud_daily/api_client.py` — subito dopo `fetch_arpa_measurements` ottiene la lista `records`, aggiungi passo di sanificazione: per ogni record converti `valore` a float; se `valore <= 0` o `stato != 'VA'` imposta `valore = None` (JSON `null`). Mantieni tutti gli altri campi.
- [x] `step_1_collection/gcloud_daily/api_client.py:75` — lasciare `ARPA_REQUIRE_VALIDATED=false` come default (vogliamo comunque ingestare il giorno, anche parziale, e la sanificazione sopra si occupa del resto).
- [x] `step_1_collection/gcloud_daily/main.py` — nessuna modifica alla logica skip-if-exists qui: la "guarigione" dei blob è compito dell'heal job (7.5.2).
- [x] **Redeploy**: `bash step_1_collection/gcloud_daily/deploy.sh` → verifica che i blob dei prossimi giorni abbiano `valore: null` al posto di `-9999` per i record `stato='NA'`.

### 7.5.2 Heal job settimanale (`step_1_collection/gcloud_backfill/`)

Obiettivo: ogni sabato notte riscaricare con `stato='VA'` i giorni nella finestra `[today-21, today-15]` (ARPA a quel punto li ha consolidati) e **sovrascrivere** i blob preliminari del bucket.

- [x] `step_1_collection/gcloud_backfill/config.py` — aggiungi `OVERWRITE_EXISTING: bool = os.environ.get("OVERWRITE_EXISTING", "false").lower() == "true"` e `HEAL_WINDOW_START_OFFSET` / `HEAL_WINDOW_END_OFFSET` (default 21 / 15).
- [x] `step_1_collection/gcloud_backfill/main.py` — in heal mode (`OVERWRITE_EXISTING=true`) ignora `BACKFILL_START`/`BACKFILL_END` e il checkpoint, calcola la finestra dinamica `[today-HEAL_WINDOW_START_OFFSET, today-HEAL_WINDOW_END_OFFSET]` e sovrascrive i blob (log "Overwriting existing blob …" in `upload_json`).
- [x] `step_1_collection/gcloud_backfill/api_client.py:76` — verifica che il filtro `stato = 'VA'` sia già presente (lo è). Nessuna modifica.
- [x] `step_1_collection/gcloud_backfill/deploy.sh` — aggiungi un secondo Cloud Run Job `backfill-heal` con env `OVERWRITE_EXISTING=true`, `HEAL_WINDOW_START_OFFSET=21`, `HEAL_WINDOW_END_OFFSET=15`, e un Cloud Scheduler `trigger-heal` con cron `0 2 * * 6` (sabato 02:00 Europe/Rome).
- [x] **Verifica deploy**: `gcloud run jobs execute backfill-heal --region=europe-west1` → nei log deve comparire "Overwriting existing blob ..." per i 7 giorni target; sul bucket i timestamp dei blob `[today-21..today-15]_*.json` si aggiornano. *(Eseguito 2026-04-15: execution `backfill-heal-mzwxh` OK, 7 blob measurements + 7 weather sovrascritti nella finestra 2026-03-25..2026-03-31.)*

### 7.5.3 Filtro serving + computazione `data_quality`

Obiettivo: belt-and-suspenders al read-time (se un blob vecchio non ancora "guarito" arriva sporco, lo puliamo lo stesso) + calcolo di `valid_days_last_7` per stazione.

- [x] `api/services/recent_data.py:124` — espandi il filtro da solo PM10 a tutti gli inquinanti. Sostituisci il blocco `invalid_pm10 = ...` con: maschera `df["valore"] <= 0` → `df.loc[mask, "valore"] = pd.NA` (agisce su PM10, NO2, O3, CO, PM2.5 indistintamente).
- [x] `api/services/recent_data.py` — nuova funzione `compute_data_quality(station_id: str) -> dict` che ritorna `{"valid_days_last_7": int, "data_quality": "ok"|"partial"|"stale", "last_valid_date": date|None}`. Regola: `ok` se `valid_days_last_7 >= 6`, `partial` se `3 <= valid_days_last_7 < 6`, `stale` se `< 3`. Calcolata dalla finestra già cached.
- [x] `api/services/history.py:28` — `_sanitize_pm10` resta com'è (già corretta).
- [x] `step_3_eda/db.py:90-92` — nessuna modifica: il training parquet viene già filtrato `valore > 0` + `stato='VA'` a monte (step_1_collection), resta gold.

### 7.5.4 UI warning (`api/templates/`, `api/routers/`)

Obiettivo: l'utente vede subito se una stazione ha dati parziali.

- [x] `api/schemas.py` — estendi `StationOut` con `data_quality: Literal["ok","partial","stale"]` e `valid_days_last_7: int`.
- [x] `api/services/history.py:56` — `get_stations()`: per ogni stazione chiama `recent_data.compute_data_quality(station_id)` e popola i nuovi campi (cached dalla finestra condivisa, no extra I/O).
- [x] `api/templates/forecast.html` — nel dropdown stazioni aggiungi prefisso `●` colorato (verde/giallo/rosso) + `title`/tooltip che dice es. "Dati recenti parziali (4/7 giorni validi) — previsione meno affidabile".
- [x] `api/templates/history.html` — stesso trattamento del dropdown.
- [x] `api/services/map_view.py` — i marker mappa già mostrano la classe allerta; aggiunto bordo grigio/rosso se `data_quality != "ok"`.
- [x] `api/services/predictor.py:predict` — se `data_quality == "stale"`, solleva `HTTPException(status_code=422, detail="Dati recenti insufficienti per una previsione affidabile (valid_days < 3)")` invece di predire.
- [x] `api/schemas.py:ForecastResponse` — aggiungi `data_quality` anche nella risposta `/forecast` per i client API.

### 7.5.5 Fix startup — prefetch + finestra dal bucket

Obiettivo: eliminare i due bug al boot: cache non prewarmed e finestra calcolata da `date.today()` che può escludere l'ultimo giorno presente nel bucket.

- [x] `api/services/recent_data.py:69-72` — sostituisci `_target_window()` basato su `date.today()` con `_target_window_from_bucket()`: lista i blob `{GCS_DATA_PREFIX}/*_measurements.json`, estrai le date dal nome, ordina, prendi gli ultimi `WINDOW_DAYS`. Fallback al calcolo attuale se il listing fallisce (es. offline dev).
- [x] `api/services/recent_data.py` — nuova funzione `prefetch() -> None` wrapper best-effort di `fetch_recent_window()` che fa `try/except` loggando eventuali errori senza rompere il boot.
- [x] `api/main.py:32-34` — nel `lifespan` aggiungi `recent_data.prefetch()` dopo `predictor.load_models()`. Import `from api.services import recent_data`.
- [ ] **Verifica**: avvia `uvicorn api.main:app --reload`, nei log deve comparire "Recent window aggregated: N rows across M stations" prima della prima richiesta HTTP. La data massima nella finestra deve coincidere con l'ultimo blob su `gs://exam-project-backfill/data/raw/`.

### 7.5.6 Robustezza rolling features ai NaN (`api/services/predictor.py`)

Oggi, alle righe 244-247: `lag1 = pm10_series.iloc[-1]`, `lag2 = pm10_series.iloc[-2]`, `roll7 = pm10_series.tail(7).mean()`, `roll3 = pm10_series.tail(3).mean()`. Con NaN nella finestra `.iloc[-1]` restituisce NaN e il modello fallisce.

- [x] `api/services/predictor.py:242-253` — rimpiazza accesso diretto `iloc[-1]` con helper `_last_valid(series, n=2)` che fa `series.dropna().tail(1).iloc[0]` (o NaN se vuota). Stesso per `lag2`.
- [x] `api/services/predictor.py:246-247` — `roll7`: usa `pm10_series.tail(7).dropna()`; se `len >= 3` → mean, altrimenti NaN. Idem `roll3` con `len >= 2`.
- [x] `api/services/predictor.py:243` — opzionale: `pm10_series = lag_data["pm10"].astype(float).ffill(limit=2)` per riempire buchi singoli isolati (max 2 giorni consecutivi).
- [x] **Verifica**: test automatizzato simulando una finestra con NaN in posizione -1 → `predict()` non solleva e `data_quality` in risposta riflette il degrado.

### 7.5.7 Verifica end-to-end

- [ ] `python -m scripts.smoke_test_predict` con una stazione **ok** → predict normale, `data_quality=="ok"`.
- [ ] Ripeti con una stazione che negli ultimi 7 giorni ha solo 4 record validi → `data_quality=="partial"`, predict funziona.
- [ ] Ripeti con stazione con 1 record valido → 422 con messaggio chiaro.
- [ ] Ispezione blob: `gcloud storage cat gs://exam-project-backfill/data/raw/$(date -d yesterday +%F)_measurements.json | jq '.[] | select(.stato=="NA") | .valore' | head` → deve stampare `null`, non `"-9999"`.
- [ ] Dopo il primo sabato post-deploy: verifica che i blob in `[today-21, today-15]` abbiano `updated` time recente e zero record con `stato='NA'`.

---

## Step 7.6 — Live-first inference (force-refresh coda + fallback Socrata/Umbraco)

> **Nota 2026-04-15:** chiamate Umbraco rimosse: in produzione l'endpoint restituiva sempre PM10 vuoto. Rimossi anche tutti i badge/flag "provisional" da UI e API; il gap-fill Socrata (validato `stato='VA'`) resta come silent fallback in `recent_data.py`. Lo step 7.6.6 di verifica end-to-end relativo ai badge non e' piu' applicabile.

> **Contesto.** Dopo 7.5 i blob nascono puliti e vengono "guariti" settimanalmente, ma tra daily e heal esiste una zona grigia: il daily salta il giorno corrente (`END_OFFSET_DAYS=1`) e skippa i blob già esistenti (`gcloud_daily/main.py:137`), quindi un blob salvato parziale resta bucato finché non parte l'heal. Verifica 2026-04-15: per `idsensore=6918` (PM10 stazione 560) il bucket aveva 34/30/-/-/14 record nei giorni 10–14 aprile, mentre Socrata (`nicp-bhqi`) li aveva tutti (34/30/29/12/14). L'endpoint pubblico Umbraco `GetDatiStazioniRealTime?idStazione=560` risponde in live per D0 ma non sempre con PM10. Soluzione in due tempi: **(A)** il daily riscrive sempre gli ultimi `N` giorni del window; **(B)** l'API fa live-first per i giorni recenti con fallback al bucket, marcando in risposta le righe `provisional`. La UI mostra l'etichetta.
>
> **Parametri di default:**
> - `FORCE_REFRESH_LAST_N_DAYS=3` (A). Il daily gira ogni giorno, quindi ciascun giorno viene riscritto 3 volte prima di "congelarsi". Copre il tipico jitter ARPA (consolidamento entro 1-3 giorni).
> - `END_OFFSET_DAYS=1` lato API **invariato** (finestra finisce a ieri, 7 giorni pieni garantiti). D0 non entra nel window ma viene aggiunto come **overlay opportunistico** solo se Umbraco restituisce PM10: niente rischio di finestre con riga vuota in coda.
> - `_LIVE_FALLBACK_WINDOW_DAYS=7` (= intero window). Socrata viene chiamato solo per i giorni effettivamente bucati nel bucket, quindi costo zero nel caso felice; ma quando un buco c'è, copre anche i giorni 4-7 del window (scoperti dal force-refresh a N=3 e non ancora raggiunti dall'heal a T-15).

### 7.6.1 Daily job: force-refresh coda (`step_1_collection/gcloud_daily/`)

Obiettivo: gli ultimi `N` giorni del window vengono sempre riscaricati e sovrascritti, anche se il blob esiste. I giorni più vecchi del window mantengono lo skip-if-exists (costo API contenuto).

- [x] `step_1_collection/gcloud_daily/config.py` — nuova env `FORCE_REFRESH_LAST_N_DAYS: int = int(os.environ.get("FORCE_REFRESH_LAST_N_DAYS", "3"))`. Documentare: "gli ultimi N giorni del window vengono sempre riscaricati per chiudere i buchi ARPA preliminari".
- [x] `step_1_collection/gcloud_daily/main.py:91` — `missing_dates(dates)` diventa `dates_to_fetch(dates) -> list[str]`: ritorna i giorni mancanti **più** gli ultimi `FORCE_REFRESH_LAST_N_DAYS` del window (deduplicati, preservando ordine cronologico).
- [x] `step_1_collection/gcloud_daily/main.py:127` — `collect_day` perde il check `blob_exists` per measurements e weather: riscrive sempre. Logga `Overwriting {date_str}_measurements.json` quando il blob esisteva già (utile per audit).
- [x] `step_1_collection/gcloud_daily/main.py:186` — `_run()`: rinomina la variabile `missing` → `to_fetch`, aggiorna log line "Missing X/Y" → "Fetching X/Y (force-refresh last N=…)".
- [x] `step_1_collection/gcloud_daily/deploy.sh` — aggiungi `--update-env-vars=FORCE_REFRESH_LAST_N_DAYS=3` al deploy del Cloud Run Job.
- [X] **Verifica deploy**: `gcloud run jobs execute daily-refresh --region=europe-west1` → nei log per i 3 giorni più recenti deve comparire `Overwriting …`; inspect `gcloud storage ls -l gs://exam-project-backfill/data/raw/*_measurements.json | head` → `updated` time fresco sugli ultimi 3 giorni; confronto record count con Socrata per `idsensore=6918` → numeri allineati.

### 7.6.2 Live ARPA client (`api/services/live_arpa.py`, nuovo)

Obiettivo: un modulo piccolo, testabile, senza dipendenze dal bucket. Due funzioni pure che ritornano record nella **stessa shape** dei blob (`[{idsensore, data, valore, stato, ...}]`), così l'aggregator esistente non cambia.

- [x] Nuovo file `api/services/live_arpa.py`.
- [x] `fetch_socrata_day(iso_date: str, target_sensor_ids: set[str], timeout_s: float = 10.0) -> list[dict]`:
  - Chiama `https://www.dati.lombardia.it/resource/nicp-bhqi.json` con `$where=data >= 'YYYY-MM-DDT00:00:00.000' AND data <= 'YYYY-MM-DDT23:59:59.999' AND stato='VA'` (solo validati), `$limit=50000`.
  - Filtra `idsensore in target_sensor_ids`.
  - Applica la stessa sanificazione di `gcloud_daily/api_client.py:_sanitize_arpa_records` (sentinel → `None`).
  - Swallow errori HTTP, ritorna `[]` + log warning.
- [x] `fetch_umbraco_today(target_station_ids: set[str], timeout_s: float = 5.0) -> list[dict]`:
  - Per ogni `idstazione` chiama `https://www.arpalombardia.it/umbraco/dettaglio/cosmosdb/GetDatiStazioniRealTime?idStazione={id}`.
  - Parse dei campi PM10 (se presenti), wrappa in record `{idsensore, data, valore, stato: "NA"}` — sentinel `stato='NA'` perché è ufficialmente preliminare.
  - Timeout aggressivo (5s) e `concurrent.futures.ThreadPoolExecutor(max_workers=4)` per parallelizzare. Qualsiasi stazione fallita viene silenziosamente droppata.
- [x] Unit test `tests/test_live_arpa.py` con `requests-mock` o `httpx.MockTransport`: verifica shape output, sanificazione sentinel, behaviour su 500/timeout.

### 7.6.3 Recent-data hybrid loader (`api/services/recent_data.py`)

Obiettivo: merge bucket + live con tracciamento `provisional`. Nessuna modifica alle signature pubbliche esistenti, solo un nuovo metodo + un campo aggiuntivo nel cache entry.

- [x] `api/services/recent_data.py:33` — **lascia `END_OFFSET_DAYS=1` invariato** (finestra 7 giorni ending yesterday, niente regressione).
- [x] `api/services/recent_data.py:41` — estendi `_cache` type a `Dict[str, Tuple[float, pd.DataFrame, frozenset[str]]]` (aggiunge `provisional_dates`).
- [x] Nuova costante `_LIVE_FALLBACK_WINDOW_DAYS = 7` (= intero window): il fallback Socrata può coprire qualsiasi giorno bucato del window, non solo la coda.
- [x] `fetch_recent_window()`:
  - Per ogni `d` in window, scarica il blob come oggi. Raccogli `bucket_dates_with_pm10: set[str]` controllando se il frame per quel giorno contiene almeno una riga PM10 non-null.
  - `gap_dates = set(window) - bucket_dates_with_pm10`. Per ogni `d in gap_dates`: chiama `live_arpa.fetch_socrata_day(d, target_sensor_ids)` e aggrega nello stesso `pm10_frames`. Aggiungi `d` a `provisional_dates`.
  - **D0 overlay opportunistico** (fuori dal window ufficiale): tenta `live_arpa.fetch_umbraco_today(target_station_ids)`. Se ritorna almeno un record PM10 valido, aggrega come riga extra con `data_giorno=today`, aggiungi `today.isoformat()` a `provisional_dates`. Se vuoto/errore, silently skip (la finestra resta di 7 giorni ending yesterday, identico al comportamento attuale).
  - Ritorna il DataFrame; salva `(timestamp, df, frozenset(provisional_dates))` in cache.
- [x] Nuova funzione `get_provisional_dates() -> frozenset[str]`: rilegge dal cache entry più recente (stessa chiave calcolata), ritorna `frozenset()` se miss.
- [x] `compute_data_quality(station_id)` — aggiungi campo `provisional_days: int` (numero di date della finestra della stazione che sono in `provisional_dates`), senza alterare la logica esistente di `ok/partial/stale`.

### 7.6.4 Schemas + predictor (`api/schemas.py`, `api/services/predictor.py`)

Obiettivo: propagare `provisional` nella risposta `/forecast` per-riga (una previsione è provisional se **almeno un lag/roll usato** viene da una data provisional).

- [x] `api/schemas.py:ForecastItem` — aggiungi `provisional: bool = False` e `provisional_reason: Optional[str] = None` (es. `"lag1 from live ARPA (2026-04-15)"`).
- [x] `api/schemas.py:ForecastResponse` — aggiungi `provisional: bool = False` (true se almeno una `ForecastItem` è provisional) e `provisional_days: int = 0`.
- [x] `api/services/predictor.py:predict` — dopo aver calcolato i lag/roll:
  - Recupera `provisional_dates = recent_data.get_provisional_dates()`.
  - Per ogni `ForecastItem` calcola `dates_used = {data di lag1, lag2, ultime 7/3 della finestra usate per roll}` (quelle effettivamente finite nelle feature).
  - `item.provisional = bool(dates_used & provisional_dates)`.
  - `item.provisional_reason` sintetica se true.
  - A livello di risposta: `response.provisional = any(i.provisional for i in items)`, `response.provisional_days = len(dates_used & provisional_dates aggregato)`.
- [x] Unit test `tests/test_predictor_provisional.py`: mock di `recent_data.get_provisional_dates()` → verifica che il flag si propaghi correttamente.

### 7.6.5 UI — badge "dati in tempo reale"

Obiettivo: chi usa la web app capisce a colpo d'occhio se la previsione poggia su dati consolidati o su dati live (meno affidabili).

- [x] `api/templates/forecast.html` — sotto la card della previsione, se `response.provisional` è true, mostra badge giallo "⚡ Previsione basata in parte su dati live (non ancora consolidati)". Tooltip: elenca le date provisional (`response.provisional_days` giorni).
- [x] `api/templates/forecast.html` — per ogni `ForecastItem.provisional=true` aggiungi una piccola icona `⚡` accanto al valore PM10 previsto, con `title={{ item.provisional_reason }}`.
- [x] `api/templates/history.html` — nella tabella storico, se una riga corrisponde a una `provisional_date` (esposta via nuovo campo opzionale in `HistoryRecord.provisional: bool = False`), stile corsivo + tooltip "Dato preliminare, in attesa di validazione ARPA".
- [x] `api/services/history.py` — quando compone i record, consulta `recent_data.get_provisional_dates()` per settare `HistoryRecord.provisional` sulle date recenti.
- [x] `api/schemas.py:HistoryRecord` — aggiungi `provisional: bool = False`.
- [x] `api/static/` (CSS) — classe `.provisional-badge` (giallo) e `.provisional-row` (corsivo + colore attenuato).
- [x] `api/services/map_view.py` — se la stazione selezionata ha `provisional_days > 0`, il marker guadagna un piccolo indicatore `⚡`. Nessun cambio di colore allerta (resta il predicted).

### 7.6.6 Verifica end-to-end

- [ ] **Daily job**: `gcloud run jobs execute daily-refresh` → log mostra `Overwriting …` per gli ultimi 3 giorni; record count su GCS per `idsensore=6918` allineato a Socrata.
- [ ] **API con bucket completo**: forza cache eviction, chiama `GET /forecast/{station_id}` → `response.provisional == false` (fallback live non attivato).
- [ ] **API con bucket bucato**: cancella manualmente un blob recente (`gcloud storage rm gs://…/2026-04-13_measurements.json`), evict cache, richiama forecast → Socrata fallback attiva, `provisional == true`, `provisional_reason` menziona la data.
- [ ] **API con D0**: con `END_OFFSET_DAYS=0` la finestra include oggi → se Umbraco risponde, D0 contribuisce al lag1 e la risposta è `provisional`. Se Umbraco fallisce, la risposta resta valida usando fino a ieri (degradazione graceful).
- [ ] **UI**: apri `http://localhost:8000/`, seleziona stazione con bucket bucato → badge ⚡ visibile nella card previsione, tooltip corretto, pagina `/history/{id}` mostra righe provisional in corsivo.
- [ ] **Regressione 7.5**: `data_quality` resta coerente; una stazione senza dati né su bucket né su Socrata continua a tornare `stale` + HTTP 422 dalla `/forecast`.

---

## Step 7.7 — Clustering Stazioni PM10 (`step_6_clustering/`)

> **Scope.** Clustering **non supervisionato delle 67 stazioni** (unità di analisi = stazione, non giorno-stazione). Ogni stazione è ridotta a un profilo numerico aggregato su tutti i giorni; l'obiettivo è scoprire *tipologie di stazioni* in Lombardia (es. urbano-industriale cronico / pianura intermedia / pedemontano-alpino pulito) e visualizzarle su mappa.
>
> **Numerazione.** La directory è `step_6_clustering/` per continuità con la pipeline ML (`step_4_regression` → `step_5_classification` → `step_6_clustering`). La sezione del doc è "7.7" perché le sezioni 6/7 sono già API/frontend e questo step è collocato qui — dopo il serving, prima del deploy — come richiesto nell'ordine di build. Dipende **solo** dal parquet, quindi è sviluppabile in parallelo a Step 6/7.
>
> **Decisioni progettuali (con motivazione).**
> - **O3 escluso dai profili**: solo 37/67 stazioni hanno dati O3 → includerlo imporrebbe un valore imputato sul 45% delle stazioni. `no2_mean` mantenuto (66/67, copertura ~99%).
> - **N=67 è piccolo**: `silhouette` e DBSCAN/OPTICS sono rumorosi su così pochi punti → KMeans + Agglomerative sono i candidati primari; DBSCAN entra solo come confronto, non come modello finale atteso.
> - **Circolarità dichiarata**: i profili usano feature derivate da PM10 (media, % giorni critici, stagionalità) → validare i cluster contro `classe_allerta` **non è indipendente** (entrambi derivano da PM10). La validazione esterna primaria usa `provincia` (geografia, indipendente) + separazione su severità continua.
> - **Label "classe dominante" scartata come validazione**: la classe dominante per stazione è `verde` (50), `giallo` (17), `arancio`/`rosso` (0) → collassa a binario sbilanciato, poco informativo per ARI/homogeneity.
> - **RobustScaler** (mediana/IQR) come default: robusto a poche stazioni outlier urbano-industriali con PM10/NO2 estremi che altrimenti dominerebbero la distanza.

### 7.7.1 Scaffold directory

- [x] Crea `step_6_clustering/__init__.py` (vuoto, marca il package)
- [x] Crea directory `step_6_clustering/artifacts/` e `step_6_clustering/artifacts/plots/`

### 7.7.2 `config.py`

- [x] `PROFILE_FEATURES` (le 8 feature di clustering, lean per N=67): `["pm10_mean", "pct_critical_days", "seasonality_ratio", "stagnation_index_mean", "dist_industrial_km", "no2_mean", "quota", "wind_speed_mean"]`
- [x] `META_COLS = ["idstazione", "nomestazione", "provincia", "comune", "lat", "lng"]` (tenute per mappa/validazione, **non** usate come feature)
- [x] `EXTERNAL_LABEL_COL = "provincia"` (label per validazione esterna indipendente)
- [x] `SCALER = "robust"` (RobustScaler; opzione `"standard"` per confronto)
- [x] `K_RANGE = range(2, 9)` (k candidati per KMeans/Agglomerative)
- [x] `KMEANS_PARAMS` (`n_init=10`, `random_state`), `AGGLOMERATIVE_PARAMS` (`linkage="ward"`)
- [x] `DBSCAN_EPS_GRID` (es. `[0.5, 0.75, 1.0, 1.25, 1.5]` su feature scalate) + `DBSCAN_MIN_SAMPLES = 3`
- [x] `RANDOM_STATE = 42`
- [x] Path: `ARTIFACTS_DIR`, `PLOTS_DIR`, `PROFILES_FILE` (`station_profiles.parquet`), `CLUSTERS_FILE` (`station_clusters.csv`), `METRICS_FILE` (`clustering_metrics.json`), `MAP_FILE` (`cluster_map.html`)

### 7.7.3 `profiles.py` — costruzione profili per stazione

- [x] Carica `step_3_eda/daily_dataset_clean.parquet`
- [x] `build_station_profiles(df) -> pd.DataFrame` con `groupby("idstazione")` e aggregazioni:
  - `pm10_mean` = media PM10 (cronico)
  - `pct_critical_days` = frazione giorni con `classe_allerta in {"arancio","rosso"}` (PM10>35; più discriminante e meno sparso del solo "rosso", che a livello stazione è quasi sempre 0)
  - `seasonality_ratio` = media PM10 `stagione=="inverno"` / media PM10 `stagione=="estate"` (ampiezza stagionale, decorrela dalla media)
  - `stagnation_index_mean` = media `stagnation_index` (regime di stagnazione; copertura ~78% → media affidabile)
  - `dist_industrial_km` = `first()` (statica per stazione)
  - `no2_mean` = media NO2 (proxy traffico)
  - `quota` = `first()`; **imputa l'unica stazione mancante** con la mediana di provincia (fallback mediana globale)
  - `wind_speed_mean` = media vento (regime di ventilazione)
  - mantieni `META_COLS` via `first()`
- [x] `assert` nessun NaN nelle `PROFILE_FEATURES` dopo la costruzione (67 righe complete)
- [x] Salva `station_profiles.parquet`

### 7.7.4 `cluster.py` — fit + selezione modello

- [x] Carica i profili; separa matrice feature `X = profiles[PROFILE_FEATURES]` dai metadati
- [x] Scala con `RobustScaler` fittato su tutte le 67 righe (nessuno split train/test: clustering non supervisionato sull'intera popolazione)
- [x] **KMeans** su `K_RANGE`: registra `inertia_` (elbow) + `silhouette`, `calinski_harabasz`, `davies_bouldin`
- [x] **AgglomerativeClustering** (Ward) su `K_RANGE`: stesse metriche interne + `scipy.cluster.hierarchy.linkage` per il dendrogramma
- [x] **DBSCAN** su `DBSCAN_EPS_GRID`: `silhouette` (escludendo il rumore), `n_clusters`, `n_noise` — confronto, aspettative basse
- [x] Seleziona `(algo, k)` finale per `silhouette` + interpretabilità; assegna le etichette finali alle 67 stazioni
- [x] Salva `scaler.joblib`, `kmeans_best.joblib`, `agglomerative_best.joblib`, `station_clusters.csv` (`idstazione,cluster`)
- [x] Scrivi `clustering_metrics.json`: metriche interne per algo/k + `"best_model"` (algo+k) + `"n_stations": 67`

### 7.7.5 `evaluate.py` — validazione + plot

- [x] Ricarica profili + etichette finali
- [x] **Validazione esterna vs `provincia`** (indipendente): `adjusted_rand_score`, `homogeneity`, `completeness`, `v_measure` → in `clustering_metrics.json`
- [x] **Separazione su severità continua**: distribuzione di `pm10_mean` e `pct_critical_days` per cluster (boxplot) + Kruskal-Wallis p-value
- [x] **Caveat circolarità**: calcola anche `homogeneity` vs `classe_allerta` dominante MA etichettala in JSON come `"non_independent": true` con nota esplicita (riportata solo per completezza, non come prova di validità)
- [x] Tabella interpretazione cluster: media di ogni feature **raw** (non scalata) per cluster → CSV/JSON (per nominare i cluster: "cronico", "intermedio", "pulito"…)
- [x] Plot in `artifacts/plots/`:
  - `elbow.png` (inertia vs k) + `silhouette_vs_k.png`
  - `dendrogram.png` (Agglomerative Ward)
  - `dbscan_kdistance.png` (k-distance per scelta eps)
  - `pca_2d_clusters.png` (PCA 2D colorata per cluster + varianza spiegata in titolo)
  - `cluster_profile_heatmap.png` (medie feature standardizzate per cluster)
  - `pm10_by_cluster_boxplot.png` (separazione severità)

### 7.7.6 `map_view.py` — mappa Folium dei cluster

- [x] `build_cluster_map(profiles, labels) -> folium.Map`: 1 marker per stazione colorato per cluster (palette categorica), popup con `nomestazione` + cluster + `pm10_mean` + `provincia`
- [x] Centra su Lombardia, riusa le convenzioni di `api/services/map_view.py`
- [x] Salva `cluster_map.html` in `artifacts/`

### 7.7.7 Verifica

- [x] `python -m step_6_clustering.cluster` completa senza errori; `station_profiles.parquet`, `station_clusters.csv`, i `.joblib` e `clustering_metrics.json` presenti
- [x] `python -m step_6_clustering.evaluate` completa senza errori; tutti i plot salvati in `artifacts/plots/`
- [x] `clustering_metrics.json`: `silhouette > 0` per il modello scelto, blocco validazione esterna vs `provincia` presente, caveat circolarità marcato `non_independent`
- [x] `cluster_map.html` si apre: 67 marker colorati per cluster
- [x] Nessuna nuova dipendenza richiesta (`scikit-learn`, `scipy`, `folium`, `matplotlib`/`seaborn` già nel progetto) — conferma in `requirements.txt`

### 7.7.8 Allineamento scelte progettuali

- [x] Conferma: O3 escluso dai profili (solo 37/67 stazioni misurano O3); NO2 mantenuto (66/67)
- [x] Conferma: con N=67, KMeans + Agglomerative sono i candidati primari; DBSCAN solo confronto (silhouette/density instabili su pochi punti)
- [x] Conferma: validazione esterna primaria su `provincia` + severità continua; `classe_allerta` riportata solo come check non indipendente (circolarità)
- [x] Conferma: documentare in `technical_doc.md` la sezione clustering (profili, scelta k, interpretazione cluster, limiti N piccolo)

---

## Step 8 — Deploy Cloud Run

> **Allineamento feedback prof:** deploy su GCP Cloud Run (come Lab 4). Niente `docker compose` per produzione: compose resta solo per dev locale della pipeline di training (MySQL + ingestion).

### 8.1 `Dockerfile` (root)

- [x] Crea `Dockerfile` a root seguendo pattern Lab 4 slide 6
- [x] `FROM python:3.11-slim`, `WORKDIR /app`
- [x] `COPY requirements.txt .` + `RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt`
- [x] `COPY api/ ./api/`, `COPY shared/ ./shared/`, `COPY step_4_regression/ ./step_4_regression/`, `COPY step_5_classification/ ./step_5_classification/`
- [x] `COPY artifacts/ ./artifacts/` (modelli `.joblib` + `daily_dataset_clean.parquet`)
- [x] `ENV PORT=8080` + `EXPOSE 8080`
- [x] `CMD ["sh","-c","uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]`

### 8.2 Configurazione env per Cloud Run

- [x] `api/config.py`: legge da env `GCS_BUCKET`, `ARTIFACTS_DIR`, `PARQUET_PATH` con default sensati
- [x] Service account Cloud Run: ruolo `roles/storage.objectViewer` sul bucket `exam-project-backfill` (per fetch 7gg recenti)

### 8.3 Deploy

- [x] `gcloud run deploy pm10-forecast --source . --region=europe-west1 --max-instances=2 --allow-unauthenticated --memory=1Gi --set-env-vars GCS_BUCKET=exam-project-backfill`
- [x] URL pubblico restituito: `https://pm10-forecast-47880508774.europe-west1.run.app`

### 8.4 `compose.yaml` (solo dev locale, NON per deploy)

- [x] Tieni `step_2_ingestion/compose.yaml` esistente per MySQL locale + pipeline training
- [x] Aggiungi commento in `README.md`: "compose.yaml serve solo per training pipeline locale; il deploy produzione è su Cloud Run"

### 8.5 Verifica deploy

- [x] `https://<service-url>/health` → `{"status":"ok"}`
- [x] `https://<service-url>/` → mappa Lombardia caricata
- [x] `https://<service-url>/forecast?station_id=<id>&days=1` → JSON previsione
- [x] `https://<service-url>/docs` → Swagger UI
- [x] Log Cloud Run: nessun errore di connessione MySQL, fetch GCS OK

---

## Step 9 — GitHub Actions CI/CD

### 9.1 Workflow CI (`test.yml`)

- [x] Crea `.github/workflows/test.yml`
- [x] Trigger: `push` su `master`/`main` + `pull_request`
- [x] Job `test`: `ubuntu-latest`, setup Python 3.11, `pip install -r requirements.txt`, `pytest tests/ -v`

### 9.2 Workflow Docker Publish (`docker-publish.yml`)

- [x] Crea `.github/workflows/docker-publish.yml`
- [x] Trigger: `push` su `master`/`main` + tag `v*.*.*`
- [x] Step: login GHCR con `GITHUB_TOKEN`
- [x] Step: build e push singola immagine FastAPI integrata -> `ghcr.io/${{ github.repository }}/pm10-forecast:latest`
- [ ] Collega il package GHCR al repository GitHub dalla pagina del package *(azione manuale una-tantum sulla pagina GitHub del package, da fare dopo il primo push che pubblica l'immagine)*

### 9.3 Test suite di base (`tests/`)

- [x] Crea `tests/__init__.py`
- [x] Test unit: `temporal_train_test_split` — verifica che nessun giorno sia in train e test contemporaneamente
- [x] Test unit: `make_temporal_cv_splits` — verifica 5 fold, nessun overlap intra-fold
- [x] Test unit: `impute_missing` — verifica che le mediane del test set NON vengano calcolate sui dati test
- [x] Test integration: `GET /health` → 200 `{"status": "ok"}`
- [x] Test integration: `GET /stations` → lista con campi `idstazione`, `lat`, `lon` presenti

### 9.4 Verifica

- [x] `pytest tests/ -v` passa localmente senza errori *(62 test, 0 errori)*
- [x] Pipeline CI verde su GitHub dopo push *(run 28798897964 "Test" e 28798897890 "Docker Publish" entrambe success su master; il primo tentativo Docker Publish era fallito per un file gitignorato, vedi fix in `fix(docker): commit sensors_registry.json required by Dockerfile`)*

---

## Step 10 — `summary.ipynb`

- [x] Crea `summary.ipynb` alla root del progetto
- [x] **Sezione 1 — Introduzione**: cella markdown con obiettivo, dataset usati, pipeline diagram (ASCII da `technical_report.md` sezione 3.1)
- [x] **Sezione 2 — Raccolta Dati**: cella codice con esempio fetch ARPA + Open-Meteo, preview JSON risultante
- [x] **Sezione 3 — Feature Engineering**: tabelle feature principali (temporali, lag/rolling, meteo derivate, spaziali) + motivazioni encoding ciclico
- [x] **Sezione 4 — EDA**: carica `daily_dataset_clean.parquet`, mostra i 16 plot da `step_3_eda/plots/` (13 EDA core + 3 diagnostico RMT) con `IPython.display`
- [x] **Sezione 5 — Regressione PM10**: carica `step_4_regression/artifacts/regression_metrics.json`, tabella comparativa R²/RMSE/MAE + plot diagnostici per modello e feature importance (ElasticNet nativa, RF/XGBoost permutation)
- [x] **Sezione 6 — Classificazione Allerta**: carica `step_5_classification/artifacts/classification_metrics.json`, tabella F1-macro per strategia, confusion matrix di tutte le 7 strategie, permutation importance dei classificatori base + ordinale, reliability diagram della calibrazione *(strategia finale ibrida da `final_model_selection.json`)*
- [x] **Sezione 7 — Demo previsione**: chiama `GET /forecast?station_id=501&days=1` via `requests`, mostra output JSON formattato *(con fallback automatico alla prima stazione con dati freschi se la 501 viene respinta dal guardrail `valid_days_last_7 < 3`)*
- [x] **Sezione 8 — Conclusioni**: tabella limiti del modello (catturabili vs non catturabili), possibili miglioramenti futuri
- [x] Verifica: notebook eseguibile da cima a fondo senza errori (`Run All`) con API attiva per la sezione demo *(notebook in inglese; eseguito 2026-07-06 via `nbconvert --execute`: 22 celle, 0 errori, 34 immagini; demo servita da Sondrio v.Paribelli per fallback — la 501 aveva solo 2 giorni validi negli ultimi 7)*

---

## Step 11 — `README.md`

- [x] Crea `README.md` alla root del progetto
- [x] **Titolo + descrizione** (2–3 righe): cosa fa, quali dati usa, cosa predice
- [x] **Project Structure**: albero directory semplificato (solo livello 1–2)
- [x] **Pipeline Overview**: diagramma ASCII (data flow da `technical_report.md` §3.1)
- [x] **Quick Start — Prerequisites**: Docker, Python 3.11+
- [x] **Quick Start — Run with Docker**: `docker build`/`docker run` sull'immagine unica FastAPI (backend+frontend integrati, porta 8080) — *nota: nessun `docker compose up` in produzione e nessuna webapp separata su :8501: dopo lo Step 7 il frontend Jinja2 è servito dalla stessa app FastAPI; `compose.yaml` resta solo per MySQL/training locale, coerente con lo Step 8.4*
- [x] **Quick Start — Run locally**: comandi step-by-step (MySQL → ingest → eda/build_dataset → train+evaluate regression → train+evaluate+calibrate classification → clustering opzionale → API su `uvicorn`, porta 8000)
- [x] **API Reference**: tabella endpoint principali (metodo, path, descrizione) + link a `/docs`
- [x] **Data Sources**: ARPA Lombardia, Open-Meteo, OpenStreetMap
- [x] **Models**: tabella breve (modello, task, metrica primaria, score ottenuto dal test set) — dati presi da `regression_metrics.json`, `final_model_selection.json`, `clustering_metrics.json`
- [x] **License**: nessuna licenza open-source (all rights reserved) — scelta confermata dall'utente

---

## Ordine di Esecuzione

```
Step 5 (classification)
    ↓
Step 6 (API) + Step 7 (webapp) + Step 7.7 (clustering)  ← paralleli (clustering dipende solo dal parquet)
    ↓
Step 8 (Docker)
    ↓
Step 9 (CI/CD) + Step 10 (notebook) + Step 11 (README)  ← parallelizzabili
```

> Step 7.7 (clustering) dipende solo da `daily_dataset_clean.parquet`: nessun vincolo su API/frontend.  
> Step 10 (notebook) richiede che Step 5, 6 e 8 siano completati (per i `.json` di metriche e l'API attiva); può includere una sezione clustering (mappa + PCA da `step_6_clustering/artifacts/`).  
> Step 11 (README) richiede che tutti i comandi di esecuzione siano verificati.
