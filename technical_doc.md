# AriaPulita — Documentazione Tecnica

> Documento di riferimento tecnico del progetto. Descrive l'architettura, le scelte progettuali, l'implementazione attuale e il piano per i passi ancora da completare.

---

## 1. Obiettivo del Progetto

Raccogliere dati giornalieri sulla qualità dell'aria dalle stazioni ARPA Lombardia, arricchirli con variabili meteorologiche da Open-Meteo, e addestrare modelli di Machine Learning per:

- **Regressione**: prevedere il valore continuo di PM10 (µg/m³) per stazione e giorno
- **Classificazione**: assegnare il livello di allerta giornaliero in 4 categorie

| Classe | Label | Soglia PM10 (µg/m³) | Riferimento normativo |
|--------|-------|---------------------|----------------------|
| 0 | verde | [0, 20) | OMS 2021 |
| 1 | giallo | [20, 35) | Direttiva EU 2008/50/CE |
| 2 | arancio | [35, 50) | Direttiva EU 2008/50/CE |
| 3 | rosso | [50, ∞) | Superamento limite giornaliero |

---

## 2. Architettura della Pipeline

```
┌─────────────────────────────────────────────────┐
│  STEP 1 — Raccolta Dati (backfill/giornaliero)  │
│  ARPA Lombardia Socrata API + Open-Meteo        │
│  Output: data/raw/{date}_measurements.json      │
│           data/raw/{date}_weather.json          │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  STEP 2 — Ingestion ETL                         │
│  Docker MySQL + ingest.py                       │
│  Output: tabelle stations, sensors,             │
│           measurements, weather_hourly          │
│  Opzionale: industrial_proximity.parquet        │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  STEP 3 — Feature Engineering & EDA             │
│  build_dataset.py + eda.py                      │
│  Output: daily_dataset_clean.parquet (ML-safe)  │
│           daily_dataset.parquet (visualizzazione)│
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  STEP 4 — Regressione PM10 ✅                   │
│  ElasticNet + XGBoost + RandomForest            │
│  Output: artifacts/ (joblib, JSON, plots)       │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  STEP 5 — Classificazione Allerta (da fare)     │
│  LogisticRegression + XGBoost + RandomForest    │
└─────────────────────────────────────────────────┘
```

---

## 3. Struttura del Progetto

```
exam_project/
├── brainstorm.md                   # Idee iniziali (deprecato, usa questo file)
├── technical_doc.md                # Questo file
├── implementation_plan.md          # Fix critici applicati alla pipeline
├── requirements.txt
│
├── data/
│   └── raw/                        # JSON giornalieri + parquet spaziali
│       ├── {date}_measurements.json
│       ├── {date}_weather.json
│       ├── sensors_registry.json   # Cache metadati sensori
│       ├── industrial_zones.geojson  # Poligoni OSM landuse=industrial
│       └── industrial_proximity.parquet  # Feature spaziali calcolate
│
├── step_1_collection/
│   ├── collector.py               # Funzioni di fetch (ARPA + Open-Meteo)
│   └── backfill.py                # CLI: scarica storico date range
│
├── step_2_ingestion/
│   ├── compose.yaml               # Docker MySQL
│   ├── schema.sql                 # DDL database
│   └── ingest.py                  # ETL: JSON → MySQL + proximity parquet
│
├── step_3_eda/
│   ├── build_dataset.py           # Feature engineering → parquet ML-safe
│   ├── eda.py                     # Analisi esplorativa + 13 plot
│   └── plots/
│   daily_dataset_clean.parquet    # Output ML (pre-imputation)
│   daily_dataset.parquet          # Output visualizzazione (post-imputation)
│
├── step_4_regression/
│   ├── __init__.py
│   ├── config.py                  # Costanti, iperparametri, path
│   ├── train.py                   # Training pipeline
│   ├── evaluate.py                # Valutazione test set + salvataggio
│   └── artifacts/
│       ├── elasticnet_best.joblib
│       ├── xgboost_best.joblib
│       ├── random_forest_best.joblib
│       ├── best_model.joblib
│       ├── regression_metrics.json
│       └── plots/
│
├── step_5_classification/         # Da implementare
│   ├── __init__.py
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   └── artifacts/
│
├── shared/
│   └── utils.py                   # Split, imputation, preprocessor, CV splits
│
└── scripts/                       # Script di utilità vari
```

---

## 4. Fonti Dati

### 4.1 ARPA Lombardia — Socrata Open Data API

| Risorsa | Endpoint |
|---------|----------|
| Misure inquinanti | `https://www.dati.lombardia.it/resource/nicp-bhqi.json` |
| Anagrafica sensori | `https://www.dati.lombardia.it/resource/ib47-atvt.json` |

**Filtri applicati:** `stato = 'VA'` (misure validate); `$where` per data.
**Paginazione:** chunk da 50.000 righe (`ARPA_PAGE_SIZE = 50_000`).

**Scoperta EDA critica:** PM10 e PM2.5 restituiscono un solo valore giornaliero (timestamp T00:00:00), non orario. NO2, O3 e CO hanno letture orarie reali. Questa asimmetria impone:
- Granularità del dataset: **giornaliera** (1 riga per stazione per giorno)
- NO2 orario usato per calcolare proxy traffico (`no2_mean`, `no2_max` al giorno)
- Feature meteo aggregate a mediane/medie giornaliere per allinearsi al PM10

### 4.2 Open-Meteo — Historical Forecast API

**Endpoint:** `https://historical-forecast-api.open-meteo.com/v1/forecast`  
**Autenticazione:** nessuna (API gratuita)  
**Retry logic:** 3 tentativi con backoff esponenziale (3s, 6s, 12s)

**Variabili orarie scaricate e aggregate giornalmente:**

| Feature grezza | Aggregazione | Feature nel dataset | Significato fisico |
|----------------|-------------|--------------------|--------------------|
| `temperature_2m` | mean, min, max | `temp_mean`, `temp_min`, `temp_max` | Temperatura |
| `relative_humidity_2m` | mean | `humidity_mean` | Umidità relativa |
| `dew_point_2m` | mean | `dewpoint_mean` | Punto di rugiada |
| `precipitation` | sum | `precip_sum` | Washout PM10 |
| `surface_pressure` | mean | `pressure_mean` | Stagnazione anticiclonica |
| `cloud_cover` | mean | `cloud_cover_mean` | Copertura nuvolosa |
| `wind_speed_10m` | mean, max | `wind_speed_mean`, `wind_speed_max` | Dispersione inquinanti |
| `wind_direction_10m` | — | — | Non usata direttamente |
| `visibility` | mean | `visibility_mean` | Proxy nebbia/foschia |
| `shortwave_radiation` | mean | `radiation_mean` | Proxy inversione termica |
| `boundary_layer_height` | mean, min | `blh_mean`, `blh_min` | Altezza strato rimescolamento |

`fog_hours` è calcolata in SQL: `SUM(CASE WHEN (temperature_2m − dew_point_2m) < 2 THEN 1 ELSE 0 END)` — conta le ore giornaliere con condizioni di nebbia/foschia.

### 4.3 Feature Spaziali — OpenStreetMap (GeoPandas)

Poligoni `landuse=industrial` scaricati dall'Overpass API per l'intera Lombardia e salvati in `data/raw/industrial_zones.geojson`. Per ogni stazione ARPA vengono calcolate:

- `dist_industrial_km` — distanza dal bordo della zona industriale più vicina
- `n_industrial_zones_15km` — numero di zone industriali entro 15 km

**Proiezione:** EPSG:32632 (UTM 32N) per distanze metriche accurate.  
**Calcolo:** `ingest.py` esegue il join spaziale se `industrial_zones.geojson` è presente; salva il risultato in `industrial_proximity.parquet` e lo unisce al dataset giornaliero in `build_dataset.py`.

---

## 5. Feature Engineering

Tutto il feature engineering è implementato in `step_3_eda/build_dataset.py`.

### 5.1 Feature Temporali

| Feature | Formula / Logica |
|---------|-----------------|
| `mese` | Mese (1–12) |
| `stagione` | `{12,1,2}→inverno`, `{3,4,5}→primavera`, `{6,7,8}→estate`, `{9,10,11}→autunno` |
| `giorno_settimana` | 0=lunedì, 6=domenica |
| `is_weekend` | `True` se sabato o domenica |
| `heating_season` | `1` se mese ∈ {10,11,12,1,2,3}, altrimenti `0` |
| `mese_sin` | `sin(2π × mese / 12)` |
| `mese_cos` | `cos(2π × mese / 12)` |
| `dow_sin` | `sin(2π × giorno_settimana / 7)` |
| `dow_cos` | `cos(2π × giorno_settimana / 7)` |

**Motivo encoding ciclico:** il modello deve sapere che dicembre (12) e gennaio (1) sono temporalmente adiacenti; una variabile lineare rompe questa continuità.

### 5.2 Feature Lag e Rolling (per stazione)

Tutte le lag/rolling features usano `groupby("idstazione")` per evitare di mescolare serie di stazioni diverse.

| Feature | Formula esatta | Note |
|---------|---------------|------|
| `pm10_lag1` | `.shift(1)` | Valore PM10 giorno precedente |
| `pm10_lag2` | `.shift(2)` | Due giorni fa |
| `pm10_roll3` | `.shift(1).rolling(3, min_periods=1).mean()` | Media 3gg escluso giorno corrente |
| `pm10_roll7` | `.shift(1).rolling(7, min_periods=1).mean()` | Media 7gg escluso giorno corrente |
| `pm10_diff` | `pm10_lag1 − pm10_lag2` | Variazione 1→2 giorni fa |
| `pressure_mean_lag1` | `.shift(1)` | |
| `pressure_mean_lag2` | `.shift(2)` | **Eliminata** (corr ~0.999 con lag1) |
| `wind_speed_mean_lag1` | `.shift(1)` | |
| `wind_speed_mean_lag2` | `.shift(2)` | |
| `blh_mean_lag1` | `.shift(1)` | |
| `blh_mean_lag2` | `.shift(2)` | |
| `temp_mean_lag1` | `.shift(1)` | |
| `temp_mean_lag2` | `.shift(2)` | |
| `pressure_roll3` | `.shift(1).rolling(3, min_periods=3).mean()` | Anti-leakage: `min_periods=3` evita medie su 1–2 campioni |
| `wind_speed_roll3` | `.shift(1).rolling(3, min_periods=3).mean()` | Idem |

**Motivo del `.shift(1)` sulle rolling:** senza shift, la finestra `rolling(3)` include il giorno corrente, creando data leakage (il modello userebbe il valore che sta cercando di predire). Questo era un bug esplicito corretto nel refactoring (fix documentato in `implementation_plan.md`).

### 5.3 Feature Meteo Derivate

**Fog proxy** (calcolata in SQL prima della pivot):
```sql
SUM(CASE WHEN (temperature_2m - dew_point_2m) < 2 THEN 1 ELSE 0 END) AS fog_hours
```

**Stagnation flag** (binario, soglie fisse da letteratura meteorologica):
```
stagnation_flag = (pressure_mean > 1013.25 hPa)
               AND (wind_speed_mean < 1.5 m/s)
               AND (blh_min < 500 m)
```
Restituisce `NaN` se qualunque input è mancante.

**Stagnation index** (continuo, adimensionale):
```
wind_clipped = max(wind_speed_mean, 0.1)   # evita divisione per zero
blh_clipped  = max(blh_min, 10.0)          # evita divisione per zero
stagnation_index = 1.0 / (wind_clipped × blh_clipped × (1.0 + precip_sum))
```
Valori più alti = stagnazione più intensa. Massimo teorico: `1 / (0.1 × 10 × 1) = 1.0`.

**Motivazione stagnation index vs flag:** il flag è binario e utile per interpretabilità, ma un indice continuo cattura la gradualità della stagnazione (es. pressione leggermente sotto soglia ma vento quasi nullo). Entrambi mantenuti nel dataset.

**Motivazione soglie fisse (non data-driven):** usare `df.median()` sull'intero dataset per definire le soglie di stagnazione sarebbe data leakage — le soglie dipenderebbero dai dati di test. Le soglie meteorologiche standard (pressione standard = 1013.25 hPa, Beaufort "calma" = < 1.5 m/s) sono deterministiche e universali.

### 5.4 Colonne Ridondanti Eliminate

```python
_REDUNDANT_COLS = ["temp_max", "temp_min", "pressure_mean_lag2"]
```

| Colonna eliminata | Alternativa mantenuta | Correlazione |
|-------------------|-----------------------|-------------|
| `temp_max` | `temp_mean` | ~0.985 |
| `temp_min` | `temp_mean` | ~0.985 |
| `pressure_mean_lag2` | `pressure_mean_lag1` | ~0.999 |

**Motivazione:** colonne con |r| > 0.98 degradano il condizionamento della matrice per ElasticNet (instabilità numerica nei coefficienti) e aggiungono rumore ai modelli ad albero senza beneficio informativo.

### 5.5 Target

- **Regressione:** `pm10` (µg/m³), valore continuo — `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` per stabilizzare la varianza sulla distribuzione asimmetrica a destra.
- **Classificazione:** `classe_allerta` (verde/giallo/arancio/rosso) — label encoding ordinale esplicito `{'verde':0, 'giallo':1, 'arancio':2, 'rosso':3}`.

---

## 6. Preprocessing Anti-Leakage

### 6.1 Pipeline in Due Fasi

**Fase 1 — `clean_dataset()` (sicura, pre-split):**  
Operazioni che non usano statistiche globali calcolate sui dati:
- Drop colonne con > 50% missing (`MISSING_THRESHOLD = 0.50`)
- Drop stazioni con < 50% copertura PM10 sul periodo
- Forward-fill per stazione degli inquinanti e delle lag/rolling feature (usa solo dati passati → nessun leakage)
- Drop righe con target `pm10` mancante

Output: `daily_dataset_clean.parquet` — può contenere NaN residui nelle feature meteo.

**Fase 2 — `impute_missing()` (da eseguire DOPO lo split):**  
Imputazione mediana con pattern fit/transform per prevenire il leakage statistico:
```python
# Training
X_train_imp, medians = impute_missing(X_train)

# Test: usa SOLO le mediane del training set
X_test_imp, _ = impute_missing(X_test, medians=medians)
```

**Colonne imputate:** prefissi `temp_`, `humidity_`, `dewpoint_`, `precip_`, `pressure_`, `cloud_`, `wind_`, `visibility_`, `radiation_`, `blh_`, `fog_`, `no2_`, `o3_`, `co_`, `pm25_`; suffissi `_lag1`, `_lag2`, `_roll3`, `_roll7`, `_diff`.

### 6.2 Anti-Leakage Checklist

| Check | Status |
|-------|--------|
| Split temporale puro (no shuffle) | ✅ |
| Mediana imputazione calcolata solo su train set | ✅ |
| Nessuna feature che usa informazione futura | ✅ |
| `TimeSeriesSplit` per CV (no KFold/StratifiedKFold) | ✅ |
| Lag features calcolate pre-split con `shift()` | ✅ |
| Rolling features con `.shift(1)` prima di `.rolling()` | ✅ |
| Soglie stagnazione fisse (non data-dependent) | ✅ |
| `StandardScaler` fit solo su train, transform su test | ✅ |
| `classe_allerta` esclusa dalle feature (drop esplicito) | ✅ |
| CV splits group-aware per giorno (no stessa data in train e val) | ✅ |

---

## 7. Infrastruttura Condivisa (`shared/utils.py`)

### 7.1 `temporal_train_test_split()`

```python
def temporal_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    date_col: str = "data_giorno",
    train_ratio: float = 0.70,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]
```

Ritorna `(X_train, y_train, X_test, y_test, train_dates)`.  
Lo split è per giorno unico: tutti gli osservazioni dello stesso giorno finiscono nello stesso set (70% dei giorni distinti in train, 30% in test), evitando che stazioni dello stesso giorno compaiano sia in train che in test.

`train_dates` è restituita direttamente per costruire i fold CV senza ricostruire le date da `df` — eliminando una dipendenza fragile sull'ordine delle righe che era presente in una versione precedente del codice.

### 7.2 `make_temporal_cv_splits()`

```python
def make_temporal_cv_splits(
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    *,
    date_col: str = "data_giorno",
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]
```

**Logica:** sklearn `TimeSeriesSplit` opera su indici di riga. Con più stazioni per giorno, applicarlo direttamente causerebbe data leakage intra-giorno (stazioni dello stesso giorno in train e validation). Questa funzione:
1. Mappa ogni giorno unico al suo array di indici riga
2. Applica `TimeSeriesSplit(n_splits=5)` sui giorni unici
3. Espande i fold da indici-giorno a indici-riga

Il risultato è una lista di 5 `(train_rows, val_rows)` temporalmente corretti.

### 7.3 `build_preprocessor()`

```python
def build_preprocessor(
    X: pd.DataFrame,
    *,
    model_type: Literal["linear", "tree"] = "linear",
    categorical_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None,
) -> ColumnTransformer
```

| Colonne | Trattamento (linear) | Trattamento (tree) |
|---------|----------------------|--------------------|
| Numeriche | `SimpleImputer(median)` + `StandardScaler` | `passthrough` |
| Categoriche (`stagione`, `provincia`) | `OrdinalEncoder` | `OrdinalEncoder` |
| Drop (`idstazione`, `nomestazione`, `comune`, `data_giorno`, `classe_allerta`) | `drop` | `drop` |
| Resto | `drop` | `drop` |

Per `stagione` l'ordine esplicito è `["inverno", "primavera", "estate", "autunno"]`. Per altre colonne categoriche le categorie sono derivate con `sorted(X[col].dropna().unique())` al momento della build, evitando il bug sklearn `OrdinalEncoder(categories=[list, "auto"])`.

---

## 8. Step 1 — Raccolta Dati

**Moduli:** `step_1_collection/collector.py`, `step_1_collection/backfill.py`

### collector.py — Funzioni Principali

| Funzione | Descrizione |
|----------|-------------|
| `fetch_sensor_registry()` | Fetch paginato anagrafica sensori ARPA |
| `fetch_arpa_measurements(date_str)` | Misure validate per un giorno |
| `fetch_station_weather_historical(lat, lng, date_str)` | Meteo orario per cella grid con retry esponenziale |
| `save_json(filename, data, output_dir)` | Serializza output in `data/raw/` |

**Parametri chiave:** `REQUEST_TIMEOUT = 30s`, `WEATHER_MAX_RETRIES = 3`, `WEATHER_RETRY_BACKOFF = 3.0s` (raddoppia ad ogni retry). Timezone: `Europe/Rome`.

### backfill.py — CLI

```bash
python -m step_1_collection.backfill \
    --start-date 2025-03-01 \
    --end-date 2026-03-31 \
    --skip-existing \
    --sleep 2
```

**Ottimizzazione coordinate:** le coordinate sono arrotondate a 1 decimale (`COORD_ROUND_DP = 1`, ~11 km di precisione). Le ~170 stazioni Lombardia si riducono a ~15–20 celle uniche per Open-Meteo → riduce drasticamente le chiamate API rispettando i rate limit.

**Output:** `data/raw/{date}_measurements.json`, `data/raw/{date}_weather.json`, `data/raw/backfill_summary_{start}_{end}.json`.

---

## 9. Step 2 — Ingestion ETL

**Moduli:** `step_2_ingestion/schema.sql`, `step_2_ingestion/ingest.py`

### Schema MySQL

| Tabella | PK | Colonne principali | Unique constraint |
|---------|----|--------------------|-------------------|
| `stations` | `idstazione` | nome, provincia, comune, quota, lat/lng | — |
| `sensors` | `idsensore` | `idstazione` (FK), tipo, unità, date validità | — |
| `measurements` | auto-increment | `idsensore`, `data`, `valore`, `stato` | `(idsensore, data)` |
| `weather_hourly` | auto-increment | `idstazione`, `dt`, 11 variabili meteo | `(idstazione, dt)` |

Insert idempotente via `INSERT IGNORE` (safe per ri-esecuzioni parziali).

### ingest.py — Flusso

1. Fetch o legge dalla cache `sensors_registry.json`
2. Connessione MySQL con 10 retry (3s delay, per attendere il container)
3. Upsert `stations` e `sensors`
4. Carica tutti `*_measurements.json` da `data/raw/`
5. Flatten JSON meteo orario → insert in `weather_hourly`
6. **Opzionale:** se `data/raw/industrial_zones.geojson` presente, calcola le feature spaziali via GeoPandas e salva `industrial_proximity.parquet`

**Config via env:** `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DB`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_BATCH_SIZE = 5_000`.

---

## 10. Step 3 — Feature Engineering & EDA

**Moduli:** `step_3_eda/build_dataset.py`, `step_3_eda/eda.py`

### build_dataset.py — Flusso

```
MySQL (misure + meteo) → pivot pollutanti → fill gap date →
lag/rolling features → feature temporali → stagnation →
feature spaziali (merge proximity parquet) → alert class →
clean_dataset() → save daily_dataset_clean.parquet
```

Output `step_3_eda/daily_dataset_clean.parquet`:
- Granularità: 1 riga per stazione per giorno
- Periodo: marzo 2025 – marzo 2026 (~13 mesi)
- ~7.000–10.000 righe (post-pulizia)
- Colonne: ~65–75 (feature + target + identificatori)

### eda.py — Analisi Eseguita

13 plot salvati in `step_3_eda/plots/`:
- Distribuzione PM10 e distribuzione classi di allerta
- Heatmap di correlazione completa
- Serie temporale PM10 per stazione
- PM10 per stagione, giorno della settimana, mese, provincia
- Scatter meteo vs PM10 (temperatura, umidità, vento, pressione)
- Stagnation index vs PM10
- Prossimità industriale vs PM10
- Profilo orario NO2 (proxy traffico)

---

## 11. Step 4 — Regressione PM10 ✅

### 11.1 Configurazione (`config.py`)

```python
TARGET_COL    = "pm10"
DROP_COLS     = ["idstazione", "nomestazione", "comune", "classe_allerta"]
# "data_giorno" è anche dropped via _DEFAULT_DROP_COLS in shared/utils.py
N_CV_SPLITS   = 5
CV_SCORING    = "neg_root_mean_squared_error"
RANDOM_STATE  = 42
RANDOM_SEARCH_N_ITER = 50
```

### 11.2 Pipeline per Modello

Tutti e tre i modelli usano `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`. **Motivazione:** la distribuzione di PM10 è asimmetrica a destra (molti giorni bassi, pochi picchi elevati); il log-transform stabilizza la varianza e riduce la sottostima sistematica dei picchi, che sono i giorni critici per la salute.

**ElasticNet:**
```
ColumnTransformer(StandardScaler su numeriche)
  → TransformedTargetRegressor
      → ElasticNet(max_iter=2000)
```
Search: `GridSearchCV` — 6 alpha × 4 l1_ratio = **24 combinazioni** × 5 fold = 120 fit.

```python
"regressor__regressor__alpha":    [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
"regressor__regressor__l1_ratio": [0.1, 0.5, 0.7, 0.9]
```

**XGBoost:**
```
ColumnTransformer(passthrough su numeriche)
  → TransformedTargetRegressor
      → XGBRegressor(tree_method="hist")
```
Search: `RandomizedSearchCV(n_iter=50)` — spazio 4×4×3×3×3×3 = 1.296 combinazioni; n_iter=50 ≈ 3.9% del grid. 250 fit totali.

```python
"n_estimators":    [100, 300, 500, 800]
"max_depth":       [3, 5, 7, 9]
"learning_rate":   [0.01, 0.05, 0.1]
"subsample":       [0.7, 0.8, 1.0]
"min_child_weight":[1, 3, 5]
"colsample_bytree":[0.7, 0.8, 1.0]
```

**RandomForest:**
```
ColumnTransformer(passthrough su numeriche)
  → TransformedTargetRegressor
      → RandomForestRegressor
```
Search: `RandomizedSearchCV(n_iter=50)` — spazio 3×3×3 = 27 combinazioni (over-sampled, equivale a ~1.85× il grid completo). 250 fit totali.

```python
"n_estimators":    [100, 300, 500]
"max_depth":       [10, 20, None]
"min_samples_leaf":[2, 5, 10]
```

**Motivazione GridSearch per ElasticNet vs RandomizedSearch per gli alberi:** ElasticNet ha uno spazio piccolo (24 combinazioni) e il grid completo è computazionalmente conveniente; XGBoost e RandomForest hanno spazi molto più grandi e RandomizedSearch con n_iter=50 cattura il 90% della qualità con frazione del costo.

**Motivazione StandardScaler solo per ElasticNet:** i modelli ad albero sono invarianti alla scala delle feature (le soglie vengono trovate indipendentemente dal range). StandardScaler su alberi spreca cicli e non migliora le metriche. ElasticNet invece è sensibile alla scala perché la penalizzazione L1/L2 tratta tutte le feature simmetricamente.

### 11.3 Risultati sul Test Set (70/30 split temporale, 7.064 campioni)

| Modello | R² | RMSE (µg/m³) | MAE (µg/m³) | RMSE rosso |
|---------|-----|-------------|-------------|------------|
| **Random Forest** | **0.555** | **11.92** | **8.31** | **19.78** |
| XGBoost | 0.521 | 12.37 | 8.64 | 21.22 |
| ElasticNet | -6.98 | 50.51 | 9.96 | 107.03 |

**Best model:** `random_forest` (salvato in `artifacts/best_model.joblib`).

**Distribuzione classi nel test set:**

| Classe | N campioni |
|--------|------------|
| verde | 1.492 (21%) |
| giallo | 2.180 (31%) |
| arancio | 1.869 (26%) |
| rosso | 1.523 (22%) |

### 11.4 Analisi dei Risultati e Limiti

**Perché R² ~0.55 con RMSE ~12 µg/m³?**

Le feature `giorno_settimana`, `stagione`, `heating_season` catturano lo **shift della media attesa** tra categorie (un lunedì di gennaio ha PM10 mediamente più alto di una domenica di agosto), ma non il valore effettivo del singolo giorno. La varianza residua è dominata da eventi meteorologici stocastici:

- Lunedì di gennaio con forte vento → PM10 basso (dispersione)
- Domenica di agosto con stagnazione → PM10 alto (accumulo)

| Driver varianza | Catturabile con questo dataset? |
|-----------------|--------------------------------|
| Vento / BLH (dispersione) | ✅ già nel dataset |
| Pioggia (wet deposition) | ✅ già nel dataset |
| Inversione termica | ⚠️ parziale (BLH come proxy) |
| Trasporto transfrontaliero (polvere sahariana, incendi) | ❌ richiederebbe HYSPLIT o satellite AOD |
| Traffico reale (eventi, neve, scioperi) | ❌ `is_weekend` è proxy statistico |

**Conclusione:** R² ~0.55 è probabilmente vicino al **tetto informativo** del dataset con sole feature meteo locali. La quota inesplicata (~45%) è in gran parte rumore episodico non prevedibile senza dati aggiuntivi.

**Perché ElasticNet fallisce (R² = -6.98)?**  
La relazione PM10 ↔ meteo è intrinsecamente non-lineare (soglie, interazioni, effetti stagionali asimmetrici). ElasticNet impone linearità e non riesce a catturare queste strutture. Il MAE di ElasticNet (9.96) è comparabile agli alberi, ma l'RMSE (50.51) è catastrofico: il modello produce predizioni molto distanti sui picchi (RMSE rosso = 107 µg/m³), che è il caso più critico per la salute pubblica.

---

## 12. Step 5 — Classificazione Allerta (da implementare)

**Modulo:** `step_5_classification/`

**Obiettivo:** prevedere la fascia di allerta (4 classi sbilanciate).  
**Metrica primaria:** F1-macro — bilancia Precision e Recall su tutte le classi equamente, dando lo stesso peso alle allerte rosse (fondamentali per la salute, ma non necessariamente maggioritarie) e alle giornate verdi.

### 12.1 Encoding del Target

```python
LABEL_MAP = {'verde': 0, 'giallo': 1, 'arancio': 2, 'rosso': 3}
y = df["classe_allerta"].map(LABEL_MAP)
```

**Non usare `OrdinalEncoder` sklearn:** non garantisce l'ordine semantico corretto per default. La mappatura esplicita con `.map()` è deterministica e verificabile.

### 12.2 Gestione Sbilanciamento

**Strategia unica:** `class_weight='balanced'` nativo nei modelli che lo supportano.

**SMOTE non utilizzato:** genera giornate meteo sintetiche fisicamente irrealistiche. Con 4 classi meteo-dipendenti, l'oversampling sintetico crea combinazioni impossibili in natura (es. alta temperatura + alta pressione + forte precipitazione nello stesso giorno).

### 12.3 Modelli

**Random Forest Classifier:**
- `class_weight='balanced'`
- `RandomizedSearchCV(n_iter=50)`, scoring: `f1_macro`, cv: `TimeSeriesSplit(5)`
- Search space: `n_estimators`, `max_depth`, `min_samples_leaf`

**XGBoost Classifier:**
- Sample weights calcolati per classe (equivalente a `class_weight='balanced'`)
- `RandomizedSearchCV(n_iter=50)`, scoring: `f1_macro`
- Search space: `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`

**Logistic Regression (ElasticNet penalty):**
```
StandardScaler → LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced')
```
- `GridSearchCV`, scoring: `f1_macro`
- Search space: `C` ∈ [0.01, 0.1, 1, 10]
- Ruolo: contributo lineare complementare ai tree-based (analogo a ElasticNet per la regressione)

### 12.4 Valutazione

- Classification report (Precision, Recall, F1 per classe)
- Confusion matrix heatmap — focus critico: falsi negativi gravi = rosso classificato come verde/giallo
- `severe_error_rate`: percentuale di errori con distanza ≥ 2 tra classe vera e predetta
- Permutation importance su test set (best model)

### 12.5 Calibrazione delle Probabilità (opzionale)

`CalibratedClassifierCV(method='sigmoid')` sul miglior modello post-tuning.  
**Motivazione:** per un sistema di allerta pubblica, "73% probabilità di Rosso" è più utile di un semplice label. Il metodo `sigmoid` è più stabile di `isotonic` con 4 classi su un dataset non enorme.

---

## 12.6 Scaletta di Implementazione Step 5

Checklist in ordine d'esecuzione. Spunta ogni task al completamento.

---

### Task 1 — Scaffold `step_5_classification/` 🔲

- [ ] Crea `step_5_classification/__init__.py` (vuoto, marca il package Python)
- [ ] Crea directory `step_5_classification/artifacts/plots/`

---

### Task 2 — `config.py` 🔲

Costanti da definire (specchio di `step_4_regression/config.py`):

```python
LABEL_MAP    = {'verde': 0, 'giallo': 1, 'arancio': 2, 'rosso': 3}
TARGET_COL   = "classe_allerta"
DROP_COLS    = ["idstazione", "nomestazione", "comune", "pm10"]
# pm10 è il valore continuo da cui è derivata la classe — includerlo sarebbe leakage diretto

N_CV_SPLITS          = 5
CV_SCORING           = "f1_macro"   # <-- cambia rispetto a step 4
RANDOM_STATE         = 42
RANDOM_SEARCH_N_ITER = 50
RANDOM_SEARCH_N_JOBS = -1

ARTIFACTS_DIR = "step_5_classification/artifacts"
PLOTS_DIR     = "step_5_classification/artifacts/plots"
METRICS_FILE  = "step_5_classification/artifacts/classification_metrics.json"
```

Grids iperparametri:

```python
# GridSearchCV — 4 combinazioni × 5 fold = 20 fit (spazio piccolo, exhaustive)
LOGISTIC_PARAM_GRID = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0],
    # penalty='elasticnet', solver='saga', l1_ratio fissi nel pipeline factory
}

# RandomizedSearchCV — stesse dimensioni di step 4
RANDOM_FOREST_PARAM_DIST = {
    "n_estimators":     [100, 300, 500],
    "max_depth":        [10, 20, None],
    "min_samples_leaf": [2, 5, 10],
}

XGBOOST_PARAM_DIST = {
    "n_estimators":    [100, 300, 500, 800],
    "max_depth":       [3, 5, 7, 9],
    "learning_rate":   [0.01, 0.05, 0.1],
    "subsample":       [0.7, 0.8, 1.0],
    "min_child_weight":[1, 3, 5],
    "colsample_bytree":[0.7, 0.8, 1.0],
}
```

---

### Task 3 — `train.py` 🔲

Il flusso è identico a `step_4_regression/train.py` — leggere quel file come riferimento. Differenze chiave:

**3a. Split e encoding target**
```python
X_train, y_train, X_test, y_test, train_dates = temporal_train_test_split(
    df, target_col=TARGET_COL, drop_cols=DROP_COLS
)
y_train = y_train.map(LABEL_MAP)   # .map() esplicito, non OrdinalEncoder
y_test  = y_test.map(LABEL_MAP)
```

**3b. Imputation + CV splits (invariato rispetto a step 4)**
```python
X_train, train_medians = impute_missing(X_train)
X_test, _              = impute_missing(X_test, medians=train_medians)

X_train_with_dates = X_train.copy()
X_train_with_dates.insert(0, "data_giorno", train_dates)
cv_splits = make_temporal_cv_splits(X_train_with_dates, n_splits=N_CV_SPLITS)
```

**3c. Pipeline factory — Logistic Regression**
```
build_preprocessor(X_train, model_type="linear")
  → Pipeline([("pre", ...), ("classifier", LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5,
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    ))])
```
GridSearchCV con `param_grid = LOGISTIC_PARAM_GRID`, scoring=`f1_macro`.  
Nessun `TransformedTargetRegressor` (target categorico, non continuo).

**3d. Pipeline factory — Random Forest Classifier**
```
build_preprocessor(X_train, model_type="tree")
  → Pipeline([("pre", ...), ("classifier", RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ))])
```
RandomizedSearchCV con `param_distributions = {"classifier__<param>": ...}`.

**3e. Pipeline factory — XGBoost Classifier**
```
build_preprocessor(X_train, model_type="tree")
  → Pipeline([("pre", ...), ("classifier", XGBClassifier(
        objective="multi:softprob", num_class=4, eval_metric="mlogloss",
        tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1
    ))])
```
`XGBClassifier` **non** supporta `class_weight`. Usare `sample_weight`:
```python
from sklearn.utils.class_weight import compute_sample_weight
sample_weight = compute_sample_weight("balanced", y_train)
# passare in search.fit():
search.fit(X_train, y_train, classifier__sample_weight=sample_weight)
# sklearn proietta automaticamente sample_weight sugli indici del fold CV
```
RandomizedSearchCV con `param_distributions = {"classifier__<param>": ...}`, scoring=`f1_macro`.

**3f. Best model selection e salvataggio**
```python
# Confronta best_score_ (f1_macro sul CV) tra tutti e tre
best_name = max(trained_models, key=lambda k: trained_models[k].best_score_)
joblib.dump(trained_models[best_name].best_estimator_,
            artifacts_dir / "best_model.joblib")
```

---

### Task 4 — `evaluate.py` 🔲

**4a. Carica modelli e predici**
```python
models = {name: joblib.load(path) for name, path in model_paths.items()}
y_pred = model.predict(X_test)   # ritorna indici 0–3
```

**4b. Metriche per ogni modello**
```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

report = classification_report(
    y_test, y_pred,
    target_names=["verde", "giallo", "arancio", "rosso"],
    output_dict=True
)
severe_error_rate = float((np.abs(y_pred - y_test) >= 2).mean())
# errori con distanza ≥ 2 classi (es. verde predetto come rosso o viceversa)
```

**4c. Plot**
- Confusion matrix heatmap per ogni modello (seaborn `heatmap`, annotata con conteggi)
- Permutation importance solo sul best model:
  ```python
  perm = permutation_importance(
      best_model, X_test, y_test,
      scoring="f1_macro", n_repeats=10, random_state=RANDOM_STATE
  )
  ```

**4d. Salva `classification_metrics.json`**
```json
{
  "logistic_regression": {
    "f1_macro": 0.0, "per_class": {}, "severe_error_rate": 0.0
  },
  "random_forest": { "..." },
  "xgboost":       { "..." },
  "best_model":    "random_forest",
  "n_test_samples": 0
}
```

---

### Task 5 — Calibrazione probabilità (opzionale) 🔲

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv="prefit")
calibrated.fit(X_test, y_test)   # usa test set come proxy — ok per calibrazione post-hoc
joblib.dump(calibrated, artifacts_dir / "best_model_calibrated.joblib")
```

Confronta reliability diagram prima/dopo con `sklearn.calibration.calibration_curve()` per ciascuna classe (one-vs-rest).

---

### Ordine di esecuzione e comandi

```
Task 1 → Task 2 → Task 3 → Task 4 → [Task 5 opzionale]
```

```bash
# dalla root del progetto
python -m step_5_classification.train
python -m step_5_classification.evaluate
```

---

## 13. Stato di Avanzamento

| Step | Descrizione | Stato |
|------|-------------|-------|
| Step 1 | Raccolta dati — `backfill.py` storico 1 anno | ✅ Completato |
| Step 2 | Ingestion — MySQL, schema, ingest.py | ✅ Completato |
| Step 3 | Feature engineering, EDA, `daily_dataset_clean.parquet` | ✅ Completato |
| Step 4 | Regressione — 3 modelli trainati, metriche salvate | ✅ Completato |
| Step 5 | Classificazione — 3 modelli da implementare | 🔲 Da fare |
| Step 6 | Visualizzazione (Metabase) | 🔲 Fuori scope (opzionale) |
| API | Flask `/predict_value` e `/predict_alert` | 🔲 Fuori scope (opzionale) |


