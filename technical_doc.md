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
│          daily_dataset.parquet (visualizzazione)│
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

**Ottimizzazione coordinate:** le coordinate sono arrotondate a 1 decimale (`COORD_ROUND_DP = 1`, ~11 km di precisione). Le ~170 stazioni Lombardia si riducono a ~15-20 celle uniche per Open-Meteo: questo riduce drasticamente le chiamate API rispettando i rate limit, senza diminuire il numero di record finali per stazione. Il compromesso e' una minore risoluzione spaziale delle feature meteo, perche' stazioni vicine condividono la stessa serie Open-Meteo.

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

Tutti e tre i modelli usano `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`. **Motivazione:** `pm10` è un target continuo, non negativo e fortemente asimmetrico a destra (molti giorni con valori moderati, pochi picchi elevati). La trasformazione `log1p` comprime gli estremi, rende più stabile la varianza e sposta l'ottimizzazione verso errori più vicini a una logica relativa che assoluta, migliorando la robustezza del fit sui picchi senza cambiare la natura del problema. `class_weight` e `scale_pos_weight`, invece, sono strumenti nati per classificazione sbilanciata e non sono la scelta naturale per una regressione continua; l'alternativa diretta in regressione sarebbe semmai pesare i singoli campioni o usare una loss custom. In questo progetto si è preferito `log1p` perché corregge prima di tutto la forma statistica del target, mantenendo una pipeline semplice, stabile e coerente per tutti i modelli.

**ElasticNet:**
```
ColumnTransformer(StandardScaler su numeriche)
  -> TransformedTargetRegressor
      -> ElasticNet(max_iter=2000)
```
Search: `GridSearchCV`, 6 alpha x 4 l1_ratio = **24 combinazioni** x 5 fold = 120 fit.

```python
"regressor__regressor__alpha":    [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
"regressor__regressor__l1_ratio": [0.1, 0.5, 0.7, 0.9]
```

**XGBoost:**
```
ColumnTransformer(passthrough su numeriche)
  -> TransformedTargetRegressor
      -> XGBRegressor(tree_method="hist")
```
Search: `RandomizedSearchCV(n_iter=50)`, spazio 4 x 4 x 3 x 3 x 3 x 3 = 1.296 combinazioni; `n_iter=50` copre circa il 3.9% del grid. 250 fit totali.

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
  -> TransformedTargetRegressor
      -> RandomForestRegressor
```
Search: `RandomizedSearchCV(n_iter=50)`, spazio 3 x 3 x 3 = 27 combinazioni. Con 250 fit totali la ricerca ricampiona piu' volte il piccolo spazio disponibile.

```python
"n_estimators":    [100, 300, 500]
"max_depth":       [10, 20, None]
"min_samples_leaf":[2, 5, 10]
```

**Motivazione GridSearch per ElasticNet vs RandomizedSearch per gli alberi:** ElasticNet ha uno spazio piccolo (24 combinazioni), quindi il grid completo e' computazionalmente conveniente. XGBoost e RandomForest hanno spazi molto piu' ampi; in questi casi `RandomizedSearchCV` offre un buon compromesso tra costo di calcolo e qualita' della soluzione.

**Motivazione StandardScaler solo per ElasticNet:** i modelli ad albero sono sostanzialmente invarianti alla scala delle feature, perche' apprendono soglie e non distanze. Applicare `StandardScaler` agli alberi aggiunge costo senza benefici misurabili. ElasticNet, invece, e' sensibile alla scala perche' la penalizzazione L1/L2 agisce direttamente sui coefficienti.

### 11.3 Risultati sul Test Set (70/30 split temporale, 15.544 campioni)

| Modello | R² | RMSE (µg/m³) | MAE (µg/m³) | RMSE rosso |
|---------|-----|-------------|-------------|------------|
| ElasticNet | 0.351 | 13.78 | 9.98 | 29.56 |
| **XGBoost** | **0.721** | **9.03** | **6.08** | **17.36** |
| Random Forest | 0.693 | 9.47 | 6.45 | 18.11 |

**Best model:** `xgboost` (salvato in `artifacts/best_model.joblib`).

**Distribuzione classi nel test set:**

| Classe | N campioni |
|--------|------------|
| verde | 5.339 (34.3%) |
| giallo | 4.992 (32.1%) |
| arancio | 3.099 (19.9%) |
| rosso | 2.114 (13.6%) |

### 11.4 Analisi dei Risultati e Limiti

**Perche' R² ~0.72 con RMSE ~9 µg/m³?**

Le feature meteorologiche e temporali disponibili spiegano una quota consistente della variabilita' giornaliera del PM10, soprattutto quando il modello puo' apprendere relazioni non lineari, soglie e interazioni tra ristagno atmosferico, stagionalita' e proxy emissivi. XGBoost beneficia in particolare della capacita' di modellare pattern complessi e della logica boosting, che corregge progressivamente gli errori residui.

Le feature `giorno_settimana`, `stagione` e `heating_season` catturano lo **shift della media attesa** tra categorie, mentre le variabili meteo descrivono i meccanismi di accumulo o dispersione. Per esempio:

- Giorno invernale con vento debole e BLH bassa -> accumulo di PM10
- Giorno piovoso o ventilato -> dispersione/deposizione e concentrazioni piu' basse

| Driver varianza | Catturabile con questo dataset? |
|-----------------|--------------------------------|
| Vento / BLH (dispersione) | Si', gia' nel dataset |
| Pioggia (wet deposition) | Si', gia' nel dataset |
| Inversione termica | Parziale, con BLH come proxy |
| Trasporto transfrontaliero (polvere sahariana, incendi) | No, richiederebbe HYSPLIT o satellite AOD |
| Traffico reale (eventi, neve, scioperi) | No, `is_weekend` e' solo un proxy statistico |

**Conclusione:** R² ~0.72 indica che il dataset contiene un segnale predittivo forte, ma resta comunque una quota non spiegata (~28%) legata a fattori episodici o non osservati. Il modello e' quindi utile per previsione operativa e analisi comparativa, ma non esaurisce tutta la dinamica fisica del fenomeno.

**Perche' ElasticNet resta inferiore (R² = 0.351)?**  
La relazione PM10 <-> meteo e' intrinsecamente non lineare, con soglie, interazioni ed effetti stagionali asimmetrici. ElasticNet impone una struttura lineare globale e, pur beneficiando della trasformazione `log1p`, non rappresenta bene ne' i picchi ne' i cambi di regime. Il risultato e' un modello piu' stabile rispetto alla versione iniziale, ma ancora nettamente peggiore dei modelli ad albero sia in RMSE complessivo sia nella fascia rossa (29.56 µg/m³ contro 17.36 di XGBoost).

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
- Sample weights calcolati con `compute_sample_weight("balanced", y_train)` passati come `sample_weight` al fit (equivalente a `class_weight='balanced'`, ma compatibile con multi-class)
- `RandomizedSearchCV(n_iter=50)`, scoring: `f1_macro`
- Search space: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `min_child_weight`, `colsample_bytree`

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

`CalibratedClassifierCV(method='isotonic')` sul miglior modello post-tuning.  
**Motivazione:** per un sistema di allerta pubblica, "73% probabilità di Rosso" è più utile di un semplice label. Il metodo `isotonic` (regressione isotonica non-parametrica) ottiene un macro-ECE inferiore rispetto a `sigmoid` (Platt scaling), che tendeva a peggiorare la calibrazione delle classi intermedie (giallo) già ben calibrate dal modello XGBoost.

---

## 13. Step 6 — Webapp & REST API

### 13.1 Architettura

```
┌──────────────────────────────────────────────────┐
│  STEP 6 — Webapp & REST API                      │
│                                                  │
│  FastAPI  →  REST API  (porta 8000)              │
│  Streamlit →  Web UI   (porta 8501)              │
│                                                  │
│  Streamlit chiama FastAPI internamente           │
└──────────────────────────────────────────────────┘
```

Streamlit non espone logica di predizione direttamente: chiama gli endpoint FastAPI via `httpx` (o `requests`). Questo mantiene la separazione API / UI e consente di usare la REST API in modo indipendente.

---

### 13.2 REST API — FastAPI (`api/`)

Struttura directory:

```
api/
├── main.py          # app FastAPI, startup, router include
├── routers/
│   ├── stations.py  # GET /stations
│   ├── forecast.py  # GET /forecast
│   └── history.py   # GET /history
├── services/
│   ├── predictor.py # carica modelli .joblib, build_features, predict
│   └── weather.py   # fetch Open-Meteo forecast
└── schemas.py       # Pydantic response models
```

#### Endpoint

| Metodo | Path | Descrizione |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — `{"status": "ok"}` |
| `GET` | `/stations` | Lista stazioni con id, nome, comune, lat, lon |
| `GET` | `/forecast` | Previsione PM10 + classe allerta per stazione e orizzonte 1-2 gg |
| `GET` | `/history` | Misurazioni storiche PM10 per stazione e intervallo date |

#### `GET /stations`

```json
[
  {
    "idstazione": "501",
    "nomestazione": "Milano - Senato",
    "comune": "Milano",
    "lat": 45.47,
    "lon": 9.19
  },
  ...
]
```

Fonte: query sulla tabella `stations` di MySQL.

#### `GET /forecast?station_id=501&days=1`

Parametri:
- `station_id` (str, required) — id stazione ARPA
- `days` (int, 1 o 2, default 1) — orizzonte di previsione

```json
{
  "station_id": "501",
  "station_name": "Milano - Senato",
  "predictions": [
    {
      "date": "2026-04-08",
      "pm10_predicted": 28.4,
      "alert_class": "giallo",
      "alert_index": 1,
      "weather_used": {
        "temp_mean": 14.2,
        "wind_speed_mean": 3.1,
        "boundary_layer_height_mean": 820.0
      }
    }
  ]
}
```

**Flusso interno `predictor.py`:**

```
1. Recupera le ultime N righe storiche della stazione da MySQL
   (servono per: pm10_lag1, pm10_lag7, pm10_rolling7)

2. Fetch previsioni meteo da Open-Meteo Forecast API
   per lat/lon della stazione, orizzonte = days
   (stesso endpoint usato in step_1_collection, ma con ?forecast=True)

3. Costruisce X_future con le stesse feature usate in training:
   - feature meteo: da Open-Meteo forecast
   - lag1:         ultimo pm10 noto da DB
   - lag7:         pm10 di 7 giorni fa da DB
   - rolling7:     media ultimi 7 giorni da DB
   - feature temporali: day_sin/cos, month_sin/cos, stagione
   - feature spaziali:  dist_industrial, n_industrial_500m (statiche per stazione)

4. model_regression.predict(X_future)   → pm10 float
   model_classification.predict(X_future) → classe allerta int → label

5. Ritorna JSON
```

**Limite orizzonte 1-2 giorni:**
- Giorno +1: tutte le lag features calcolate su dati reali → predizione affidabile
- Giorno +2: lag1 = predizione giorno +1 (valore stimato), lag7/rolling7 ancora reali
- Giorno +3 e oltre: non esposto — l'errore si accumula troppo

#### `GET /history?station_id=501&from_date=2026-01-01&to_date=2026-03-31`

```json
{
  "station_id": "501",
  "records": [
    {"date": "2026-01-01", "pm10": 35.2, "alert_class": "arancio"},
    ...
  ]
}
```

Fonte: query su MySQL (tabella `measurements`) oppure `daily_dataset.parquet`.

---

### 13.3 Web Interface — Streamlit (`webapp/`)

Struttura directory:

```
webapp/
├── app.py           # entry point Streamlit, navigazione pagine
├── pages/
│   ├── 1_map.py     # Mappa Lombardia con allerte
│   ├── 2_forecast.py # Previsione per stazione selezionata
│   └── 3_history.py  # Serie storica PM10
└── utils/
    └── api_client.py # Wrapper httpx per chiamare FastAPI
```

#### Pagina 1 — Mappa allerte

- Mappa Lombardia con `folium` (o `pydeck`) integrata in Streamlit via `st.components`
- Marker per ogni stazione, colorato in base all'allerta prevista per domani:
  - Verde / Giallo / Arancio / Rosso
- Click su marker → popup con nome stazione, comune, PM10 previsto
- Pulsante "Aggiorna previsioni" → chiama `GET /forecast` per tutte le stazioni

#### Pagina 2 — Previsione stazione

- Dropdown: seleziona stazione
- Slider: orizzonte 1 o 2 giorni
- Output:
  - Valore PM10 previsto (numero + badge colorato classe allerta)
  - Tabella feature meteo usate (temperatura, vento, BLH)
  - Nota disclaimer: "Previsione basata su modello ML + dati meteo Open-Meteo. Orizzonte 2 giorni: la lag feature del giorno 2 è stimata."

#### Pagina 3 — Serie storica

- Dropdown: seleziona stazione
- Date picker: intervallo
- Grafico lineare PM10 nel tempo (`st.line_chart` o `plotly`)
- Bande orizzontali colorate per soglie allerta (verde/giallo/arancio/rosso)

---

## 14. Docker & Deploy

### 14.1 Struttura Docker

```
exam_project/
├── Dockerfile.api         # FastAPI backend
├── Dockerfile.webapp      # Streamlit frontend
└── compose.yaml           # Orchestrazione completa (sovrascrive step_2_ingestion/compose.yaml)
```

**`compose.yaml` (root — unificato):**

```yaml
services:

  mysql:
    image: mysql:8.0
    container_name: exam_mysql
    restart: unless-stopped
    environment:
      MYSQL_ROOT_PASSWORD: rootpass
      MYSQL_DATABASE: airquality
      MYSQL_USER: airuser
      MYSQL_PASSWORD: airpass
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
      - ./step_2_ingestion/schema.sql:/docker-entrypoint-initdb.d/01_schema.sql:ro
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-u", "root", "-prootpass"]
      interval: 10s
      timeout: 5s
      retries: 10

  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: exam_api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      DB_HOST: mysql
      DB_PORT: 3306
      DB_NAME: airquality
      DB_USER: airuser
      DB_PASS: airpass
    depends_on:
      mysql:
        condition: service_healthy
    volumes:
      - ./step_4_regression/artifacts:/app/step_4_regression/artifacts:ro
      - ./step_5_classification/artifacts:/app/step_5_classification/artifacts:ro
      - ./data:/app/data:ro

  webapp:
    build:
      context: .
      dockerfile: Dockerfile.webapp
    container_name: exam_webapp
    restart: unless-stopped
    ports:
      - "8501:8501"
    environment:
      API_BASE_URL: http://api:8000
    depends_on:
      - api

volumes:
  mysql_data:
```

**`Dockerfile.api`:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY shared/ ./shared/
COPY step_4_regression/ ./step_4_regression/
COPY step_5_classification/ ./step_5_classification/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`Dockerfile.webapp`:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp/ ./webapp/

EXPOSE 8501
CMD ["streamlit", "run", "webapp/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 14.2 Comandi

```bash
# Build e avvio completo
docker compose up --build

# Solo API in sviluppo locale
uvicorn api.main:app --reload --port 8000

# Solo Streamlit in sviluppo locale
streamlit run webapp/app.py
```

---

## 15. GitHub Actions

### 15.1 Workflow CI — `test.yml`

Trigger: push su `master` o `main`, pull request.

```yaml
# .github/workflows/test.yml
name: CI

on:
  push:
    branches: [master, main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: pytest tests/ -v
```

### 15.2 Workflow Docker — `docker-publish.yml`

Trigger: push su `master` / tag `v*.*.*`.  
Pubblica le immagini su GitHub Container Registry (GHCR).

```yaml
# .github/workflows/docker-publish.yml
name: Docker Publish

on:
  push:
    branches: [master, main]
    tags: ["v*.*.*"]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push API image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.api
          push: true
          tags: ghcr.io/${{ github.repository }}/api:latest

      - name: Build and push Webapp image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.webapp
          push: true
          tags: ghcr.io/${{ github.repository }}/webapp:latest
```

---

## 16. summary.ipynb

Notebook Jupyter che racconta l'intero progetto in forma narrativa. Struttura consigliata:

### Sezioni

| # | Sezione | Contenuto |
|---|---------|-----------|
| 1 | **Introduzione** | Obiettivo, dataset, pipeline overview |
| 2 | **Raccolta Dati** | Esempio chiamata ARPA Lombardia + Open-Meteo, preview JSON |
| 3 | **Feature Engineering** | Descrizione feature, lag/rolling, encoding ciclico, feature spaziali |
| 4 | **EDA** | Carica `daily_dataset.parquet`, mostra i 13 plot principali (distribuzione PM10, heatmap correlazioni, stagionalità, trend) |
| 5 | **Regressione PM10** | Carica `regression_metrics.json`, confronto MAE/RMSE/R² tra 3 modelli, feature importance XGBoost |
| 6 | **Classificazione Allerta** | Carica `classification_metrics.json`, confronto F1-macro, confusion matrix best model |
| 7 | **Predizione Futura (demo)** | Chiama `GET /forecast?station_id=501&days=1` via `requests`, mostra output |
| 8 | **Conclusioni** | Limiti del modello, possibili miglioramenti |

### Note implementative

- Tutto in **read-only**: il notebook non addestra nulla, carica solo artefatti già salvati.
- Celle di testo in inglese (per uniformità con il README).
- Ogni sezione inizia con una cella Markdown che spiega il contesto.
- Plot mostrati con `matplotlib`/`seaborn` inline (`%matplotlib inline`).

---

## 17. README.md

Il README alla root del repository deve coprire:

### Struttura consigliata

```markdown
# AriaPulita — Air Quality Forecasting for Lombardy

Short description (2-3 righe): cosa fa il progetto, quali dati usa, cosa predice.

## Project Structure
(albero directory semplificato)

## Pipeline Overview
(diagramma ASCII — già presente in technical_doc.md sezione 2)

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+

### Run with Docker
docker compose up --build
# API:     http://localhost:8000/docs
# Webapp:  http://localhost:8501

### Run locally (development)
pip install -r requirements.txt
# 1. Start MySQL
docker compose up mysql -d
# 2. Run ingestion
python -m step_2_ingestion.ingest
# 3. Build dataset
python -m step_3_eda.build_dataset
# 4. Train regression
python -m step_4_regression.train
python -m step_4_regression.evaluate
# 5. Train classification
python -m step_5_classification.train
python -m step_5_classification.evaluate
# 6. Start API + webapp
uvicorn api.main:app --reload
streamlit run webapp/app.py

## API Reference
Tabella endpoint principali + link a http://localhost:8000/docs

## Data Sources
- ARPA Lombardia Socrata API
- Open-Meteo Historical Forecast API
- OpenStreetMap (industrial zones via Overpass API)

## Models
Tabella breve: modello, task, metrica, score ottenuto

## License
```

---

## 18. Stato di Avanzamento

| Step | Descrizione | Stato |
|------|-------------|-------|
| Step 1 | Raccolta dati — `backfill.py` storico 1 anno | ✅ Completato |
| Step 2 | Ingestion — MySQL, schema, ingest.py | ✅ Completato |
| Step 3 | Feature engineering, EDA, `daily_dataset_clean.parquet` | ✅ Completato |
| Step 4 | Regressione — 3 modelli trainati, metriche salvate | ✅ Completato |
| Step 5 | Classificazione — 3 modelli | 🔲 Da fare |
| Step 6 | REST API — FastAPI `api/` | 🔲 Da fare |
| Step 6 | Web Interface — Streamlit `webapp/` | 🔲 Da fare |
| Docker | `Dockerfile.api`, `Dockerfile.webapp`, `compose.yaml` root | 🔲 Da fare |
| CI/CD | `.github/workflows/test.yml` + `docker-publish.yml` | 🔲 Da fare |
| Docs | `summary.ipynb`, `README.md` | 🔲 Da fare |


