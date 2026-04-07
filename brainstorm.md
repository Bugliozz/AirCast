# AriaPulita — Piano di Progetto

> **Legenda difficoltà / modello LLM consigliato (task operativi):**
> 🟢 Facile → Haiku | 🟠 Medio → Sonnet | 🔴 Difficile/Lungo → Opus

---

## 1. Idea Centrale

Raccogliere dati orari sulla qualità dell'aria dalle stazioni ARPA Lombardia, arricchirli con dati meteo da Open-Meteo, e addestrare modelli di Machine Learning per prevedere il valore continuo di PM10 (Regressione) e classificare il livello di allerta giornaliero (Classificazione) in 4 categorie:

- **Verde** (buono)
- **Giallo** (accettabile)
- **Arancio** (mediocre)
- **Rosso** (scarso/pessimo)

**Soglie di riferimento:** Direttiva europea 2008/50/CE + limiti OMS 2021.

---

## 2. Architettura della Pipeline

Il progetto segue un flusso lineare e locale, orchestrato tramite script Python e container Docker.

```
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1: Raccolta Dati & Backfill (Scraping API)                  │
│  ├─ Script Python locali (backfill storico o esecuzione daily)   │
│  ├─ ARPA Lombardia Socrata API (PM10, PM2.5, NO2, O3, CO)        │
│  ├─ Open-Meteo (temp, precipitazioni, vento, visibilità, BLH)    │
│  └─ Output: File JSON/CSV salvati in locale (es. data/raw/)      │
└──────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 2: Storage & Ingestione (ETL)                               │
│  ├─ Docker Compose locale per i database                         │
│  ├─ MySQL: Tabelle strutturate (misure, stazioni, meteo)         │
│  └─ GeoPandas/Shapely: Distanza stazioni-zone industriali (OSM)  │
└──────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3: Exploratory Data Analysis (EDA)                          │
│  ├─ Lettura dati puliti da MySQL via Pandas                      │
│  ├─ Analisi missing values e distribuzioni (sbilanciamento)      │
│  ├─ Heatmap stazioni × ora, correlazioni meteo-inquinanti        │
│  └─ Output: Grafici e insight salvati in cartella plots/         │
└──────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4: Machine Learning — Regressione                           │
│  ├─ Target: Valore PM10 giornaliero (µg/m³)                      │
│  ├─ Training: Random Forest, Elastic Net, XGBoost                │
│  └─ Deploy: Flask API locale (GET /predict_value?stazione=...)   │
└──────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 5: Machine Learning — Classificazione                       │
│  ├─ Target: 4 classi di allerta (verde, giallo, arancio, rosso)  │
│  ├─ Gestione Sbilanciamento: class_weight='balanced'             │
│  ├─ Training: Random Forest, Logistic Regression, XGBoost        │
│  └─ Deploy: Estensione Flask API (GET /predict_alert?...)        │
└──────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 6: Visualizzazione (Metabase)                               │
│  ├─ Connesso direttamente a MySQL                                │
│  ├─ KPI: % giorni verdi per provincia                            │
│  └─ Mappe e trend temporali per inquinante                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Struttura del Progetto

```
exam_project/
├── main.py                    # Launcher principale per orchestrare la pipeline
├── requirements.txt           # Dipendenze globali
├── data/                      # Cartella locale per i dati grezzi scaricati
│   └── raw/                   # JSON/CSV generati dallo Step 1
│
├── step_1_collection/
│   ├── collector.py           # Estrazione giornaliera
│   └── backfill.py            # Loop per scaricare lo storico (es. 1 anno)
│
├── step_2_ingestion/
│   ├── compose.yaml           # MySQL + Metabase
│   ├── schema.sql             # Init MySQL
│   └── ingest.py              # Legge da data/raw/ e popola i DB
│
├── step_3_eda/
│   ├── eda.py
│   ├── build_dataset.py       # Feature engineering + clean_dataset() + impute_missing()
│   └── plots/
│
├── step_4_regression/
│   ├── __init__.py
│   ├── config.py              # Iperparametri, search space
│   ├── train.py               # Pipeline di training
│   ├── evaluate.py            # Valutazione finale su test set
│   └── artifacts/             # Modelli salvati, metriche, plot
│
├── step_5_classification/
│   ├── __init__.py
│   ├── config.py              # Iperparametri, class weights
│   ├── train.py               # Pipeline di training
│   ├── evaluate.py            # Valutazione finale su test set
│   └── artifacts/             # Modelli salvati, metriche, plot
│
├── shared/
│   └── utils.py               # Funzioni condivise (split, imputation, preprocessor)
│
└── api/
    ├── app.py                 # Endpoint /predict_value e /predict_alert
    └── requirements_api.txt
```

---

## 4. Fonti Dati

### 4.1 ARPA Lombardia — Socrata Open Data API

**Misure orarie:**
- `https://www.dati.lombardia.it/resource/nicp-bhqi.json` (PM10, PM2.5, NO2, O3, CO)

**Anagrafica sensori:**
- `https://www.dati.lombardia.it/resource/ib47-atvt.json`

**Filtri:** `$where` per data, stato = 'VA' per misure validate.

> **Scoperta EDA:** I sensori PM10 e PM2.5 restituiscono un solo valore giornaliero (timestamp T00:00:00), non dati orari. NO2, O3 e CO hanno letture orarie reali.

**Implicazioni:**
- Il target dei modelli sarà il valore giornaliero
- NO2 verrà usato come proxy orario per le fasce di traffico (rush-hour 7-9 e 20-22)
- Le feature meteo andranno aggregate come medie giornaliere per allinearsi al PM10

### 4.2 Open-Meteo (Gratuita, no API key)

**URL:** `https://api.open-meteo.com/v1/forecast`

**Feature meteorologiche raccolte (hourly → aggregate giornaliere):**

| Feature | Significato fisico |
|---|---|
| `temperature_2m` | Temperatura al suolo |
| `relative_humidity_2m` | Favorisce crescita igroscopica PM e aerosol secondario |
| `dew_point_2m` | Per calcolo fog proxy (T − Td < 2°C) |
| `precipitation` | Effetto washout sul PM10 |
| `surface_pressure` | Alta pressione persistente = stagnazione inquinanti |
| `cloud_cover` | Copertura nuvolosa, legata a inversioni termiche notturne |
| `wind_speed_10m` | Dispersione degli inquinanti |
| `wind_direction_10m` | Direzione del vento (zona industriale vs. Alpi) |
| `visibility` | Proxy nebbia/foschia |
| `shortwave_radiation` | Irraggiamento solare, guida fotochimica O3 e proxy inversione |
| `boundary_layer_height` (BLH) | **Più importante:** se basso, gli inquinanti ristagnano |

---

## 5. Feature Engineering e Preprocessing

### 5.1 Feature Base

#### Temporali & Spaziali

- **Mese / stagione** — nebbia padana in autunno/inverno
- **Giorno della settimana** — domenica = traffico minimo, festivo / pre-festivo
- **Provincia / zona** — urbana, suburbana, rurale, industriale
- **Quota (m slm)**
- **Feature Derivata (da GeoPandas):** Distanza dalla zona industriale OSM più vicina (`dist_industrial_km`) e conteggio zone entro 15km (`n_industrial_zones_15km`), calcolate via spatial join con proiezione UTM 32N.

#### Derivate da Variabili Meteo

- **Fog proxy:** flag binario dove `(temperature_2m − dew_point_2m) < 2°C`
- **Stagnation flag (booleano):** soglie fisse — pressione > 1013.25 hPa, vento < 1.5 m/s, BLH < 500 m
- **Stagnation index (continuo):** `1 / (wind_speed_mean × blh_min × (1 + precip_sum))` + clipping per divisioni per zero
- **Lag features:** PM10 e variabili meteo chiave del giorno precedente (lag 1, lag 2) — il PM10 ha forte autocorrelazione temporale
- **Rolling means:** media mobile a 3–7 giorni di `surface_pressure` e `wind_speed`, per catturare la persistenza anticiclonica

### 5.2 Feature Engineering Aggiuntivo

> Modifiche a `step_3_eda/build_dataset.py` e/o `shared/utils.py`.

- [x] **1.1 — Encoding ciclico sin/cos per mese e giorno della settimana** 🟢
  - `mese_sin = sin(2π × mese / 12)`, `mese_cos = cos(2π × mese / 12)`
  - `dow_sin = sin(2π × giorno_settimana / 7)`, `dow_cos = cos(2π × giorno_settimana / 7)`
  - Motivo: il modello deve sapere che Dicembre (12) e Gennaio (1) sono vicini

- [x] **1.2 — Indice di stagnazione continuo** 🟢
  - `stagnation_index = 1 / (wind_speed_mean × blh_min × (1 + precip_sum))`
  - Clippare a un massimo ragionevole (aggiungere epsilon per evitare divisioni per zero)
  - Mantenere anche il `stagnation_flag` booleano per interpretabilità

- [x] **1.3 — Rolling media 7 giorni del PM10** 🟢
  - `pm10_roll7 = pm10.rolling(7, min_periods=1).mean()` per stazione
  - Cattura il trend di medio periodo (accumulo persistente)

- [x] **1.4 — Aggiornare `requirements.txt`** 🟢
  - Aggiungere: `scikit-learn >= 1.4`, `xgboost >= 2.0`, `joblib`

### 5.3 Gestione Missing Values

- **Eliminazione:** colonne/stazioni con >50% missing
- **Imputazione Serie Temporali:** Solo `forward fill` (`ffill`) per inquinanti e feature derivate (lag, rolling).
  - `backward fill` (`bfill`) **evitato deliberatamente**: riempirebbe i NaN a inizio serie ma introdurrebbe **data leakage**, guardando nel futuro per riempire un dato mancante.
- **Imputazione Meteo:** Mediana calcolata **esclusivamente sul training set** (vedi sezione Anti-Leakage).
- **Target:** Eliminazione righe in caso di target mancante (nessuna imputazione).

### 5.4 Anti-Leakage: Pipeline in Due Fasi

La pipeline di preprocessing è progettata in due fasi per prevenire il data leakage.

**Fase 1 — `clean_dataset()` (sicura, pre-split):**
Operazioni che non usano statistiche globali, eseguite sull'intero dataset prima del train/test split:
- Drop colonne con >50% missing
- Drop stazioni con scarsa copertura PM10
- Forward-fill per stazione (usa solo dati passati)
- Drop righe con target mancante
- Output: `daily_dataset_clean.parquet` (può contenere NaN residui)

**Fase 2 — `impute_missing()` (da eseguire DOPO lo split):**
Imputazione basata sulla mediana con pattern fit/transform:
```python
# Nel training
train_imputed, medians = impute_missing(train_df)
# Nel test — usa le mediane del training
test_imputed, _ = impute_missing(test_df, medians=medians)
```

**Problemi di leakage risolti:**
1. **Stagnation flag:** Usava `df.median()` sull'intero dataset. Sostituito con soglie meteorologiche fisse (P > 1013.25 hPa, wind < 1.5 m/s, BLH < 500 m).
2. **Imputazione mediana globale:** I NaN residui venivano riempiti con `df[col].median()` calcolata sull'intero dataset. Ora la mediana è calcolata solo sul training set.

**Operazioni verificate come sicure:**
- `shift(1)`, `shift(2)` — lag, usa solo dati passati
- `rolling(3, min_periods=1).mean()` — finestra [t−2, t−1, t]
- `ffill` per stazione — propaga solo valori passati
- Soglie allerta PM10 — costanti fisse da direttiva EU
- Feature temporali (mese, stagione, giorno settimana) — deterministiche dalla data

### 5.5 Target

- **Regressione:** Valore PM10 giornaliero (µg/m³) — continuo
- **Classificazione:** `{0: verde, 1: giallo, 2: arancio, 3: rosso}` in base a soglie su media giornaliera PM10 (direttiva EU 2008/50/CE)

---

## 6. Stato Attuale e Dataset

**Dataset disponibile:** `step_3_eda/daily_dataset_clean.parquet`
- Granularità: **giornaliera** (1 riga per stazione per giorno)
- Target regressione: `pm10` (µg/m³)
- Target classificazione: `classe_allerta` (verde / giallo / arancio / rosso)
- ~65–75 colonne, ~10k–30k righe (stazioni × giorni validi)
- Periodo: marzo 2025 – marzo 2026 (~13 mesi)

**Feature già implementate in `build_dataset.py`:**
- Lag: `pm10_lag1`, `pm10_lag2`, lag meteo (pressure, wind, BLH, temp)
- Rolling: `pressure_roll3`, `wind_speed_roll3`
- Stagnation flag booleano (soglie fisse)
- `is_weekend`, `fog_hours`, `mese`, `stagione`, `giorno_settimana`
- Spaziali: `dist_industrial_km`, `n_industrial_zones_15km`

---

## 7. Setup ML e Infrastruttura Condivisa

### 7.1 Setup Generale

- **Split:** Temporal holdout 70/30 — ordinare per `data_giorno`, prendere i primi 70% dei giorni come train, il restante 30% come test. Lo split è per giorno (NON per riga): tutte le stazioni dello stesso giorno vanno nello stesso set. **Niente shuffle, niente split casuale.**
- **Cross-Validation (Training):** `TimeSeriesSplit` a 5 fold per **entrambi** regressione e classificazione — evita data leakage temporale.
- **Feature Selection:** Non applicata esplicitamente. I modelli ad albero (RF, XGBoost) e la regolarizzazione L1/L2 (Elastic Net, Logistic Regression) gestiscono la selezione implicitamente. Viene applicato solo un `VarianceThreshold(0.001)` difensivo.

### 7.2 Infrastruttura Condivisa (`shared/utils.py`)

> File singolo riutilizzabile da Step 4 e Step 5.

- [x] **2.1 — Creazione `shared/utils.py`** 🟢
  - Scaffold del file con import e docstring

- [x] **2.2 — Split holdout temporale** 🟠
  - Funzione `temporal_train_test_split()`
  - Ordinare per `data_giorno`, prendere i primi 70% dei giorni distinti come train
  - Restituire `(X_train, y_train, X_test, y_test)` con le date di cutoff loggate

- [x] **2.3 — Imputation post-split** 🟢
  - Flusso: `clean_parquet` → split temporale → `impute_missing(train)` → `impute_missing(test, medians=train_medians)`

- [x] **2.4 — ColumnTransformer per preprocessing** 🟠
  - Feature numeriche per modelli lineari: `StandardScaler`
  - Feature numeriche per tree-based: `passthrough`
  - Feature categoriche (`stagione`, `provincia`): `OrdinalEncoder` o `OneHotEncoder`
  - Drop colonne identificative: `idstazione`, `nomestazione`, `comune`, `data_giorno`
  - **Bugfix:** `OrdinalEncoder` riceveva `categories=[list, "auto"]` (mix non valido per sklearn). Le categorie di colonne non-`stagione` vengono ora derivate con `sorted(X[col].unique())` al momento della costruzione del preprocessor.

- [x] **2.5 — TimeSeriesSplit wrapper** 🔴
  - Funzione che riceve il dataframe di training e restituisce indici per `TimeSeriesSplit(n_splits=5)`
  - **ATTENZIONE:** `TimeSeriesSplit` sklearn lavora su indici row, NON su gruppi. Con stazioni multiple per giorno, stazioni dello stesso giorno potrebbero finire in train e test contemporaneamente (leakage). Costruire gli indici manualmente: ricavare i giorni unici ordinati, splittarli in 5 fold temporali, poi espandere agli indici riga corrispondenti.
  - Verificare che ogni fold contenga almeno qualche campione per classe (log warning se mancano)

---

## 8. Step 4 — Regressione PM10

> Modulo: `step_4_regression/`

```
step_4_regression/
├── __init__.py
├── config.py       # Iperparametri, search space
├── train.py        # Pipeline di training
├── evaluate.py     # Valutazione finale su test set
└── artifacts/      # Modelli salvati, metriche, plot
```

**Obiettivo:** Prevedere il valore continuo del PM10 (µg/m³).

### 8.1 — Preparazione

- [x] **3.1.1 — Scaffold della directory e dei file** 🟢

- [x] **3.1.2 — Config iperparametri** 🟢
  - `GridSearchCV` per ElasticNet (spazio piccolo: 4×4 = 16 combinazioni)
  - `RandomizedSearchCV(n_iter=30)` per XGBoost e RandomForest
  - Motivazione: la griglia completa di XGBoost fa 3×3×3×3=81 combinazioni × 5 fold = 405 fit; con `n_iter=30` si ottiene il 90% della qualità con il 37% del costo.
  - Search space:
    - ElasticNet: `alpha` [0.01, 0.1, 1, 10], `l1_ratio` [0.1, 0.5, 0.7, 0.9]
    - XGBoost: `n_estimators` [100, 300, 500], `max_depth` [3, 5, 7], `learning_rate` [0.01, 0.05, 0.1], `subsample` [0.7, 0.8, 1.0]
    - RandomForest: `n_estimators` [100, 300, 500], `max_depth` [10, 20, None], `min_samples_leaf` [2, 5, 10]
  - **Bugfix:** `classe_allerta` non era in `DROP_COLS`, quindi finiva nelle feature numeriche causando label leakage diretto (è derivata da `pm10`, il target). Fix: aggiunta `"classe_allerta"` a `DROP_COLS` in `config.py`.

### 8.2 — Trasformazione del Target

- [x] **3.2.1 — Log-transform del target** 🟢
  - `y_train_log = np.log1p(y_train)` → `y_pred = np.expm1(y_pred_log)` per le predizioni
  - Usare `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` per incapsularlo nella pipeline
  - Motivo: stabilizza la varianza, aiuta ElasticNet sulla coda lunga

### 8.3 — Training dei Modelli Base

- [x] **3.3.1 — ElasticNet (con StandardScaler + log target)** 🟠
  - Pipeline: `StandardScaler` → `ElasticNet`
  - Wrappato in `TransformedTargetRegressor`
  - CV: `TimeSeriesSplit(5)`, scoring: `neg_root_mean_squared_error`
  - Ruolo: intercettare trend lineari, capacità di estrapolazione sui picchi

- [x] **3.3.2 — XGBoost Regressor** 🟠
  - Pipeline: nessun scaling necessario
  - CV: `TimeSeriesSplit(5)`, scoring: `neg_root_mean_squared_error`
  - Ruolo: catturare interazioni non lineari e soglie complesse

- [x] **3.3.3 — Random Forest Regressor (baseline)** 🟠
  - Pipeline: nessun scaling necessario
  - CV: `TimeSeriesSplit(5)`, scoring: `neg_root_mean_squared_error`
  - Ruolo: baseline robusta, meno sensibile a overfitting

### 8.4 — Stacking Ensemble *(opzionale, post Step 4 e Step 5)*

- [ ] **3.4.1 — StackingRegressor** 🔴
  - Base estimators: ElasticNet (best), XGBoost (best), Random Forest (best)
  - Meta-learner: `RidgeCV` (semplice, evita overfitting dello stacking)
  - `cv=TimeSeriesSplit(5)` per generare le meta-features

### 8.5 — Valutazione Finale (Test Set)

- [x] **3.5.1 — Metriche su test holdout** 🟠
  - RMSE (metrica primaria — penalizza fortemente grandi errori, evita sottostime nei giorni di picco)
  - MAE (robustezza agli outlier)
  - R² (varianza spiegata)
  - RMSE separato per fascia: verde, giallo, arancio, rosso
  - Tabella riassuntiva di tutti i modelli

- [x] **3.5.2 — Plot diagnostici** 🟠
  - Scatter: y_pred vs y_true (con linea identità)
  - Residui vs y_pred (verificare omoschedasticità)
  - Residui vs tempo (verificare assenza di trend)
  - Distribuzione dei residui (idealmente normale, centrata su 0)

- [x] **3.5.3 — Feature importance** 🟠
  - Permutation importance sul **best model** (su test set), top 15 feature
  - Per gli altri modelli: `.feature_importances_` nativa (RF, XGBoost) come sanity check

- [x] **3.5.4 — Salvataggio artifacts** 🟢
  - Best model pipeline completa: `joblib.dump()` in `artifacts/`
  - Metriche in JSON: `artifacts/regression_metrics.json`
  - Plot in PNG: `artifacts/plots/`

### 8.6 — Nota sui Risultati e Limiti del Modello

**Risultati ottenuti (test set, 70/30 split temporale):**

| Modello | R² | RMSE (µg/m³) | RMSE "rosso" |
|---|---|---|---|
| Random Forest | 0.555 | 11.9 | 19.8 |
| XGBoost | 0.521 | 12.4 | 21.2 |
| ElasticNet | -6.98 | 50.5 | 107.0 |

**Perché RMSE ~12 nonostante feature come giorno della settimana e stagione?**

`giorno_settimana`, `stagione`, `heating_season` catturano lo **shift della media attesa** tra categorie (un lunedì di gennaio tende ad avere PM10 più alto di una domenica di agosto), ma non il valore effettivo di quel giorno specifico. La varianza residua è dominata da eventi meteorologici stocastici che sovrascrivono il segnale sociale:

- Lunedì di gennaio con forte vento → PM10 basso (dispersione)
- Domenica di agosto con stagnazione → PM10 alto (accumulo)

I principali driver della varianza giornaliera non spiegata:

| Driver | Capturable con questo dataset? |
|---|---|
| Vento / BLH (dispersione) | ✅ già in dataset |
| Pioggia (wet deposition) | ✅ già in dataset |
| Inversione termica | ⚠️ parzialmente (BLH come proxy) |
| Trasporto transfrontaliero (polvere sahariana, incendi) | ❌ richiederebbe traiettorie HYSPLIT o satellite AOD |
| Traffico reale (eventi, scioperi, neve) | ❌ giorno_settimana è solo proxy statistico |

**Conclusione:** R² ~0.55 con RMSE ~12 µg/m³ è probabilmente vicino al **tetto informativo** del dataset con sole feature meteo locali. La quota di varianza inesplicata (~45%) è in gran parte rumore episodico non prevedibile senza dati aggiuntivi. ElasticNet fallisce strutturalmente (R² = -6.98) perché la relazione PM10 ↔ meteo è non-lineare e i modelli ad albero sono intrinsecamente più adatti.

---

## 9. Step 5 — Classificazione Allerta

> Modulo: `step_5_classification/`

```
step_5_classification/
├── __init__.py
├── config.py       # Iperparametri, class weights
├── train.py        # Pipeline di training
├── evaluate.py     # Valutazione finale su test set
└── artifacts/      # Modelli salvati, metriche, plot
```

**Obiettivo:** Prevedere la fascia di allerta (4 classi sbilanciate). Metrica primaria: **F1-macro** — bilancia Precision e Recall su tutte le classi equamente, assicurandosi che le allerte rosse (fondamentali per la salute, ma minoritarie) pesino tanto quanto le giornate verdi.

### 9.1 — Preparazione

- [ ] **4.1.1 — Scaffold della directory e dei file** 🟢

- [ ] **4.1.2 — Encoding del target** 🟢
  - `classe_allerta` → label encoding ordinale: verde=0, giallo=1, arancio=2, rosso=3
  - **NON usare `OrdinalEncoder` sklearn**: non garantisce l'ordine semantico corretto per default. Usare mappatura esplicita: `LABEL_MAP = {'verde': 0, 'giallo': 1, 'arancio': 2, 'rosso': 3}` con `.map()`.
  - Verificare la distribuzione delle classi nel train e nel test set (log delle percentuali)

- [ ] **4.1.3 — Config iperparametri** 🟢
  - `GridSearchCV` per Logistic Regression (spazio piccolo)
  - `RandomizedSearchCV(n_iter=30)` per RandomForest e XGBoost
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight='balanced'`
  - XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight` o sample weights
  - Logistic Regression: `C` [0.01, 0.1, 1, 10], `penalty='elasticnet'`, `class_weight='balanced'`

### 9.2 — Gestione Sbilanciamento

- [ ] **4.2.1 — Strategia: `class_weight='balanced'` (unica)** 🟢
  - Applicare nativamente a tutti i modelli che lo supportano
  - **SMOTE non utilizzato:** genera giornate meteo sintetiche fisicamente irrealistiche. Con 4 classi meteo-dipendenti, l'oversampling sintetico crea combinazioni impossibili in natura (es. alta temperatura + alta pressione + forte precipitazione).

### 9.3 — Training dei Modelli

- [ ] **4.3.1 — Random Forest Classifier** 🟠
  - `class_weight='balanced'`
  - CV: `TimeSeriesSplit(5)`, scoring: `f1_macro`

- [ ] **4.3.2 — XGBoost Classifier** 🟠
  - `scale_pos_weight` calcolato per classe, oppure sample weights
  - CV: `TimeSeriesSplit(5)`, scoring: `f1_macro`

- [ ] **4.3.3 — Logistic Regression (ElasticNet penalty)** 🟠
  - Pipeline: `StandardScaler` → `LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced')`
  - CV: `TimeSeriesSplit(5)`, scoring: `f1_macro`
  - Ruolo: contributo lineare complementare ai tree-based (analogo a ElasticNet per la regressione)

### 9.4 — Calibrazione delle Probabilità *(opzionale, post Step 4 e Step 5)*

- [ ] **4.4.1 — CalibratedClassifierCV** 🟠
  - Applicare al miglior modello dopo il tuning
  - Metodo: `'sigmoid'` (più stabile di `'isotonic'` con 4 classi su dataset non enorme)
  - Motivo: per un sistema di allerta pubblica, "73% probabilità di Rosso" è molto più utile di un semplice label

### 9.5 — Valutazione Finale (Test Set)

- [ ] **4.5.1 — Classification Report** 🟢
  - Precision, Recall, F1 per ogni classe
  - F1-macro (metrica primaria) e F1-weighted (per contesto)

- [ ] **4.5.2 — Matrice di Confusione** 🟢
  - Heatmap annotata con conteggi
  - **Focus critico:** Falsi Negativi gravi = rosso classificato come verde/giallo
  - Obiettivo salute pubblica: minimizzare i FN sulle classi pericolose

- [ ] **4.5.3 — Analisi degli errori gravi** 🟢
  - Calcolare la percentuale di errori con distanza ≥ 2 (es. rosso→verde, rosso→giallo)
  - Riportare come singola metrica di sicurezza: `severe_error_rate`

- [ ] **4.5.4 — Feature importance** 🟠
  - Permutation importance sul **best model** (su test set), top 15 feature
  - Per gli altri modelli: `.feature_importances_` nativa come sanity check

- [ ] **4.5.5 — Salvataggio artifacts** 🟢
  - Best model pipeline + eventuale calibratore: `joblib.dump()` in `artifacts/`
  - Metriche in JSON: `artifacts/classification_metrics.json`
  - Confusion matrix + report in `artifacts/plots/`

---

## 10. Cross-Analisi e Confronto

- [ ] **5.1 — Coerenza regressione-classificazione** 🟠
  - Prendere le predizioni del regressore, discretizzarle con le soglie PM10, confrontare con il classificatore
  - Se il regressore discretizzato batte il classificatore, potrebbe non servire un classificatore separato

- [ ] **5.2 — Analisi residui per stazione e stagione** 🟠
  - Il modello fallisce su certe stazioni? Certe stagioni?
  - Plot: barplot RMSE per stazione e per mese

- [ ] **5.3 — Spot check leave-one-station-out** 🟠
  - Sul test set, calcolare RMSE/F1 separatamente per ogni stazione
  - Verificare che le stazioni con performance molto peggiore della media non siano stazioni underrepresented nel training
  - Non richiede re-training: è un'analisi post-hoc delle predizioni già generate in 8.5 e 9.5

---

## 11. Note Architetturali

### 11.1 Anti-Leakage Checklist

- [ ] Split temporale puro (niente shuffle)
- [ ] Imputation mediana calcolata solo su train set
- [ ] Nessuna feature che usa informazione futura
- [ ] TimeSeriesSplit per la CV (niente KFold/StratifiedKFold)
- [ ] Lag features calcolate pre-split (usano solo passato, safe)
- [ ] Soglie stagnazione fisse (non data-dependent)
- [ ] Feature engineering su tutto il dataset solo se deterministico (sin/cos, is_weekend)
- [ ] StandardScaler fit solo su train, transform su test

### 11.2 Modelli Esclusi e Motivazione

| Modello | Motivo esclusione |
|---|---|
| LightGBM | Ridondante con XGBoost (entrambi gradient boosting), differenze trascurabili su ~10–30k righe |
| KNN | Maledizione della dimensionalità, non gestisce bene lo sbilanciamento su 4 classi |
| SMOTE | Genera campioni sintetici meteo fisicamente impossibili |
| StratifiedKFold | Mescola il tempo, causa data leakage |
| Neural Networks | Dataset troppo piccolo (~10–30k righe), rischio overfitting senza beneficio |
| NO2 disaggregato orario | Richiede nuova query SQL + join, beneficio incerto rispetto a `is_weekend` come proxy traffico |
| Binning distanza industriale | I tree-based trovano da soli le soglie ottimali sulla feature continua; il binning butta via informazione |
| Learning curves | Computazionalmente costose, risultato non azionabile (il periodo dati è fisso) |

### 11.3 Feature Spaziali — GeoPandas e OpenStreetMap

I poligoni `landuse=industrial` vengono scaricati da OpenStreetMap (Overpass API) per l'intera Lombardia e salvati in GeoJSON (`data/raw/industrial_zones.geojson`).

GeoPandas e Shapely calcolano, per ogni stazione ARPA:
- `dist_industrial_km` — distanza dal bordo della zona industriale più vicina
- `n_industrial_zones_15km` — numero di zone entro 15km

La proiezione EPSG:32632 (UTM 32N) garantisce distanze metriche accurate. Il risultato viene salvato in `data/raw/industrial_proximity.parquet` e unito al dataset giornaliero nello Step 3.

### 11.4 Dipendenze Aggiuntive

```
scikit-learn >= 1.4
xgboost >= 2.0
joblib
```

---

## 12. Ordine di Esecuzione

```
1. Feature Engineering Aggiuntivo (Sezione 5.2)
   |
2. Infrastruttura Condivisa (Sezione 7.2)
   |
   ├── 3. Step 4: Regressione (Sezione 8)
   |       8.1 Scaffold → 8.2 Log target → 8.3 Modelli base (3)
   |       → 8.5 Valutazione
   |
   └── 4. Step 5: Classificazione (Sezione 9)
           9.1 Scaffold → 9.2 Sbilanciamento → 9.3 Modelli (3)
           → 9.5 Valutazione
   |
5. Cross-Analisi (Sezione 10)
   |
6. (Opzionale) Stacking Ensemble (8.4) / Calibrazione (9.4)
```

---

## 13. Avanzamento del Progetto

| Step | Descrizione | Stato |
|---|---|---|
| Step 1 | Raccolta dati — `backfill.py` per storico 1 anno | ✅ Completato |
| Step 2 | Ingestion — `compose.yaml`, `schema.sql`, `ingest.py` | ✅ Completato |
| Step 3 | EDA — missing values, class distribution, correlazioni | ✅ Completato |
| Step 4 | Regressione — training, tuning, salvataggio pipeline | 🔲 In corso (vedi Sezione 8) |
| Step 5 | Classificazione — training, tuning, salvataggio pipeline | 🔲 Da fare (vedi Sezione 9) |
| API | Flask — endpoint `/predict_value` e `/predict_alert` | 🔲 Da fare |
| Step 6 | Visualizzazione — Metabase connesso a MySQL | 🔲 Da fare |
