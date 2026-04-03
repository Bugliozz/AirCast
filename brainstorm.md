# AriaPulita — Brainstorm Progetto

## 💡 Idea Centrale

Raccogliere dati orari sulla qualità dell'aria dalle stazioni ARPA Lombardia, arricchirli con dati meteo da Open-Meteo, e addestrare modelli di Machine Learning per prevedere il valore continuo di PM10 (Regressione) e classificare il livello di allerta giornaliero (Classificazione) in 4 categorie:

- **Verde** (buono)
- **Giallo** (accettabile)
- **Arancio** (mediocre)
- **Rosso** (scarso/pessimo)

**Soglie di riferimento:** Direttiva europea 2008/50/CE + limiti OMS 2021.

---

## 🏗️ Architettura della Pipeline (Flusso Logico)

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
│  ├─ Gestione Sbilanciamento: SMOTE vs class_weight='balanced'    │
│  ├─ Training: Random Forest, KNN, XGBoost (metrica: F1-macro)    │
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

## 📁 Struttura del Progetto

```
exam_project/
├── main.py                    # Launcher principale per orchestrare la pipeline
├── requirements.txt           # Dipendenze globali
├── data/                      # Cartella locale per i dati grezzi scaricati
│   └── raw/                   # JSON/CSV generati dallo Step 1
│
├── step_1_collection/         # Ex Step 3 (Scraping locale)
│   ├── collector.py           # Estrazione giornaliera
│   └── backfill.py            # Loop per scaricare lo storico (es. 1 anno)
│
├── step_2_ingestion/          # Ex Step 2 (Database ETL)
│   ├── compose.yaml           # MySQL + Metabase
│   ├── schema.sql             # Init MySQL
│   └── ingest.py              # Legge da data/raw/ e popola i DB
│
├── step_3_eda/                # Ex Step 1 (Esplorazione)
│   ├── eda.py
│   └── plots/
│
├── step_4_regression/         # Regressione continua PM10
│   ├── train.py
│   └── artifacts/             # Modelli salvati (.pkl / .joblib)
│
├── step_5_classification/     # Classificazione allerta 4 classi
│   ├── train.py
│   ├── evaluate.py
│   └── artifacts/
│
└── api/                       # Flask API unificata per Step 4 e Step 5
    ├── app.py                 # Endpoint /predict_value e /predict_alert
    └── requirements_api.txt
```

---

## 📊 Fonti Dati

### 1. ARPA Lombardia — Socrata Open Data API

**Misure orarie:**
- https://www.dati.lombardia.it/resource/nicp-bhqi.json (PM10, PM2.5, NO2, O3, CO)

**Anagrafica sensori:**
- https://www.dati.lombardia.it/resource/ib47-atvt.json

**Filtri:** `$where` per data, stato = 'VA' per misure validate.

⚠️ **Scoperta EDA:** I sensori PM10 e PM2.5 restituiscono un solo valore giornaliero (timestamp T00:00:00), non dati orari. NO2, O3 e CO hanno letture orarie reali.

**Implicazioni:**
- Il target dei modelli sarà il valore giornaliero
- NO2 verrà usato come proxy orario per le fasce di traffico (rush-hour 7-9 e 20-22)
- Le feature meteo andranno aggregate come medie giornaliere per allinearsi al PM10

### 2. Open-Meteo (Gratuita, no API key)

**URL:** https://api.open-meteo.com/v1/forecast

**Feature meteorologiche raccolte (hourly):**

- `temperature_2m` — temperatura al suolo
- `relative_humidity_2m` — favorisce crescita igroscopica PM e aerosol secondario
- `dew_point_2m` — per calcolo fog proxy (T - Td < 2°C)
- `precipitation` — effetto washout sul PM10
- `surface_pressure` — alta pressione persistente = stagnazione inquinanti
- `cloud_cover` — copertura nuvolosa, legata a inversioni termiche notturne
- `wind_speed_10m` — dispersione degli inquinanti
- `wind_direction_10m` — direzione del vento (vento da zona industriale vs. Alpi)
- `visibility` — proxy nebbia/foschia
- `shortwave_radiation` — irraggiamento solare, guida fotochimica O3 e proxy inversione
- `boundary_layer_height` (BLH) — la più importante: se basso, gli inquinanti ristagnano

---

## 🔧 Feature Engineering e Preprocessing

### Temporali & Spaziali

- **Mese / stagione** — nebbia padana in autunno/inverno
- **Giorno della settimana** — domenica = traffico minimo, festivo / pre-festivo
- **Provincia / zona** — urbana, suburbana, rurale, industriale
- **Quota (m slm)**
- **Feature Derivata (da GeoPandas):** Distanza dalla zona industriale OSM più vicina (`dist_industrial_km`) e conteggio zone entro 15km (`n_industrial_zones_15km`), calcolate via spatial join con proiezione UTM 32N.

### Derivate da Variabili Meteo

- **Fog proxy:** flag binario dove (temperature_2m − dew_point_2m) < 2°C
- **Stagnation index:** combinazione di surface_pressure alta + wind_speed basso + BLH basso → identifica episodi di accumulo
- **Lag features:** PM10 e variabili meteo chiave del giorno precedente (1-day, 2-day lag) — il PM10 ha forte autocorrelazione temporale
- **Rolling means:** media mobile a 3–7 giorni di surface_pressure e wind_speed, per catturare la persistenza anticiclonica

### Missing Values

- **Eliminazione:** colonne/stazioni con >50% missing
- **Imputazione Serie Temporali:** Solo `forward fill` (`ffill`) per inquinanti e feature derivate (lag, rolling).
  - **Nota di Progettazione:** L'uso di `backward fill` (`bfill`) è stato evitato deliberatamente. Sebbene riempirebbe i `NaN` a inizio serie, introdurrebbe **data leakage**, "guardando nel futuro" per riempire un dato mancante. Questo doperebbe le performance del modello in validazione ma lo renderebbe inaffidabile in produzione.
- **Imputazione Meteo:** Mediana, ma **calcolata esclusivamente sul training set** (vedi sezione Anti-Leakage sotto).
- **Target:** Eliminazione righe in caso di target mancante (nessuna imputazione)

### Anti-Leakage: Pipeline in Due Fasi

La pipeline di preprocessing è stata progettata in due fasi per prevenire il data leakage:

**Fase 1 — `clean_dataset()` (sicura, pre-split):**
Operazioni che non usano statistiche globali e possono essere eseguite sull'intero dataset prima del train/test split:
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
1. **Stagnation flag:** Usava `df.median()` sull'intero dataset per determinare "alta pressione" e "vento debole". Sostituito con soglie meteorologiche fisse: pressione > 1013.25 hPa, vento < 1.5 m/s, BLH < 500 m.
2. **Imputazione mediana globale:** I NaN residui (dopo ffill) venivano riempiti con `df[col].median()` calcolata sull'intero dataset, includendo il test set. Ora la mediana è calcolata solo sul training set e applicata al test set tramite `impute_missing()`.

**Operazioni verificate come sicure (nessun leakage):**
- `shift(1)`, `shift(2)` — lag, usa solo dati passati
- `rolling(3, min_periods=1).mean()` — finestra [t-2, t-1, t], il meteo corrente è disponibile da forecast
- `ffill` per stazione — propaga solo valori passati
- Soglie allerta PM10 — costanti fisse da direttiva EU
- Feature temporali (mese, stagione, giorno settimana) — deterministiche dalla data

### Target

- **Regressione:** Valore PM10 giornaliero (µg/m³)
- **Classificazione:** {0: verde, 1: giallo, 2: arancio, 3: rosso} in base a soglie su media giornaliera PM10

---

## 🤖 Scelte di Machine Learning

### Setup Generale

- **Split:** 70/30 stratified holdout split (Test set usato UNA SOLA VOLTA alla fine)
- **Cross-Validation (Training):**
  - Classificazione: StratifiedKFold a 5 fold
  - Regressione: TimeSeriesSplit a 5 fold (evita il data leakage temporale)
- **Feature Selection:** Non applicata esplicitamente. L'utilizzo di modelli ad albero (RF, XGB) e regolarizzazioni L1 (Elastic Net) gestirà la selezione implicitamente. Viene applicato solo un VarianceThreshold(0.001) difensivo.

### Step 4: Regressione

**Obiettivo:** Prevedere il valore continuo del PM10

- **Modelli:** Random Forest (baseline), Elastic Net (lineare interpretabile), XGBoost (performance top)
- **Metrica Tuning:** neg_root_mean_squared_error (penalizza fortemente i grandi errori, evitando sottostime pericolose nei giorni di picco inquinamento)
- **Valutazione Finale:** RMSE, MAE, R²

### Step 5: Classificazione

**Obiettivo:** Prevedere la fascia di allerta (4 classi sbilanciate)

- **Gestione Sbilanciamento:** Confronto empirico in CV tra class_weight='balanced' nativo degli algoritmi e SMOTE (oversampling sintetico)
- **Modelli:** Random Forest, KNN, XGBoost
- **Metrica Tuning:** F1-macro. Bilancia Precision e Recall su tutte le classi equamente, assicurandosi che le allerte rosse (fondamentali per la salute, ma minoritarie come frequenza) pesino tanto quanto le giornate verdi

---

## 🌐 Utilizzo Database Avanzati

### GeoPandas + OpenStreetMap (Feature Spaziali)

I poligoni `landuse=industrial` vengono scaricati da OpenStreetMap (Overpass API) per l'intera Lombardia e salvati in GeoJSON (`data/raw/industrial_zones.geojson`).
GeoPandas e Shapely calcolano, per ogni stazione ARPA, la distanza dal bordo della zona industriale più vicina (`dist_industrial_km`) e il numero di zone entro 15km (`n_industrial_zones_15km`).
La proiezione EPSG:32632 (UTM 32N) garantisce distanze metriche accurate. Il risultato viene salvato in `data/raw/industrial_proximity.parquet` e unito al dataset giornaliero nello Step 3.

---

## ✅ TODO Attuali

- [x] Brainstorming e riorganizzazione della pipeline locale
- [x] step_1_collection: Scrivere backfill.py per scaricare almeno 1 anno di storico in data/raw/
- [x] step_2_ingestion: Preparare compose.yaml (MySQL, Metabase, Neo4j) e lo script di caricamento ETL
- [x] step_3_eda: Eseguire script per missing values, class distribution e correlazioni
- [ ] step_4_regression: Addestrare modelli, fare tuning, salvare pipeline
- [ ] step_5_classification: Addestrare classificatore 4 classi testando SMOTE
- [ ] api: Sviluppare app Flask per esporre i modelli addestrati
