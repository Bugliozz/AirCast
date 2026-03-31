# AriaPulita — Previsione qualità dell'aria in Lombardia

## Idea centrale

Raccogliere dati orari sulla qualità dell'aria dalle stazioni ARPA Lombardia,
arricchirli con dati meteo da Open-Meteo, e addestrare un modello che classifica
il livello di allerta giornaliero per inquinanti chiave (PM10, PM2.5, NO2):

```
[verde (buono) | giallo (accettabile) | arancio (mediocre) | rosso (scarso/pessimo)]
```

Soglie di riferimento: direttiva europea 2008/50/CE + limiti OMS 2021.

---

## Copertura argomenti corso

| Esercizio di riferimento            | Componente del progetto                                        |
|-------------------------------------|----------------------------------------------------------------|
| Ex1 — EDA, visualizzazione          | Dashboard esplorativa: heatmap stazioni, distribuzione classi  |
| Ex2 — ETL, MySQL, Neo4j, OpenSearch | Ingestione misure → MySQL; rete stazioni → Neo4j; bollettini → OpenSearch |
| Ex3 — Regressione + Flask API       | Modello regressione: previsione valore PM10 (µg/m³)           |
| Ex4 — Scraping + GCP                | **Due** scraper schedulati: ARPA API + Open-Meteo             |
| Ex5 — Classificazione, imbalanced   | Classificatore 4 classi, giorni rossi = minoranza → SMOTE     |

---

## Architettura complessiva

```
┌──────────────────────────────────────────────────────────────────┐
│  GCP Cloud Scheduler (ogni ora)                                  │
│         ↓                                                        │
│  Cloud Run: step_3/app.py                                        │
│   ├─ Fonte 1: ARPA Lombardia Socrata API                         │
│   │    misure orarie PM10, PM2.5, NO2, O3, CO                   │
│   └─ Fonte 2: Open-Meteo (gratuito, no key)                      │
│        temp, precipitazioni, vento, visibilità, BLH             │
│         ↓                                                        │
│  Cloud Storage (JSON raw, flat: YYYY-MM-DD_measurements/weather) │
└──────────────────────────────────────────────────────────────────┘
          ↓ (batch giornaliero)
┌──────────────────────────────────────────────────────────────────┐
│  step_2 — Ingestione (Ex2-style)                                 │
│   ├─ MySQL: tabelle misure, stazioni, meteo                      │
│   ├─ Neo4j: grafo stazioni ↔ province ↔ zone industriali        │
│   └─ OpenSearch: bollettini ARPA Lombardia (testo libero)        │
└──────────────────────────────────────────────────────────────────┘
          ↓
┌──────────────────────────────────────────────────────────────────┐
│  step_1 — EDA (Ex1-style)                                        │
│   ├─ Distribuzione classi (sbilanciamento)                       │
│   ├─ Correlazione inquinanti ↔ meteo                             │
│   ├─ Heatmap stazioni × ora del giorno                           │
│   └─ Serie temporali per linea/stagione                          │
└──────────────────────────────────────────────────────────────────┘
          ↓
┌──────────────────────────────────────────────────────────────────┐
│  step_4 — Regressione (Ex3-style)                                │
│   ├─ Target: valore PM10 orario (µg/m³)                          │
│   ├─ Feature: meteo, ora, stazione, giorno settimana, stagione   │
│   ├─ RandomizedSearchCV/GridSearchCV: RF, Elastic Net, XGBoost   │
│   └─ Flask API: GET /predict?stazione=Milano&ora=09              │
└──────────────────────────────────────────────────────────────────┘
          ↓
┌──────────────────────────────────────────────────────────────────┐
│  step_5 — Classificazione (Ex5-style)                            │
│   ├─ Target: 4 classi (verde/giallo/arancio/rosso)               │
│   ├─ SMOTE vs class_weight='balanced' (confronto)                │
│   ├─ RandomizedSearchCV/GridSearchCV: RF, KNN, XGBoost           │
│   ├─ Metrica tuning: F1-macro (bilancia tutte le classi)         │
│   └─ Confusion matrix, ROC per classe, classification report     │
└──────────────────────────────────────────────────────────────────┘
          ↓
┌──────────────────────────────────────────────────────────────────┐
│  Metabase dashboard                                              │
│   ├─ KPI: % giorni verdi per provincia                           │
│   ├─ Mappa stazioni per livello medio PM10                       │
│   └─ Trend mensile per inquinante                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Struttura cartelle

```
exam_project/
├── brainstorm.md
├── main.py                    # launcher (stesso pattern degli esercizi)
├── requirements.txt           # aggrega le dipendenze degli step
│
├── step_1/                    # EDA (Ex1-style)
│   ├── eda.py
│   └── plots/
│
├── step_2/                    # Ingestione (Ex2-style)
│   ├── ingest.py
│   ├── compose.yaml           # MySQL + Metabase + Neo4j + OpenSearch
│   └── schema.sql
│
├── step_3/                    # Scraper/Collector (Ex4-style)  ← INIZIAMO QUI
│   ├── app.py                 # Flask: /health, / (collect)
│   ├── main.py                # CLI: serve | collect
│   ├── requirements.txt
│   └── Dockerfile
│
├── step_4/                    # Regressione (Ex3-style)
│   ├── train.py
│   ├── app.py                 # Flask API /predict
│   └── artifacts/
│
└── step_5/                    # Classificazione (Ex5-style)
    ├── train.py
    ├── evaluate.py
    └── artifacts/
```

---

## Fonti dati

### ARPA Lombardia — Socrata Open Data API

| Risorsa | Endpoint | Note |
|---------|----------|------|
| Misure orarie | `https://www.dati.lombardia.it/resource/nicp-bhqi.json` | PM10, PM2.5, NO2, O3, CO |
| Anagrafica sensori | `https://www.dati.lombardia.it/resource/ib47-atvt.json` | Stazioni con lat/lng |

Query parametri Socrata:
- `$where`: filtro per data e tipo inquinante
- `$limit`: fino a 500.000 righe
- `stato = 'VA'`: solo misure validate

> **⚠️ Scoperta EDA (step_1):** i sensori PM10 e PM2.5 di ARPA Lombardia restituiscono
> **un solo valore giornaliero** per stazione (timestamp `T00:00:00`), NON dati orari.
> Al contrario, NO2, O3 e CO hanno **letture orarie reali** (24 valori/giorno).
>
> Implicazioni:
> - Le feature "ora del giorno" e "fascia traffico" **non si applicano a PM10** direttamente
> - Per l'analisi intraday si usa **NO2 come proxy del traffico** (profilo orario con doppio
>   picco rush-hour 7-9 e 20-22 chiaramente visibile)
> - Il target di regressione (step_4) diventa il **valore PM10 giornaliero** (non orario)
> - Le feature meteo vanno aggregate come **media giornaliera** per il join con PM10
> - Le feature orarie di NO2/O3 (media, max, min, std per giorno) possono alimentare
>   il modello come feature aggiuntive

### Open-Meteo (gratuita, no API key)

URL: `https://api.open-meteo.com/v1/forecast`

Feature rilevanti:
- `temperature_2m` — temperatura (°C)
- `precipitation` — pioggia (mm/h)
- `wind_speed_10m` — vento (km/h, disperde gli inquinanti)
- `visibility` — visibilità (m, proxy nebbia in Pianura Padana)
- `boundary_layer_height` — altezza strato limite (m) — la feature più importante:
  più è basso, più gli inquinanti sono intrappolati vicino al suolo

---

## Feature engineering

**Temporali:**
- ora del giorno (fascia traffico: 7-9, 17-19)
- giorno della settimana (domenica = traffico minimo)
- mese / stagione (nebbia padana: ottobre–febbraio)
- festivo / pre-festivo

**Stazione:**
- provincia / zona (urbana, suburbana, rurale, industriale)
- quota (m slm)
- distanza da autostrada / zona industriale (ricavabile da OpenRailwayMap/OSM)

**Meteo (da Open-Meteo, per stazione):**
- temperatura
- precipitazioni (pioggia lava il PM10)
- velocità vento (disperde gli inquinanti)
- visibilità (nebbia → inversione termica → accumulo PM)
- boundary layer height (BLH) — principale proxy di dispersione atmosferica

**Target:**
- Classificazione: `{0: verde, 1: giallo, 2: arancio, 3: rosso}`
  basata su media giornaliera PM10 vs soglie EU/OMS
- Regressione: valore PM10 giornaliero (µg/m³)

---

## Pipeline ML — Decisioni

### Cross-validation


```
70/30 stratified holdout split (seed fisso)
├── Train (70%): tuning iperparametri via CV a 5 fold
│   ├─ StratifiedKFold per classificazione (preserva proporzioni classi)
│   └─ TimeSeriesSplit(n_splits=5) per regressione (rispetta ordine temporale,
│      evita leakage: il modello non "vede" dati futuri nel train fold)
└── Test (30%): toccato UNA SOLA VOLTA per valutazione finale
```

- **RandomizedSearchCV** per XGBoost (griglia grande, troppe combinazioni per grid esaustiva)
- **GridSearchCV** per RF, Elastic Net, KNN (griglia contenuta)
- **Nested CV**: non necessaria per un progetto d'esame; aggiunge complessità
  computazionale (CV dentro CV) senza beneficio proporzionato a questo livello

### Feature selection

**Non la facciamo.** Motivazioni:
- Elastic Net fa selezione implicita (L1 porta a zero i coefficienti irrilevanti)
- RF e XGBoost ignorano naturalmente le feature non informative via importanza/split
- Lo spazio feature è piccolo e curato (~15-25 feature): non siamo in un setting
  high-dimensional (genomica, NLP) dove la selezione è critica
- Meglio investire tempo in **feature engineering** (lag PM10, is_rush_hour, stagione, ecc.)

Unica misura difensiva: `VarianceThreshold(threshold=0.001)` per eliminare colonne costanti
o quasi-costanti (coerente con ex5).

### Missing values

I dati ARPA hanno buchi attesi (manutenzione sensori, guasti). Strategia:

1. **EDA**: calcolare tasso di missingness per colonna, stazione, periodo
2. **Drop colonne** con >50% missing (stesso approccio del report ML precedente)
3. **Imputazione**:
   - Feature temporali (misure inquinanti): `forward fill` → `backward fill`
     (dato time-series, la lettura delle 9 è più legata a quella delle 8 che alla mediana globale)
   - Feature meteo: imputazione con mediana (meno autocorrelate)
   - Fit dell'imputer **solo su train**, applicazione su test (no leakage)
4. **Mai imputare il target**: drop delle righe dove manca

### Addestramento locale vs GCloud

**Addestramento locale.** Il dataset è ~100 stazioni × 24 ore × 365 giorni ≈ 876K righe.
GridSearchCV con XGBoost su questa scala richiede minuti, non ore.
GCloud resta dedicato al collector/deployment (step_3 già deployato).

### Regressione (step_4) — Modelli

| Modello | Ruolo | Motivazione |
|---------|-------|-------------|
| **Random Forest** | Baseline non lineare | Robusto, gestisce non-linearità, tuning minimo |
| **Elastic Net** | Lineare interpretabile | Mostra quali feature meteo/temporali guidano PM10; L1+L2 gestisce collinearità tra feature meteo correlate |
| **XGBoost** | Performance | Tipicamente il miglior accuracy; gestisce interazioni tra feature naturalmente |

- **Metrica di tuning**: `neg_root_mean_squared_error` — RMSE penalizza quadraticamente
  gli errori grandi, forzando il modello a non sottostimare i picchi di PM10
  (che sono proprio le giornate dove serve l'allerta sanitaria)
- **Metriche di valutazione** (test set): RMSE, MAE, R²

### Classificazione (step_5) — Metrica e modelli

**Scelta della metrica**: i falsi negativi (giorno rosso predetto verde) sono più gravi
dei falsi positivi — persone vulnerabili non avvertite → rischio sanitario.
Tuttavia, tuning su recall pura può essere manipolato (predire tutto come rosso).

→ **F1-macro**: bilancia precision e recall su tutte e 4 le classi equamente,
senza far dominare la classe maggioritaria (verde).

| Modello | Ruolo | Note |
|---------|-------|------|
| **Random Forest** | Baseline robusta | Usare `class_weight='balanced'` per gestire sbilanciamento |
| **KNN** | Instance-based | Interessante per pattern spaziali/temporali; richiede scaling (RobustScaler) |
| **XGBoost** | Performance | Usare `scale_pos_weight` o `sample_weight` per classi sbilanciate |

**Gestione sbilanciamento**: confrontare due approcci:
1. `class_weight='balanced'` / `scale_pos_weight` (più semplice, dentro il modello)
2. SMOTE via `imblearn.pipeline` (oversampling sintetico della minoranza)

Scegliere quello che dà il miglior F1-macro in CV.

---

## Perché Neo4j ha senso

Le stazioni non sono indipendenti: una zona industriale a monte (es. Sesto San Giovanni)
produce inquinamento che si propaga verso le stazioni downwind (es. Milano centro).
Neo4j modella questi legami:

```
(Stazione {nome: 'Sesto SG'})-[:INFLUENZA {distanza_km: 8}]->(Stazione {nome: 'Milano Via Pascal'})
(Zona {tipo: 'industriale'})-[:CONTIENE]->(Stazione)
(Provincia {nome: 'Milano'})-[:HA_STAZIONI]->(Stazione)
```

Il grafo permette query tipo: "quali stazioni a valle di una zona industriale
hanno superato il limite PM10 nelle ultime 24h?"

---

## Perché OpenSearch ha senso

ARPA Lombardia pubblica bollettini testuali giornalieri con commenti sulle condizioni
atmosferiche e previsioni qualità aria. Questi sono ricercabili full-text:
- "nebbia persistente in pianura" → feature categorica per il modello
- "vento forte da nord" → proxy per buona dispersione
- "episodio acuto PM10 per saharan dust" → anomalia da escludere dal training

---

## GCP setup (ricicla da Ex4)

```bash
PROJECT_ID="linear-potion-490413-s9"   # stesso progetto di Malpensa
REGION="europe-west1"
SERVICE_NAME="aria-pulita-collector"
BUCKET="aria-pulita-data-marco"
SCHEDULER_JOB="aria-pulita-job"
SCHEDULE="0 * * * *"               # ogni ora (top of hour)
```

Deploy identico a Ex4:
1. `docker build` + `docker push` su Artifact Registry
2. `gcloud run deploy` con env var BUCKET_NAME
3. `gcloud scheduler jobs create http` per trigger automatico

---

## TODO

- [x] Brainstorming e design
- [x] `step_3/app.py` — collector Flask (ARPA + Open-Meteo)
- [x] `step_3/main.py` — CLI runner
- [x] `step_3/Dockerfile` + `requirements.txt`
- [x] Deploy GCP: bucket `aria-pulita-data-marco` + Cloud Run `aria-pulita-collector` + Cloud Scheduler `aria-pulita-job` (ogni ora)
- [x] Raccogliere dati storici (almeno 1 anno per avere stagionalità) — `step_3/backfill.py`
- [x] `step_1/eda.py` — EDA su dataset raccolto (10 plot: missing values, class distribution, correlazioni, heatmap stazione×ora, time series, boxplot per classe, meteo per classe, profilo orario NO2, giorno settimana, BLH vs PM10)
- [ ] `step_2/` — ingestione MySQL + Neo4j + OpenSearch
- [ ] `step_4/` — modello regressione PM10
- [ ] `step_5/` — classificatore 4 classi + imbalanced handling
