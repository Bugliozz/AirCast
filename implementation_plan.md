# Piano di Implementazione — Step 4 (Regressione) & Step 5 (Classificazione)

> Documento operativo. Ogni task ha una checkbox per tracciare l'avanzamento.
> Basato sull'EDA completata e sul dataset `daily_dataset_clean.parquet`.
>
> **Legenda difficoltà / modello LLM consigliato:**
> 🟢 Facile → Haiku | 🟠 Medio → Sonnet | 🔴 Difficile/Lungo → Opus

---

## 0. Prerequisiti e Stato Attuale

**Dataset disponibile:** `step_3_eda/daily_dataset_clean.parquet`
- Granularita: **giornaliera** (1 riga per stazione per giorno)
- Target regressione: `pm10` (ug/m3)
- Target classificazione: `classe_allerta` (verde / giallo / arancio / rosso)
- ~65-75 colonne, ~10k-30k righe (stazioni x giorni validi)
- Periodo: marzo 2025 – marzo 2026 (~13 mesi)

**Feature gia implementate in build_dataset.py:**
- Lag: `pm10_lag1`, `pm10_lag2`, lag meteo (pressure, wind, BLH, temp)
- Rolling: `pressure_roll3`, `wind_speed_roll3`
- Stagnation flag booleano (soglie fisse: P>1013.25, wind<1.5, BLH<500)
- `is_weekend`, `fog_hours`, `mese`, `stagione`, `giorno_settimana`
- Spaziali: `dist_industrial_km`, `n_industrial_zones_15km`

---

## 1. Feature Engineering Aggiuntivo

> Modifiche a `step_3_eda/build_dataset.py` e/o nuovo modulo condiviso.

- [x] **1.1 — Encoding ciclico sin/cos per mese e giorno della settimana** 🟢
  - `mese_sin = sin(2 * pi * mese / 12)`, `mese_cos = cos(2 * pi * mese / 12)`
  - `dow_sin = sin(2 * pi * giorno_settimana / 7)`, `dow_cos = cos(2 * pi * giorno_settimana / 7)`
  - Motivo: il modello deve sapere che Dicembre (12) e Gennaio (1) sono vicini

- [x] **1.2 — Indice di stagnazione continuo** 🟢
  - `stagnation_index = 1 / (wind_speed_mean * blh_min * (1 + precip_sum))`
  - Clippare a un massimo ragionevole per evitare divisioni per zero (aggiungere epsilon)
  - Mantenere anche il `stagnation_flag` booleano per interpretabilita

- [x] **1.3 — Rolling media 7 giorni del PM10** 🟢
  - `pm10_roll7 = pm10.rolling(7, min_periods=1).mean()` per stazione
  - Cattura il trend di medio periodo (accumulo persistente)

- [x] **1.4 — Aggiornare `requirements.txt`** 🟢
  - Aggiungere: `scikit-learn >= 1.4`, `xgboost >= 2.0`, `joblib`
  - Verificare compatibilita con le dipendenze esistenti

---

## 2. Infrastruttura Condivisa

> File singolo `shared/utils.py` riutilizzabile da Step 4 e Step 5.

- [x] **2.1 — Creazione `shared/utils.py`** 🟢
  - Scaffold del file con import e docstring

- [x] **2.2 — Split holdout temporale** 🟠
  - Funzione `temporal_train_test_split()`
  - Ordinare per `data_giorno`, prendere i primi 70% dei giorni come train, il restante 30% come test
  - Lo split e per giorno (NON per riga): tutte le stazioni dello stesso giorno vanno nello stesso set
  - Restituire `(X_train, y_train, X_test, y_test)` con le date di cutoff loggate
  - **CRITICO**: niente shuffle, niente split casuale

- [x] **2.3 — Imputation post-split** 🟢
  - Usare `impute_missing()` gia presente in `build_dataset.py`
  - Flusso: `clean_parquet` -> split temporale -> `impute_missing(train)` -> `impute_missing(test, medians=train_medians)`

- [x] **2.4 — ColumnTransformer per preprocessing** 🟠
  - Feature numeriche per modelli lineari: `StandardScaler`
  - Feature numeriche per tree-based: `passthrough` (non necessitano scaling)
  - Feature categoriche (`stagione`, `provincia`): `OrdinalEncoder` o `OneHotEncoder`
  - Drop colonne identificative: `idstazione`, `nomestazione`, `comune`, `data_giorno`

- [ ] **2.5 — TimeSeriesSplit wrapper** 🔴
  - Funzione che riceve il dataframe di training e restituisce indici per `TimeSeriesSplit(n_splits=5)`
  - Lo split deve essere su base giornaliera (raggruppare per `data_giorno`)
  - **ATTENZIONE**: `TimeSeriesSplit` sklearn lavora su indici row, NON su gruppi. Con stazioni multiple per giorno, stazioni dello stesso giorno potrebbero finire in train e test contemporaneamente (leakage). Costruire gli indici manualmente: ricavare i giorni unici ordinati, splittarli in 5 fold temporali, poi espandere agli indici riga corrispondenti.
  - Verificare che ogni fold contenga almeno qualche campione per classe (log warning se mancano)

---

## 3. Step 4 — Regressione

> Modulo: `step_4_regression/`

```
step_4_regression/
├── __init__.py
├── config.py       # Iperparametri, search space
├── train.py        # Pipeline di training
├── evaluate.py     # Valutazione finale su test set
└── artifacts/      # Modelli salvati, metriche, plot
```

### 3.1 — Preparazione

- [ ] **3.1.1 — Scaffold della directory e dei file** 🟢

- [ ] **3.1.2 — Config iperparametri** 🟢
  - Usare `GridSearchCV` per ElasticNet (spazio piccolo: 4×4=16 combinazioni), `RandomizedSearchCV(n_iter=30)` per XGBoost e RandomForest.
  - Motivazione: la griglia completa di XGBoost fa 3×3×3×3=81 combinazioni × 5 fold = 405 fit; con `n_iter=30` si ottiene il 90% della qualità con il 37% del costo.
  - Search space:
    - ElasticNet: `alpha` [0.01, 0.1, 1, 10], `l1_ratio` [0.1, 0.5, 0.7, 0.9]
    - XGBoost: `n_estimators` [100, 300, 500], `max_depth` [3, 5, 7], `learning_rate` [0.01, 0.05, 0.1], `subsample` [0.7, 0.8, 1.0]
    - RandomForest: `n_estimators` [100, 300, 500], `max_depth` [10, 20, None], `min_samples_leaf` [2, 5, 10]

### 3.2 — Trasformazione del Target

- [ ] **3.2.1 — Log-transform del target** 🟢
  - `y_train_log = np.log1p(y_train)` — trasformazione
  - `y_pred = np.expm1(y_pred_log)` — inverse transform per le predizioni
  - Usare `TransformedTargetRegressor` di sklearn per incapsularlo nella pipeline
  - Motivo: stabilizza la varianza, aiuta ElasticNet sulla coda lunga

### 3.3 — Training dei Modelli Base

- [ ] **3.3.1 — ElasticNet (con StandardScaler + log target)** 🟠
  - Pipeline: `StandardScaler` -> `ElasticNet`
  - Wrappato in `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`
  - CV: `TimeSeriesSplit(5)`, scoring: `neg_root_mean_squared_error`
  - Ruolo: intercettare trend lineari, capacita di estrapolazione sui picchi

- [ ] **3.3.2 — XGBoost Regressor** 🟠
  - Pipeline: nessun scaling necessario
  - CV: stessa `TimeSeriesSplit(5)`, scoring: `neg_root_mean_squared_error`
  - Ruolo: catturare interazioni non lineari e soglie complesse

- [ ] **3.3.3 — Random Forest Regressor (baseline)** 🟠
  - Pipeline: nessun scaling necessario
  - CV: stessa `TimeSeriesSplit(5)`, scoring: `neg_root_mean_squared_error`
  - Ruolo: baseline robusta, meno sensibile a overfitting

### 3.4 — Stacking Ensemble (opzionale, miglioramento futuro)

> Da implementare solo se i 3 modelli base hanno performance insufficienti
> o se c'e tempo per sperimentare dopo aver completato Step 4 e Step 5.

- [ ] **3.4.1 — StackingRegressor** 🔴
  - Base estimators: ElasticNet (best), XGBoost (best), Random Forest (best)
  - Meta-learner: `RidgeCV` (semplice, evita overfitting dello stacking)
  - `cv=TimeSeriesSplit(5)` per generare le meta-features

### 3.5 — Valutazione Finale (Test Set)

- [ ] **3.5.1 — Metriche su test holdout** 🟠
  - RMSE (metrica primaria)
  - MAE (robustezza agli outlier)
  - R2 (varianza spiegata)
  - Calcolare anche RMSE separato per fasce: verde, giallo, arancio, rosso
  - Tabella riassuntiva di tutti i modelli

- [ ] **3.5.2 — Plot diagnostici** 🟠
  - Scatter: y_pred vs y_true (con linea identita)
  - Residui vs y_pred (verificare omoschedasticita)
  - Residui vs tempo (verificare assenza di trend)
  - Distribuzione dei residui (idealmente normale, centrata su 0)

- [ ] **3.5.3 — Feature importance** 🟠
  - Permutation importance sul **best model** (su test set), top 15 feature
  - Per gli altri modelli: usare `.feature_importances_` nativa (RF, XGBoost) come sanity check

- [ ] **3.5.4 — Salvataggio artifacts** 🟢
  - Best model pipeline completa: `joblib.dump()` in `artifacts/`
  - Metriche in JSON: `artifacts/regression_metrics.json`
  - Plot in PNG: `artifacts/plots/`

---

## 4. Step 5 — Classificazione

> Modulo: `step_5_classification/`

```
step_5_classification/
├── __init__.py
├── config.py       # Iperparametri, class weights
├── train.py        # Pipeline di training
├── evaluate.py     # Valutazione finale su test set
└── artifacts/      # Modelli salvati, metriche, plot
```

### 4.1 — Preparazione

- [ ] **4.1.1 — Scaffold della directory e dei file** 🟢

- [ ] **4.1.2 — Encoding del target** 🟢
  - `classe_allerta` -> label encoding ordinale: verde=0, giallo=1, arancio=2, rosso=3
  - **NON usare `OrdinalEncoder` sklearn**: di default non garantisce l'ordine semantico corretto. Usare mappatura esplicita: `LABEL_MAP = {'verde': 0, 'giallo': 1, 'arancio': 2, 'rosso': 3}` e applicarla con `.map()`.
  - Verificare la distribuzione delle classi nel train e nel test set (log delle percentuali)

- [ ] **4.1.3 — Config iperparametri** 🟢
  - Usare `GridSearchCV` per Logistic Regression (spazio piccolo), `RandomizedSearchCV(n_iter=30)` per RandomForest e XGBoost (stesso ragionamento del task 3.1.2).
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight='balanced'`
  - XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight` o sample weights
  - Logistic Regression: `C` [0.01, 0.1, 1, 10], `penalty='elasticnet'`, `class_weight='balanced'`

### 4.2 — Gestione Sbilanciamento

- [ ] **4.2.1 — Strategia: `class_weight='balanced'` (primaria)** 🟢
  - Applicare nativamente a tutti i modelli che lo supportano
  - NON usare SMOTE: genera giornate meteo sintetiche fisicamente irrealistiche
  - Motivo: con 4 classi meteo-dipendenti, l'oversampling sintetico crea combinazioni
    (es. alta temperatura + alta pressione + forte precipitazione) impossibili in natura

### 4.3 — Training dei Modelli

- [ ] **4.3.1 — Random Forest Classifier** 🟠
  - `class_weight='balanced'`
  - CV: `TimeSeriesSplit(5)`, scoring: `f1_macro`

- [ ] **4.3.2 — XGBoost Classifier** 🟠
  - `scale_pos_weight` calcolato per classe, oppure sample weights
  - CV: `TimeSeriesSplit(5)`, scoring: `f1_macro`

- [ ] **4.3.3 — Logistic Regression (ElasticNet penalty)** 🟠
  - Pipeline: `StandardScaler` -> `LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced')`
  - CV: `TimeSeriesSplit(5)`, scoring: `f1_macro`
  - Ruolo: contributo lineare complementare ai tree-based (come ElasticNet per la regressione)

### 4.4 — Calibrazione delle Probabilita (opzionale, miglioramento futuro)

> Da implementare solo se c'e tempo dopo aver completato Step 4 e Step 5.

- [ ] **4.4.1 — CalibratedClassifierCV** 🟠
  - Applicare al miglior modello dopo il tuning
  - Metodo: `'sigmoid'` (piu stabile di `'isotonic'` con 4 classi su dataset non enorme)
  - Motivo: per un sistema di allerta pubblica, "73% probabilita di Rosso"
    e molto piu utile di un semplice label

### 4.5 — Valutazione Finale (Test Set)

- [ ] **4.5.1 — Classification Report** 🟢
  - Precision, Recall, F1 per ogni classe
  - F1-macro (metrica primaria)
  - F1-weighted (per contesto)

- [ ] **4.5.2 — Matrice di Confusione** 🟢
  - Heatmap annotata con conteggi
  - **Focus critico**: Falsi Negativi gravi = rosso classificato come verde/giallo
  - Obiettivo salute pubblica: minimizzare i FN sulle classi pericolose

- [ ] **4.5.3 — Analisi degli errori gravi** 🟢
  - Calcolare la percentuale di errori con distanza >= 2 (es. rosso->verde, rosso->giallo)
  - Riportare come singola metrica di sicurezza: `severe_error_rate`

- [ ] **4.5.4 — Feature importance** 🟠
  - Permutation importance sul **best model** (su test set), top 15 feature
  - Per gli altri modelli: usare `.feature_importances_` nativa come sanity check

- [ ] **4.5.5 — Salvataggio artifacts** 🟢
  - Best model pipeline + calibratore: `joblib.dump()` in `artifacts/`
  - Metriche in JSON: `artifacts/classification_metrics.json`
  - Confusion matrix + report in `artifacts/plots/`

---

## 5. Cross-Analisi e Confronto

- [ ] **5.1 — Coerenza regressione-classificazione** 🟠
  - Prendere le predizioni del regressore, discretizzarle con le soglie PM10, confrontare con il classificatore
  - Se il regressore discretizzato batte il classificatore, potrebbe non servire un classificatore separato

- [ ] **5.2 — Analisi residui per stazione e stagione** 🟠
  - Il modello fallisce su certe stazioni? Certe stagioni?
  - Plot: barplot RMSE per stazione e per mese

- [ ] **5.3 — Spot check leave-one-station-out** 🟠
  - Sul test set, calcolare RMSE/F1 separatamente per ogni stazione.
  - Verificare che le stazioni con performance molto peggiore della media non siano stazioni usate poco nel training (modello che ha memorizzato pattern stazione-specifici invece di generalizzare).
  - Non richiede re-training: è un'analisi post-hoc delle predizioni già generate in 3.5 e 4.5.


---

## 6. Note Architetturali

### Anti-Leakage Checklist

- [ ] Split temporale puro (niente shuffle)
- [ ] Imputation mediana calcolata solo su train set
- [ ] Nessuna feature che usa informazione futura
- [ ] TimeSeriesSplit per la CV (niente KFold/StratifiedKFold)
- [ ] Lag features calcolate pre-split (usano solo passato, safe)
- [ ] Soglie stagnazione fisse (non data-dependent)
- [ ] Feature engineering su tutto il dataset solo se deterministico (sin/cos, is_weekend)
- [ ] StandardScaler fit solo su train, transform su test

### Modelli Esclusi e Motivazione

| Modello | Motivo esclusione |
|---|---|
| LightGBM | Ridondante con XGBoost (entrambi gradient boosting), differenze trascurabili su ~10-30k righe |
| KNN | Maledizione della dimensionalita, non gestisce bene sbilanciamento su 4 classi |
| SMOTE | Genera campioni sintetici meteo fisicamente impossibili |
| StratifiedKFold | Mescola il tempo, causa data leakage |
| Neural Networks | Dataset troppo piccolo (~10-30k righe), rischio overfitting senza beneficio |
| NO2 disaggregato orario | Richiede nuova query SQL + join, beneficio incerto rispetto a `is_weekend` come proxy traffico |
| Binning distanza industriale | I tree-based trovano da soli le soglie ottimali sulla feature continua; il binning butta via informazione |
| Learning curves | Computazionalmente costose, risultato non azionabile (il periodo dati è fisso) |

### Dipendenze Aggiuntive

```
scikit-learn >= 1.4
xgboost >= 2.0
joblib
```

---

## Ordine di Esecuzione

```
1. Feature Engineering (Sezione 1)
   |
2. Infrastruttura Condivisa (Sezione 2)
   |
   ├── 3. Step 4: Regressione (Sezione 3)
   |       3.1 Scaffold -> 3.2 Log target -> 3.3 Modelli base (3 modelli)
   |       -> 3.5 Valutazione
   |
   └── 4. Step 5: Classificazione (Sezione 4)
           4.1 Scaffold -> 4.2 Sbilanciamento -> 4.3 Modelli (3 modelli)
           -> 4.5 Valutazione
   |
5. Cross-Analisi (Sezione 5)
   |
6. (Opzionale) Stacking Ensemble (Sezione 3.4) / Calibrazione (Sezione 4.4)
```
