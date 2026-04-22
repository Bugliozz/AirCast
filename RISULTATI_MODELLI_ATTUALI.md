# Risultati Modelli Attuali

Riepilogo delle metriche correnti di regressione e classificazione lette dagli artifact del progetto.

- Regressione: `step_4_regression/artifacts/regression_metrics.json`
- Classificazione: `step_5_classification/artifacts/classification_metrics.json`
- Campioni di test: `15_544`
- Dataset valutato in regressione: `step_3_eda/daily_dataset_clean.parquet`

## 1. Sintesi Esecutiva

- Best model regressione: `xgboost`
- Best model classificazione: `xgboost`
- La regressione `xgboost` e' nettamente migliore di `elasticnet` e leggermente migliore di `random_forest`
- La classificazione `xgboost` ha il miglior `f1_macro`, ma il margine su `random_forest` e' minimo
- Nella classificazione gli errori gravi restano contenuti, ma il collo di bottiglia principale sembra essere la separazione tra classi adiacenti

## 2. Regressione

Artifact generato il `2026-04-10T11:29:56Z`.

### 2.1 Metriche globali

| Modello | RMSE | MAE | R2 |
|---|---:|---:|---:|
| `elasticnet` | 13.7807 | 9.9806 | 0.3505 |
| `xgboost` | **9.0326** | **6.0817** | **0.7210** |
| `random_forest` | 9.4739 | 6.4516 | 0.6930 |

### 2.2 RMSE per fascia di allerta

| Modello | Verde | Giallo | Arancio | Rosso |
|---|---:|---:|---:|---:|
| `elasticnet` | 7.1032 | 6.1053 | 14.4677 | 29.5647 |
| `xgboost` | **5.6537** | **6.7883** | **8.6185** | **17.3629** |
| `random_forest` | 6.0840 | 7.0035 | 9.1543 | 18.1051 |

### 2.3 Lettura rapida

- `xgboost` e' il miglior modello anche nelle fasce alte di PM10
- La fascia `rosso` resta la piu' difficile per tutti i modelli
- `elasticnet` degrada molto sui picchi: `RMSE rosso = 29.5647`

## 3. Classificazione

Artifact generato il `2026-04-11T15:16:50Z`.

### 3.1 Metriche globali

| Modello | F1-macro | Severe error rate |
|---|---:|---:|
| `logistic_regression` | 0.6286 | 2.86% |
| `random_forest` | 0.6392 | **2.11%** |
| `xgboost` | **0.6413** | 2.37% |

`severe_error_rate` = quota di predizioni con distanza tra classe vera e predetta maggiore o uguale a 2.

### 3.2 Dettaglio per classe del best model (`xgboost`)

| Classe | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `verde` | 0.8539 | 0.7269 | 0.7853 | 5339 |
| `giallo` | 0.6108 | 0.5741 | 0.5919 | 4992 |
| `arancio` | 0.4679 | 0.6383 | 0.5400 | 3099 |
| `rosso` | 0.6534 | 0.6429 | 0.6481 | 2114 |

### 3.3 Lettura rapida

- `xgboost` e' il migliore per `f1_macro`, ma il vantaggio su `random_forest` e' molto piccolo
- `random_forest` ha il `severe_error_rate` piu' basso tra i classificatori valutati
- Le classi intermedie (`giallo`, `arancio`) restano le piu' difficili da separare bene

## 4. Distribuzione del Test Set

La distribuzione delle classi nel test set e' coerente tra regressione e classificazione:

| Classe | Campioni |
|---|---:|
| `verde` | 5339 |
| `giallo` | 4992 |
| `arancio` | 3099 |
| `rosso` | 2114 |

## 5. Nota Interpretativa

- Regressione: oggi la scelta piu' forte e' `xgboost`, soprattutto per stabilita' globale e gestione delle fasce alte
- Classificazione: `xgboost` e' primo per `f1_macro`, ma non in modo schiacciante
- Operativamente conviene continuare a monitorare non solo `f1_macro`, ma anche `recall` della classe `rosso` e `severe_error_rate`
