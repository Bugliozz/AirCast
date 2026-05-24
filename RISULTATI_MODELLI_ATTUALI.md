# Risultati Modelli Attuali

Riepilogo delle metriche correnti di regressione e classificazione lette dagli artifact del progetto.

- Regressione: `step_4_regression/artifacts/regression_metrics.json`
- Classificazione: `step_5_classification/artifacts/classification_metrics.json`
- Ibrida: `step_5_classification/artifacts/hybrid_metrics.json`
- Clustering: `step_6_clustering/artifacts/clustering_metrics.json`
- RMT: `step_3_eda/rmt_summary.json`
- Campioni di test: `16_260`
- Dataset: `step_3_eda/daily_dataset_clean.parquet`

## 1. Sintesi Esecutiva

- Run aggiornata il `2026-05-22` con dati estesi fino a `2026-05-07`
- Best model regressione: `xgboost` (R²=0.7081, RMSE=9.048)
- Best strategia per `f1_macro`: `hybrid_xgboost_reg_xgboost_cls` (0.6421)
- **Strategie production-ready (vincoli §8)**: `xgboost_ordinal` e `hybrid_xgboost_reg_xgboost_cls` — entrambe superano `recall_rosso >= 0.65` e `severe_error_rate <= 2.5%`
- `xgboost_ordinal` massimizza il recall critico (0.7201) al costo di f1_macro più basso (0.6333)
- L'ibrido bilancia: `f1_macro=0.6421` (il migliore tra tutte le strategie), `recall_rosso=0.664`, `severe_error_rate=2.24%`
- Clustering: KMeans k=2 è il migliore (sil=0.467); separazione pm10 statisticamente significativa (p=8.4×10⁻⁷)

- RMT: 9 componenti su 41 feature numeriche sopra il bulk Marchenko-Pastur; spiegano ~74.0% della varianza spettrale

## 2. Split del dataset

- **Train**: `2024-01-01` → `2025-08-22`, `38_335` campioni
- **Test**: `2025-08-23` → `2026-05-07`, `16_260` campioni
- Stazioni train/test/intersezione: `67` / `67` / `67`
- Distribuzione classi test: verde=5555, giallo=5445, arancio=3135, rosso=2125

## 3. Regressione

Artifact generato il `2026-05-22T11:42:20Z`.

### 3.1 Metriche globali

| Modello | RMSE | MAE | R² |
|---|---:|---:|---:|
| `elasticnet` | 16.5985 | 7.9241 | 0.0176 |
| `xgboost` | **9.0482** | **6.1189** | **0.7081** |
| `random_forest` | 9.3885 | 6.3934 | 0.6857 |

### 3.2 RMSE per fascia di allerta

| Modello | Verde | Giallo | Arancio | Rosso |
|---|---:|---:|---:|---:|
| `elasticnet` | 6.107 | 6.515 | 12.208 | 41.012 |
| `xgboost` | **5.498** | **6.705** | **8.756** | **17.864** |
| `random_forest` | 5.921 | 7.023 | 9.263 | 18.162 |

### 3.3 Lettura rapida

- `xgboost` è il miglior modello in tutte le fasce; `elasticnet` degrada fortemente sui picchi (RMSE rosso=41.0)
- Rispetto alla run precedente (2026-04-10): R² XGBoost scende da 0.721 a 0.708 — il nuovo periodo di test (2025-09→2026-05) include più campioni rosso proporzionalmente

## 4. Classificazione

Artifact generato il `2026-05-22T12:30:49Z`; valutazione ibrida generata il `2026-05-22T12:48:55Z`.

### 4.1 Metriche globali

| Strategia | F1-macro | Recall rosso | Severe error rate | Over alert rate | Under alert rate |
|---|---:|---:|---:|---:|---:|
| `logistic_regression` | 0.6181 | 0.6551 | 2.72% | 24.07% | 13.48% |
| `random_forest` | 0.6322 | 0.5751 | 2.11% | 20.25% | 15.07% |
| `xgboost` | 0.6399 | 0.6174 | **1.96%** | 21.08% | 13.79% |
| `regression_to_class_xgboost` | 0.6254 | 0.3953 | 1.56% | 11.16% | 22.60% |
| `xgboost_ordinal` (**Fase 3**) | 0.6333 | **0.7201** | 2.39% | 19.92% | 14.81% |
| `hybrid_xgboost_reg_xgboost_cls` (**Fase 2**) | **0.6421** | 0.6640 | 2.24% | 16.41% | 16.89% |

`severe_error_rate` = quota di predizioni con |classe_pred − classe_vera| ≥ 2.

### 4.2 Dettaglio per classe — `xgboost` (best classifier puro)

| Classe | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `verde` | 0.8258 | 0.7323 | 0.7763 | 5555 |
| `giallo` | 0.6223 | 0.6028 | 0.6124 | 5445 |
| `arancio` | 0.4758 | 0.6147 | 0.5364 | 3135 |
| `rosso` | 0.6527 | 0.6174 | 0.6346 | 2125 |

### 4.3 Dettaglio per classe — `xgboost_ordinal` (best recall rosso)

| Classe | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `verde` | 0.8034 | 0.7999 | 0.8017 | 5343 |
| `giallo` | 0.6238 | 0.5706 | 0.5960 | 5021 |
| `arancio` | 0.4887 | 0.4858 | 0.4872 | 3102 |
| `rosso` | 0.5894 | 0.7201 | 0.6482 | 2115 |

### 4.4 Dettaglio per classe — `hybrid` (best f1_macro)

| Classe | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `verde` | 0.7969 | 0.7838 | 0.7903 | 5555 |
| `giallo` | 0.6239 | 0.6702 | 0.6462 | 5445 |
| `arancio` | 0.5431 | 0.4565 | 0.4960 | 3135 |
| `rosso` | 0.6103 | 0.6640 | 0.6360 | 2125 |

### 4.5 Vincoli hard sezione 8

| Strategia | `recall_rosso >= 0.65` | `severe_error_rate <= 2.5%` | Esito |
|---|---:|---:|---|
| `logistic_regression` | OK, 0.6551 | FAIL, 2.72% | **Fail** |
| `random_forest` | FAIL, 0.5751 | OK, 2.11% | **Fail** |
| `xgboost` | FAIL, 0.6174 | OK, 1.96% | **Fail** |
| `regression_to_class_xgboost` | FAIL, 0.3953 | OK, 1.56% | **Fail** |
| `xgboost_ordinal` (**Fase 3**) | **OK, 0.7201** | **OK, 2.39%** | **PASS** |
| `hybrid_xgboost_reg_xgboost_cls` (**Fase 2**) | **OK, 0.6640** | **OK, 2.24%** | **PASS** |

**Due strategie superano entrambi i vincoli**. `xgboost_ordinal` domina per `recall_rosso`; `hybrid` domina per `f1_macro` e `severe_error_rate`.

### 4.6 Calibrazione e parametri ibridi

- Modello selezionato per calibrazione e supporto ibrido: `xgboost`
- Calibrazione: `isotonic`, 5 split temporali
- Artifact: `step_5_classification/artifacts/xgboost_hybrid_calibrated.joblib`

**ECE per classe (10 bin)**:

| Classe | ECE prima | ECE dopo |
|---|---:|---:|
| `verde` | 0.0363 | 0.0447 |
| `giallo` | 0.0433 | 0.0318 |
| `arancio` | 0.0513 | 0.0289 |
| `rosso` | 0.0103 | 0.0083 |

Nota: la classe `verde` mostra un ECE leggermente peggiorato dopo calibrazione; le classi critiche (`arancio`, `rosso`) migliorano nettamente.

**Parametri ibridi** (ottimizzati via OOF su train, `alpha=HYBRID_ALPHA`, `beta=HYBRID_BETA`):
- `delta = 7.0`, `p_threshold = 0.25`
- OOF score: 0.4317 (f1_macro=0.606, recall_rosso=0.689, severe_error_rate=1.92%)
- Override rate sul test: 11.83% (1924 campioni escalati dal regressore al classificatore)

**Tradeoff ibrido vs strategie di riferimento**:
- vs `regression_to_class_xgboost`: recall_rosso +0.269, over_alert_rate +0.052, severe_error_rate +0.007, f1_macro +0.017
- vs `xgboost` calibrato (ibrido): recall_rosso +0.047, over_alert_rate -0.047, severe_error_rate +0.003, f1_macro +0.039

### 4.7 Lettura rapida

- `xgboost_ordinal` (Fase 3) e `hybrid` (Fase 2) sono le uniche due strategie a superare entrambi i vincoli operativi
- `hybrid` ha il miglior f1_macro assoluto (0.6421) e penalizza meno la classe `arancio` rispetto all'ordinale
- `xgboost_ordinal` ha il recall_rosso più alto (0.7201) — preferibile quando il costo di falsi negativi su picchi è dominante
- Per deployment, `hybrid` offre il miglior equilibrio; `xgboost_ordinal` è preferibile in contesti ad alta sensibilità per allerta rossa

## 5. Clustering

Artifact generato il `2026-05-22`. Input: profili stazione (67 stazioni, 8 feature).

### 5.1 Confronto algoritmi

| Algoritmo | k | Silhouette | Calinski-Harabasz | Davies-Bouldin | Interpretabile |
|---|---:|---:|---:|---:|---|
| KMeans | **2** | **0.467** | 33.4 | 1.134 | Sì |
| KMeans | 3 | 0.358 | 39.0 | 0.994 | No (degenerate) |
| Agglomerative (Ward) | 2 | 0.694 | 33.2 | 0.507 | No (degenerate, sizes=[65,2]) |
| DBSCAN | eps=0.75 | 0.407 | — | — | No (55 noise points) |

KMeans k=2 è il best model: unico non-degenerate con silhouette > 0.4.

### 5.2 Descrizione cluster

- **Cluster 0** (56 stazioni): stazioni tipiche, pm10 medio-basso
- **Cluster 1** (11 stazioni): stazioni con pm10 elevato, più giorni critici

### 5.3 Validazione esterna

| Metrica | Valore |
|---|---:|
| Adjusted Rand Index (vs provincia) | 0.038 |
| Homogeneity | 0.119 |
| Completeness | 0.644 |
| V-measure | 0.201 |

- ARI basso: i cluster non corrispondono a province — catturano profili di inquinamento più che geografia amministrativa
- Completeness alta (0.644): le stazioni della stessa provincia tendono a finire nello stesso cluster

### 5.4 Separazione statistica

| Feature | Kruskal-Wallis stat | p-value |
|---|---:|---:|
| `pm10_mean` | 24.26 | 8.4 × 10⁻⁷ |
| `pct_critical_days` | 26.13 | 3.2 × 10⁻⁷ |

Separazione tra i due cluster altamente significativa su entrambe le feature chiave.

## 6. Distribuzione del Test Set

| Classe | Campioni |
|---|---:|
| `verde` | 5555 |
| `giallo` | 5445 |
| `arancio` | 3135 |
| `rosso` | 2125 |

## 7. Note interpretative

- **Regressione**: `xgboost` resta la scelta dominante; ElasticNet inutilizzabile su fasce alte (RMSE rosso=41)
- **Classificazione**: con il dataset esteso a 2026-05-07 compaiono due strategie production-ready invece di una; il vincolo `recall_rosso >= 0.65` è ora soddisfatto sia dall'ordinale che dall'ibrido
- **Clustering**: k=2 è l'unica soluzione non-degenerate interpretabile; la forte significatività statistica conferma che i cluster separano stazioni a diverso regime di inquinamento, non artefatti geografici
- **Confronto run precedente** (2026-04-24): f1_macro XGBoost scende da 0.6419 a 0.6399 sul nuovo test set più lungo; recall_rosso logistic_regression scende da 0.6534 a 0.6551 (stabile); l'ibrido con i nuovi param (delta=7.0, p_threshold=0.25) passa i vincoli hard per la prima volta su questo split

## 8. Diagnostica RMT

Artifact generato il `2026-05-22`. Input: `daily_dataset_clean.parquet`, 54.595 righe.

| Metrica | Valore |
|---|---:|
| Feature numeriche analizzate | 41 |
| `lambda+` Marchenko-Pastur | 1.056 |
| Componenti sopra MP | 9 |
| Varianza spettrale spiegata sopra MP | 74.0% |
| Componenti sopra null empirico p95 | 9 |

Prime letture dagli autovettori:
- Componente 1: stagionalita'/temperatura/fotolisi (`mese_cos`, `temp_*`, `heating_season`, `radiation_mean`, `o3_*`)
- Componente 2: pressione/geografia/persistenza PM10 (`pressure_*`, `lat`, `quota`, `pm10_roll*`, `pm10_lag*`)
- Componente 3: ventilazione e stagnazione (`wind_speed_*`, `wind_speed_roll3`, `stagnation_index`, `precip_sum`)

Output principali: `rmt_summary.json`, `rmt_eigenvalues.csv`, `rmt_component_loadings.csv`, plot `14`-`16` in `step_3_eda/plots/`.
