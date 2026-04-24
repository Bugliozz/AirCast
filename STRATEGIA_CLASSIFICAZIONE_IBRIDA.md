# Step 5.9 — Strategia di classificazione ibrida (regressore + classificatore a supporto)

Piano di implementazione operativo per la decisione finale di allerta: il regressore (Step 4) e' il backbone, il classificatore (Step 5) interviene come **correttore in zona-soglia** con parametri `(δ, p_threshold)` tunati su validation.

Scopo: formalizzare in modo riproducibile i punti `[ ] Esperimento aggiuntivo` dello Step 5.7 in [`implementation_steps.md`](implementation_steps.md).

---

## 0. Motivazione e tradeoff

- in inferenza il sistema decide un **livello di allerta** (verde/giallo/arancio/rosso) a partire da `pm10_hat` e dalle soglie ARPA `[20, 35, 50]`
- la regressione conserva l'ordine naturale delle classi ed e' piu' stabile del classifier multiclasse vicino alle soglie
- gli errori di classificazione sono soprattutto tra classi adiacenti (vedi Step 5.7): l'informazione utile per "spostare" una decisione tra due classi adiacenti e' la probabilita' calibrata del classificatore **solo quando `pm10_hat` e' vicino a una soglia**
- fuori dalla zona-soglia la classe indotta dal regressore e' gia' sufficiente: sovrascriverla con l'`argmax` di un classificatore e' rumore

Conseguenza: strategia ibrida minimale, non un ensemble complesso. La regola deve restare spiegabile.

---

## 0.1 Fasi e gate di uscita

Il piano e' organizzato in 3 fasi. Ogni fase ha un **gate** che decide se valga la pena passare alla successiva. Se il gate non e' soddisfatto, ci si ferma e si documenta il risultato. Questo evita di costruire infrastruttura (calibrazione, grid `δ×p`, ordinal) prima di sapere se serve davvero.

| Fase | Contenuto | Gate per passare alla fase successiva |
|---|---|---|
| **1** | §1 guard-rail + §5 metriche uniformi + §6.1 baseline `regressione→soglie` con il best regressor attuale | la baseline `regression→soglie` va misurata a vincoli di §8. Se li rispetta gia' e batte il classifier attuale, diventa la baseline operativa; le fasi successive devono battere questo punto di partenza |
| **2** | §4 calibrazione del classifier + §6.2–6.4 strategia ibrida + tuning **out-of-fold** `(δ, p)` | hybrid deve battere **sia** `regression→soglie` puro **sia** il classifier puro su `f1_macro` a vincoli di §8 rispettati. Se nessuna strategia rispetta §8, non si dichiara un modello production-ready: si salva solo il miglior candidato sperimentale e si documenta il vincolo non soddisfatto |
| **3** | §9 ordinal side benchmark | opzionale; valore informativo residuo se `severe_error_rate` e' ancora sopra target |

Le sezioni §1, §5, §7, §8, §10, §11, §12 sono trasversali (si estendono / aggiornano fase dopo fase) e non hanno tag di fase.

---

## 1. Guard-rail anti-leakage

Obbligatori per garantire che i confronti tra strategie siano validi. Un confronto su split diversi non dice nulla.

- [ ] Verifica esplicita in `step_5_classification/train.py` che `temporal_train_test_split` produce `max(data_giorno_train) < min(data_giorno_test)` — assert hard, non solo commento
- [ ] Log in `classification_metrics.json` di `train_date_range`, `test_date_range`, `n_stations_train`, `n_stations_test`, `n_stations_intersection`
- [ ] Tutti i benchmark girano con lo **stesso identico split** (stessi indici) usato per i modelli gia' salvati (`xgboost_best.joblib`, `random_forest_best.joblib`, `logistic_regression_best.joblib`) — pena confronti non validi
- [ ] `make_temporal_cv_splits` con `n_splits=5` identico a quello usato per gli altri modelli
- [ ] `random_state=RANDOM_STATE` fissato in **tutti** i `RandomizedSearchCV` — senza questo il confronto a `n_iter` fissato dipende dall'ordine di sampling degli iperparametri e diventa non riproducibile

Acceptance: se uno dei check fallisce (assert sulle date, intersezione stazioni incoerente, CV diverso, random_state mancante), lo script si ferma.

---

## 2. Setup modulo (`step_5_classification/`)  `[Fase 1]`

### 2.1 `config.py` — nuove costanti  `[Fase 1–2]`

- [ ] `ALERT_THRESHOLDS = [20.0, 35.0, 50.0]` (gia' implicito nella pipeline `regressione → soglie`, esplicitalo qui)
- [ ] `HYBRID_DELTA_GRID = [3.0, 5.0, 7.0]` — larghezza zona-soglia in μg/m³ (Fase 2, vedi §6)
- [ ] `HYBRID_PROB_GRID = [0.25, 0.35, 0.45]` — soglia probabilita' per regola prudenziale (Fase 2, vedi §6)
- [ ] `HYBRID_ALPHA = 1.0`, `HYBRID_BETA = 0.5` — coefficienti del metric obiettivo composto (§6.4)

---

## 3. Classificatore di supporto — candidati ammessi

Il classificatore usato nella strategia ibrida deve:

- essere **calibrato** (vedi §4): la regola prudenziale dipende da `predict_proba` affidabili
- essere addestrato sullo **stesso split temporale** del regressore (vedi §1)
- avere `f1_macro` competitivo tra i classificatori gia' salvati (baseline Step 5)

Candidati di default: i modelli gia' prodotti in Step 5 (`xgboost`, `random_forest`, `logistic_regression`). La strategia ibrida **non** introduce nuovi classificatori in questa sede: il suo valore aggiunto e' nella regola di decisione (§6), non nell'esplorazione di altre famiglie.

---

## 4. Calibrazione del classificatore (obbligatoria per la strategia ibrida)  `[Fase 2]`

La regola prudenziale di §6.3 dipende da probabilita' calibrate. Replica il pattern `CalibratedClassifierCV` gia' in uso (Step 5.5, `method="isotonic"`). **Solo il classifier scelto per l'ibrido entra in questa fase** — non calibrare modelli che non verranno mai selezionati.

Nota anti-leakage: la calibrazione finale per il modello candidato si puo' fare su tutto `X_train`, ma il tuning di `(δ, p_threshold)` in §6.4 deve usare probabilita' **out-of-fold**. Ogni probabilita' usata per scegliere gli iperparametri deve provenire da un modello e da un calibratore che non hanno visto quel campione in fit.

- [ ] Calibra il classifier candidato con `CalibratedClassifierCV(method="isotonic", cv=tscv)`
- [ ] Se il secondo migliore e' a distanza `f1_macro <= 0.005`: calibra anche lui (tie-break da risolvere con l'intera tabella di §7)
- [ ] Reliability diagram one-vs-rest per le 4 classi (stesso stile di `best_model_calibrated`) salvato in `artifacts/plots/`
- [ ] Metrica di calibrazione: `expected_calibration_error` (ECE) per classe, salvata nel JSON delle metriche

---

## 5. Metriche da salvare (uguali per tutti i modelli, altrimenti non sono comparabili)

Aggiorna `evaluate.py` per produrre in `classification_metrics.json` e `regression_metrics.json` i seguenti campi **per ogni strategia** (classifier puro, `regression→soglie`, ibrido):

### 5.1 Regressione

- [ ] `rmse`, `mae`, `r2` globali
- [ ] `rmse_per_fascia`, `mae_per_fascia` (verde/giallo/arancio/rosso) — stessa definizione gia' in uso
- [ ] `rmse_near_thresholds`: RMSE calcolato solo sui campioni con `y_true` in `[t - δ, t + δ]` per ogni `t ∈ ALERT_THRESHOLDS` e `δ = 5.0` (default) — informazione cruciale per capire dove l'ibrido puo' aiutare

### 5.2 Classificazione

- [ ] `f1_macro`, `accuracy`
- [ ] `precision`, `recall`, `f1` per classe
- [ ] `severe_error_rate` = `mean(|y_pred - y_true| >= 2)` (gia' definito, mantieni)
- [ ] `recall_rosso`, `precision_rosso` esplicitati in campo dedicato (leggibilita')
- [ ] `over_alert_rate` = `mean(y_pred > y_true)` e `under_alert_rate` = `mean(y_pred < y_true)` — necessari per leggere il costo operativo dei falsi allarmi e delle mancate allerte
- [ ] `confusion_matrix` (4x4)
- [ ] `ece_per_classe` (solo per modelli calibrati)
- [ ] per la sola strategia ibrida: `hybrid_override_rate`, `hybrid_escalation_rate`, `n_overrides`, `n_escalations` rispetto a `class_reg`

### 5.3 Meta

- [ ] `train_date_range`, `test_date_range`, `n_test_samples`, `n_stations_*` (vedi §1)
- [ ] `strategy_name` per documentare cosa ha prodotto la riga di metriche (`classifier_xgboost_calibrated`, `regression_to_class_xgboost`, `hybrid_xgb_reg_xgb_cls`, ...)

---

## 6. Strategia ibrida `regressione → soglie + correzione classificatore`

### 6.1 Baseline strutturale — `regressione → soglie` pura  `[Fase 1]`

- [ ] Implementa `shared/regression_to_class.py`: `pm10 → classe` usando `ALERT_THRESHOLDS = [20, 35, 50]`
- [ ] **Fase 1 (obbligatorio)**: valuta questa pipeline con il best regressor di Step 4 (attualmente `xgboost`, vedi `RISULTATI_MODELLI_ATTUALI.md`) e salva le metriche di §5.2 con chiave `regression_to_class_xgboost`
- [ ] Questa baseline e' la pietra di paragone per Fase 2: l'ibrido deve batterla, altrimenti non vale la complessita' aggiuntiva

### 6.2 Strategia ibrida con zona-soglia e regola prudenziale  `[Fase 2]`

Definizione esplicita della decisione finale. Per ogni sample:

1. calcola `pm10_hat` dal regressore
2. calcola `class_reg = soglie(pm10_hat)`
3. calcola `proba_cls = classifier.predict_proba(X)` (calibrato)
4. calcola `class_cls = argmax(proba_cls)`
5. sia `d = min_t |pm10_hat - t|` per `t ∈ ALERT_THRESHOLDS`
6. **fuori zona-soglia** (`d > δ`): `class_final = class_reg`
7. **in zona-soglia** (`d <= δ`):
   - se `class_cls == class_reg`: `class_final = class_reg`
   - altrimenti (discordanza): applica regola prudenziale (§6.3)

### 6.3 Regola prudenziale — chiusa e verificabile  `[Fase 2]`

Quando il regressore e il classificatore sono discordi **in zona-soglia**:

- sia `class_severe = max(class_reg, class_cls)` (quella piu' grave tra le due)
- sia `p_severe = proba_cls[class_severe]`
- se `p_severe >= p_threshold`: `class_final = class_severe` (bias prudenziale)
- altrimenti: `class_final = class_reg` (default conservativo sul regressore, che e' piu' stabile)

Il bias prudenziale si attiva **solo** in zona-soglia + discordanza + alta confidenza sulla classe grave. Non e' un override globale.

Nota: questa regola fa solo escalation rispetto alla scelta meno grave tra regressore e classificatore; non fa de-escalation. Per questo in §5.2 e §7 va sempre monitorato `over_alert_rate`, altrimenti un aumento di `recall_rosso` potrebbe nascondere un costo operativo eccessivo in falsi allarmi.

### 6.4 Tuning di `δ` e `p_threshold`  `[Fase 2]`

- [ ] Su `X_train` (mai su test): grid search su `HYBRID_DELTA_GRID × HYBRID_PROB_GRID` usando predizioni **out-of-fold** costruite con `make_temporal_cv_splits`
- [ ] Per ogni fold temporale esterno:
  - fitta un clone del regressore solo sul train-fold e predici `pm10_hat` sul val-fold
  - fitta un clone del classifier solo sul train-fold
  - calibra il clone del classifier usando solo dati del train-fold (inner temporal CV, sempre costruita sul train-fold; se i giorni non bastano, fallisci esplicitamente invece di usare il validation fold)
  - predici `predict_proba` sul val-fold
- [ ] Concatena le predizioni out-of-fold di tutti i val-fold e usa **solo queste** per scegliere `(δ*, p*)`; non usare mai predizioni in-sample di modelli fittati su tutto `X_train`
- [ ] Salva la cache OOF in `artifacts/hybrid_oof_predictions.parquet` con: indice originale, `y_true`, `pm10_hat_oof`, `class_reg_oof`, `proba_cls_oof_*`, `fold`
- [ ] Metric obiettivo composta: `score = f1_macro - α * severe_error_rate - β * (1 - recall_rosso)` con `α = 1.0`, `β = 0.5` (esplicitati in `config.py` come `HYBRID_ALPHA`, `HYBRID_BETA`)
- [ ] Salva `(δ*, p*)` in `artifacts/hybrid_params.json` insieme a curva di `score` vs `(δ, p)`
- [ ] Se il grid e' piatto (differenze di `score` < 0.005 tra tutte le combinazioni): log esplicito — significa che la zona-soglia non sta portando informazione utile e l'ibrido degenera nel regression→soglie puro

### 6.5 Valutazione finale della strategia ibrida  `[Fase 2]`

- [ ] Dopo aver scelto `(δ*, p*)`, usa i modelli finali fittati su tutto `X_train`: best regressor di Step 4 + classifier candidato calibrato su tutto `X_train`
- [ ] Su test set (ignoto al tuning): calcola tutte le metriche di §5.2 per la combinazione `(regressore, classificatore_calibrato, δ*, p*)` selezionata
- [ ] La cache OOF di §6.4 serve **solo** per tuning; non sostituisce la valutazione finale su test
- [ ] Salva in `artifacts/hybrid_metrics.json`
- [ ] Confronta contro: `regression_to_class_xgboost` puro, classifier calibrato puro, baseline classifier non calibrato attuale
- [ ] Documenta esplicitamente il tradeoff `recall_rosso` vs `over_alert_rate`: l'ibrido e' utile solo se recupera allerte rosse senza produrre un aumento operativo eccessivo di falsi allarmi

---

## 7. Protocollo di confronto finale

**Tabella — Classificazione** in `RISULTATI_MODELLI_ATTUALI.md`. La tabella si popola incrementalmente: le righe vengono aggiunte solo man mano che le rispettive fasi vengono eseguite.

| Strategia | Fase | f1_macro | severe_error_rate | recall_rosso | precision_rosso | over_alert_rate | RMSE_rosso |
|---|---|---|---|---|---|---|---|
| `logistic_regression` (stato attuale) | — | ... | ... | ... | ... | ... | — |
| `random_forest` classifier (stato attuale) | — | ... | ... | ... | ... | ... | — |
| `xgboost` classifier (stato attuale) | — | ... | ... | ... | ... | ... | — |
| `xgboost` classifier calibrato | 2 | ... | ... | ... | ... | ... | — |
| `xgboost` regression→soglie | 1 | ... | ... | ... | ... | ... | ... |
| **Ibrido** (best_reg, best_cls_calib, δ\*, p\*) | 2 | ... | ... | ... | ... | ... | ... |
| `xgboost_ordinal` | 3 | ... | ... | ... | ... | ... | — |

- [ ] Script `scripts/compare_classification_strategies.py` genera la tabella automaticamente leggendo i JSON degli artifact
- [ ] Aggiorna `RISULTATI_MODELLI_ATTUALI.md` con la tabella e un commento di 5-8 righe sulla lettura

---

## 8. Criterio di selezione production-ready

Regola di scelta del modello finale — decisa **prima** di vedere i numeri, per evitare cherry-picking:

1. **Vincolo hard**: `severe_error_rate <= 0.025` (oggi `random_forest` ha `0.0211`, `xgboost` `0.0237`). Qualsiasi strategia che supera 2.5% e' scartata.
2. **Vincolo hard**: `recall_rosso >= 0.65` (oggi `xgboost` ha `0.6429`). Sotto questa soglia il sistema non e' un'allerta utile.
3. Tra le strategie che rispettano entrambi i vincoli: sceglie quella con `f1_macro` massimo.
4. A parita' di `f1_macro` (± 0.005): preferisci la strategia piu' semplice (classifier puro > `regression→soglie` > ibrido).
5. Se nessuna strategia rispetta entrambi i vincoli hard: `final_model_selection.json` deve riportare `production_ready=false`, nessuna strategia viene promossa a modello finale operativo, e si salva separatamente `best_research_candidate` con la motivazione del vincolo fallito. In questo caso `api/services/predictor.py` non viene aggiornato automaticamente.

- [ ] Criterio documentato in `technical_doc.md` (sezione "Selezione del modello di allerta")
- [ ] Risultato della selezione scritto in `artifacts/final_model_selection.json` con: `production_ready`, strategia scelta se presente, metriche, `best_research_candidate` se nessuna strategia passa i vincoli, motivazione (quale vincolo ha scartato le alternative)

---

## 9. Side benchmark opzionale — ordinal classification  `[Fase 3]`

Per chiudere il cerchio sul fatto che le classi sono ordinate (gli errori di §5.7 sono tra classi adiacenti), vale la pena includere un confronto ordinale. Basso costo, alto valore informativo.

- [ ] Implementa `OrdinalClassifier` wrapper che scompone il problema 4-classi in 3 classificatori binari `P(y > k)` (pattern Frank & Hall 2001)
- [ ] Wrappa `XGBClassifier` e salva `xgboost_ordinal.joblib`
- [ ] Valuta con le metriche di §5.2 e aggiungi alla tabella di §7 come riga separata
- [ ] Se `severe_error_rate` scende significativamente rispetto al multiclass piatto, documentalo come finding — ma non rientra nella selezione di §8 se non supera i vincoli hard

---

## 10. Documentazione

- [ ] `technical_doc.md`: nuova sezione "Step 5.9 — Strategia ibrida regressore + classificatore" che riassume §6 e §8 (decisione finale + motivazione)
- [ ] `README.md`: una riga sotto "Modelli" — "La classificazione finale usa una strategia ibrida regressione→soglie con correzione calibrata vicino alle soglie di allerta"
- [ ] `implementation_steps.md`: spunta i punti `[ ] Esperimento aggiuntivo` dello Step 5.7 con riferimento a questo documento
- [ ] `RISULTATI_MODELLI_ATTUALI.md`: aggiorna con la tabella di §7 e il risultato di §8

---

## 11. Verifica end-to-end

La verifica e' incrementale: si esegue alla fine di ogni fase, non tutta insieme alla fine.

**Fine Fase 1**:
- [ ] `python -m step_5_classification.evaluate` produce `classification_metrics.json` con la riga `regression_to_class_xgboost` popolata
- [ ] Guard-rail di §1 tutti verdi (date range, stazioni, random_state, CV identico)

**Fine Fase 2**:
- [ ] `python -m step_5_classification.calibrate` estende la calibrazione al classifier candidato
- [ ] `python -m step_5_classification.tune_hybrid` costruisce `hybrid_oof_predictions.parquet`, esegue la grid search out-of-fold di §6.4 e scrive `hybrid_params.json`
- [ ] `python -m step_5_classification.evaluate_hybrid` valuta su test con modelli finali fittati su tutto `X_train` e scrive `hybrid_metrics.json`
- [ ] Riga `Ibrido` popolata nella tabella di §7
- [ ] Gate §0.1 valutato: l'ibrido batte `regression→soglie` puro e il classifier puro a vincoli di §8 rispettati? Se no, non viene promosso automaticamente; si applica il criterio completo di §8

**Fine progetto**:
- [ ] `python -m scripts.compare_classification_strategies` genera la tabella di §7
- [ ] `python -m scripts.select_final_model` applica il criterio di §8 e scrive `final_model_selection.json`
- [ ] `api/services/predictor.py`: aggiorna `predict()` solo se `final_model_selection.json` ha `production_ready=true` e la strategia selezionata e' diversa da quella attuale (caricando `hybrid_params.json` al startup se serve)

---

## 12. Fuori scope — esplicitamente escluso

Per evitare scope creep:

- nessuna riqualificazione dello Step 4 (regressione) **dentro questo documento**: il best regressor resta quello selezionato dalla pipeline attuale
- nessun nuovo classificatore introdotto in questa sede: l'ibrido si costruisce sopra i classificatori gia' prodotti in Step 5
- nessun cambio delle soglie di allerta `[20, 35, 50]`: sono definite dal dominio (ARPA), non sono tunable
- nessun ensemble "stacking" piu' complesso della regola §6.3 — la regola deve restare spiegabile
- nessuna calibrazione di modelli che non entrano in Fase 2 (i.e. non si calibra tutto "per completezza"): la calibrazione costa ~5× il training e si fa solo sul candidato selezionato
