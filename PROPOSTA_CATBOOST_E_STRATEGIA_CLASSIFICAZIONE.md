# Step 5.9 — CatBoost benchmark + strategia di classificazione ibrida

Piano di implementazione operativo per:

- introdurre `CatBoostRegressor` e `CatBoostClassifier` come benchmark station-aware
- isolare il contributo di `idstazione` dal contributo di CatBoost (per evitare conclusioni errate)
- costruire una classificazione finale ibrida (`regressione -> soglie` + correzione classificatore) con parametri esplicitamente definiti e tunati su validation

Tutto quello che in [`implementation_steps.md`](implementation_steps.md) e' ancora `[ ] Esperimento aggiuntivo` nello Step 5.7 viene formalizzato qui.

---

## 0. Motivazione e tradeoff (versione condensata)

- in inferenza il sistema lavora sempre sulla stessa rete di stazioni lombarde -> sfruttare `idstazione` e' coerente con lo scenario reale
- CatBoost gestisce nativamente le categoriche (`provincia`, `stagione`, `idstazione`) senza imporre ordine artificiale
- tradeoff: piu' aderenza al deployment, minore trasferibilita' a stazioni non viste. Accettabile per il progetto.
- gli errori di classificazione sono soprattutto tra classi adiacenti (vedi Step 5.7) -> la regressione conserva l'ordine naturale ed e' il backbone naturale della decisione finale

Formulazione corretta in documentazione: CatBoost non e' "peggiore perche' meno generale", e' **station-aware**. I modelli tree-based standard restano migliori come benchmark di generalizzazione.

---

## 0.1 Fasi e gate di uscita

Il piano e' organizzato in 5 fasi. Ogni fase ha un **gate** che decide se valga la pena passare alla successiva. Se il gate non e' soddisfatto, ci si ferma e si documenta il risultato. Questo riduce il rischio di costruire infrastruttura costosa (calibrazione, grid search `δ×p`, ordinal) prima di sapere se serve davvero.

| Fase | Contenuto | Gate per passare alla fase successiva |
|---|---|---|
| **1** | §1 guard-rail + §5 metriche uniformi + §6.1 baseline `regressione→soglie` con XGBoost attuale | se `regression→soglie` rispetta gia' i vincoli di §8 meglio del classifier attuale, questa diventa la baseline operativa; le fasi successive devono battere questo punto di partenza |
| **2** | §3.1 CatBoost classifier clean + §3.2 station-aware (solo classifier, **non** regressor) | CatBoost station-aware deve migliorare `f1_macro` di almeno `0.01` rispetto al miglior classifier attuale **e** non peggiorare `severe_error_rate` |
| **3** | §3.3 baseline di controllo XGBoost + target encoding di `idstazione` | se XGBoost+TE chiude >= 70% del gap contro CatBoost station-aware: stop qui, CatBoost non giustifica la dipendenza aggiuntiva. Procedere in Fase 4 solo con XGBoost+TE |
| **4** | §4 calibrazione + §6 hybrid + §6.4 tuning `δ,p` — **solo** per le 2 migliori coppie `(regressore, classificatore)` uscite da Fase 2/3 | hybrid deve battere il migliore di Fase 2/3 su `f1_macro` a vincoli di §8 rispettati |
| **5** | §9 ordinal side benchmark | opzionale; valore informativo residuo se `severe_error_rate` e' ancora sopra target |

Ogni sottosezione operativa riporta un tag `[Fase N]` per chiarezza. Le sezioni §1, §5, §7, §8, §10, §11, §12 sono trasversali (si estendono / aggiornano fase dopo fase) e non hanno tag di fase.

---

## 1. Guard-rail anti-leakage (PRIMA di toccare CatBoost)

Obbligatori perche' reintrodurre `idstazione` come feature espone al rischio di memorizzazione se lo split non e' temporale pulito.

- [ ] Verifica esplicita in `step_5_classification/train.py` che `temporal_train_test_split` produce `max(data_giorno_train) < min(data_giorno_test)` — assert hard, non solo commento
- [ ] Log in `classification_metrics.json` di `train_date_range`, `test_date_range`, `n_stations_train`, `n_stations_test`, `n_stations_intersection`
- [ ] Tutti i benchmark CatBoost girano con lo **stesso identico split** (stessi indici) usato per `xgboost_best.joblib` / `random_forest_best.joblib` / `logistic_regression_best.joblib` — pena confronti non validi
- [ ] `make_temporal_cv_splits` con `n_splits=5` identico a quello usato per gli altri modelli
- [ ] `random_state=RANDOM_STATE` fissato in **tutti** i `RandomizedSearchCV` (XGBoost, RandomForest, CatBoost clean, CatBoost station-aware, XGBoost+TE) — senza questo, il confronto a `n_iter=50` dipende dall'ordine di sampling degli iperparametri e diventa non riproducibile
- [ ] **Target encoder — vincolo anti-leakage dedicato**: qualsiasi encoder basato sul target (§3.3) deve essere rifittato **per fold** dentro la Pipeline sklearn passata a `RandomizedSearchCV`. Mai fittarlo una sola volta su `X_train` intero prima della cross-validation: il fold di validation vedrebbe target encoded con i suoi stessi valori. Sanity check obbligatorio: per una riga in validation, il valore di `idstazione_te` non deve coincidere con la media del target di quella stazione calcolata sull'intero training set.

Acceptance: se uno dei check fallisce (assert sulle date, intersezione stazioni incoerente, CV diverso, random_state mancante, target encoder fittato fuori dal fold), lo script si ferma.

---

## 2. Setup modulo (`step_5_classification/`)  `[Fase 1]`

### 2.1 Dipendenze  `[Fase 1]`

- [ ] Aggiungi `catboost>=1.2,<2.0` a `requirements.txt` (attivato in Fase 2)
- [ ] Aggiungi `category_encoders>=2.6` **oppure** verifica `scikit-learn>=1.3` (per `sklearn.preprocessing.TargetEncoder`) — necessario per §3.3, attivato in Fase 3
- [ ] `pip install -r requirements.txt` verificato in locale

### 2.2 `config.py` — nuove costanti  `[Fase 1–4]`

- [ ] `CATBOOST_CATEGORICAL_COLS = ["provincia", "stagione"]` (senza `idstazione`) — Fase 2
- [ ] `CATBOOST_CATEGORICAL_COLS_STATION_AWARE = ["provincia", "stagione", "idstazione"]` — Fase 2
- [ ] `CATBOOST_PARAM_DIST` per `RandomizedSearchCV(n_iter=50)` — Fase 2:
  - `iterations ∈ [300, 500, 800]`
  - `depth ∈ [4, 6, 8]`
  - `learning_rate ∈ [0.03, 0.05, 0.1]`
  - `l2_leaf_reg ∈ [1, 3, 5, 7]`
  - `subsample ∈ [0.7, 0.8, 1.0]`
- [ ] Parametri fissi CatBoost classifier: `loss_function="MultiClass"`, `class_weights="Balanced"`, `random_seed=RANDOM_STATE`, `verbose=False`, `allow_writing_files=False`
- [ ] Parametri fissi CatBoost regressor (solo benchmark, vedi §3.4): `loss_function="RMSE"`, `random_seed=RANDOM_STATE`, `verbose=False`, `allow_writing_files=False`
- [ ] **Nota di equivalenza pesi** (da documentare in technical_doc.md): XGBoost/RandomForest classifier usano `compute_sample_weight("balanced")` per-sample; CatBoost usa `class_weights="Balanced"` che applica l'inverso delle frequenze di classe sul training set. I due approcci producono risultati numericamente molto vicini ma non identici — documentato e accettato per mantenere ciascun modello nella sua configurazione idiomatica.
- [ ] `HYBRID_DELTA_GRID = [3.0, 5.0, 7.0]` — larghezza zona-soglia in μg/m³ (Fase 4, vedi §6)
- [ ] `HYBRID_PROB_GRID = [0.25, 0.35, 0.45]` — soglia probabilita' per regola prudenziale (Fase 4, vedi §6)
- [ ] `HYBRID_ALPHA = 1.0`, `HYBRID_BETA = 0.5` — coefficienti del metric obiettivo composto (§6.4)
- [ ] `ALERT_THRESHOLDS = [20.0, 35.0, 50.0]` (gia' implicito nella pipeline `regressione -> soglie`, esplicitalo qui)

### 2.3 Dataset helper dedicato per CatBoost  `[Fase 2]`

Il `build_preprocessor` esistente in [shared/utils.py:178](shared/utils.py#L178) e' progettato per `model_type ∈ {"linear", "tree"}` e codifica le categoriche con `OrdinalEncoder` dentro un `ColumnTransformer`. Aggiungere un terzo ramo `"catboost"` passthrough e' tecnicamente semplice ma concettualmente sbagliato: `ColumnTransformer.transform` ritorna un `ndarray`, mentre CatBoost vuole un `DataFrame` con dtypes preservati e usa `cat_features` per nome o per indice. Forzare CatBoost dentro il preprocessor generico richiederebbe di rimappare indici a mano dopo il transform — bug silenzioso appena qualcuno tocca l'ordine delle colonne.

Soluzione: percorso dedicato, non un terzo ramo del preprocessor generico.

- [ ] Nuovo modulo `shared/catboost_dataset.py` con funzione `prepare_catboost_frame(X, cat_cols, drop_cols) -> pd.DataFrame`:
  - droppa identificatori non-feature (`nomestazione`, `comune`, `pm10`, `data_giorno` se presente)
  - imputa le numeriche con la mediana di training (riusa `shared.utils.impute_missing`)
  - casta le colonne in `cat_cols` a `str` (CatBoost accetta anche int/float ma `str` rende il contratto piu' esplicito ed evita sorprese con categoricals numeriche come `idstazione`)
  - ritorna un DataFrame con dtypes preservati, pronto per `CatBoostClassifier.fit(X, y, cat_features=cat_cols)`
- [ ] `build_preprocessor` in `shared/utils.py` **non viene modificato**: CatBoost non passa di li'.
- [ ] `DROP_COLS` per CatBoost station-aware: rimuovi `idstazione` dalla drop list; mantieni `nomestazione`, `comune`, `pm10` sempre droppate.
- [ ] Pipeline di training per CatBoost: non usa `sklearn.Pipeline`; il frame e' gia' pronto quando entra in `RandomizedSearchCV`, che vede solo il modello (niente step di preprocessing sklearn-side).

---

## 3. Benchmark CatBoost — classificatori + baseline di controllo

Scopo di questa sezione: isolare l'effetto di `idstazione` dall'effetto di CatBoost in se', restando nel perimetro della **classificazione**. CatBoostRegressor e' trattato come benchmark puro (§3.4) e non entra nelle pipeline ibride.

### 3.1 Variante A — CatBoostClassifier "pulito" (senza `idstazione`)  `[Fase 2]`

- [ ] `X_train_cb, X_test_cb = prepare_catboost_frame(X, cat_cols=CATBOOST_CATEGORICAL_COLS, ...)` su train/test
- [ ] Fit `CatBoostClassifier` con `cat_features=CATBOOST_CATEGORICAL_COLS` e `RandomizedSearchCV(n_iter=50, scoring="f1_macro", cv=tscv, random_state=RANDOM_STATE)`
- [ ] Salva: `catboost_classifier_clean.joblib`

### 3.2 Variante B — CatBoostClassifier station-aware (con `idstazione`)  `[Fase 2]`

- [ ] `X_train_cb, X_test_cb = prepare_catboost_frame(X, cat_cols=CATBOOST_CATEGORICAL_COLS_STATION_AWARE, ...)` — include `idstazione` come categorical
- [ ] Stesso setup di §3.1 ma `cat_features=CATBOOST_CATEGORICAL_COLS_STATION_AWARE`
- [ ] Salva: `catboost_classifier_station.joblib`

### 3.3 Baseline di controllo — XGBoost + target encoding di `idstazione`  `[Fase 3]`

Serve a rispondere a: "il guadagno viene da CatBoost o dal fatto che stai dando `idstazione` al modello?". Se XGBoost con target encoding chiude la maggior parte del gap, CatBoost non vale la complessita' aggiuntiva.

**Pattern obbligatorio — TargetEncoder dentro Pipeline sklearn, rifittato per fold**:

Fittare un `TargetEncoder` una sola volta su `X_train` intero e passare la matrice pre-encoded a `RandomizedSearchCV` e' **leakage**: ogni fold di validation riceve valori encoded che incorporano i target delle sue stesse righe. Conseguenza concreta: f1 della baseline TE artificialmente gonfiato, rischio di concludere erroneamente che CatBoost non vale. L'encoder deve essere dentro la Pipeline, cosi' sklearn lo rifitta per fold.

- [ ] Usa `sklearn.preprocessing.TargetEncoder` (sklearn >= 1.3) oppure `category_encoders.TargetEncoder`. **Attenzione:** `idstazione` viene target-encoded, ma le altre categoriche (`provincia`, `stagione`) devono comunque essere trasformate in modo compatibile con XGBoost; non possono passare come stringhe raw. Quindi la Pipeline deve avere un preprocessor esplicito che faccia:
  - `TargetEncoder` su `idstazione`
  - `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` su `provincia`, `stagione`
  - passthrough delle feature numeriche
  Esempio schematico:
  ```python
  pipe = Pipeline([
      ("pre", ColumnTransformer([
          ("idstazione_te", TargetEncoder(smooth="auto", cv=5, random_state=RANDOM_STATE), ["idstazione"]),
          ("cat_other", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["provincia", "stagione"]),
          ("num", "passthrough", numeric_cols),
      ], remainder="drop", verbose_feature_names_out=False)),
      ("clf", XGBClassifier(...)),
  ])
  search = RandomizedSearchCV(pipe, XGBOOST_PARAM_DIST_PIPE, n_iter=50, cv=tscv, scoring="f1_macro", random_state=RANDOM_STATE)
  ```
- [ ] Prefissi dei parametri nello spazio di ricerca: `clf__n_estimators`, `clf__max_depth`, etc. (pipeline-prefixed)
- [ ] **Sanity check anti-leakage** (obbligatorio, assert hard prima del salvataggio): per ogni stazione presente sia in train sia in validation, il valore medio di `idstazione_te` sulle righe di validation del primo fold **non** deve coincidere con la media del target di quella stazione calcolata sull'intero training set (tolleranza relativa 1e-6). Se coincide, l'encoder e' stato fittato fuori dal fold.
- [ ] Applica lo stesso pattern a una seconda Pipeline `TargetEncoder + XGBRegressor` **solo** se la Fase 3 decide di valutare anche `regression→soglie` con TE (gate: CatBoost station-aware non ha chiuso il gap in Fase 2 **e** TE classifier e' competitivo)
- [ ] Salva: `xgboost_classifier_te.joblib` (Fase 3), eventualmente `xgboost_regressor_te.joblib` (solo se gate superato)
- [ ] Riporta nel confronto finale di §7

### 3.4 CatBoostRegressor — benchmark puro di regressione (separato)  `[Fase 2, opzionale]`

CatBoostRegressor **non** entra in §6 (hybrid) e **non** entra nelle righe `regression→soglie` di §7. Viene valutato solo come benchmark di regressione per completezza informativa.

- [ ] Fit `CatBoostRegressor` con `cat_features=CATBOOST_CATEGORICAL_COLS_STATION_AWARE` e `RandomizedSearchCV(n_iter=50, scoring="neg_root_mean_squared_error", cv=tscv, random_state=RANDOM_STATE)`
- [ ] Salva: `catboost_regressor_station.joblib`
- [ ] Pubblica le metriche (§5.1) in una **tabella separata** in `RISULTATI_MODELLI_ATTUALI.md` sotto "Benchmark regressione station-aware"
- [ ] **Se** CatBoostRegressor batte XGBoost su `rmse` di >= 10% e su `rmse_rosso` di >= 15%: apri un Step 4.b separato (PR a parte) per riqualificare la regressione. Non toccare Step 5.9 per questo — manterebbe il cambio confinato al suo scope.

---

## 4. Calibrazione classificatori (obbligatoria per la strategia ibrida)  `[Fase 4]`

La regola prudenziale di §6 dipende da probabilita' calibrate. Replica il pattern `CalibratedClassifierCV` gia' in uso (Step 5.5, `method="isotonic"`). **Solo le 2 migliori coppie uscite da Fase 2/3 entrano in questa fase** — non calibrare modelli che non verranno mai selezionati.

- [ ] Calibra il classifier vincente di Fase 2/3 con `CalibratedClassifierCV(method="isotonic", cv=tscv)`
- [ ] Se il secondo migliore e' a distanza `f1_macro <= 0.005`: calibra anche lui (tie-break da risolvere con l'intera tabella di §7)
- [ ] Reliability diagram one-vs-rest per le 4 classi (stesso stile di `best_model_calibrated`) salvato in `artifacts/plots/`
- [ ] Metrica di calibrazione: `expected_calibration_error` (ECE) per classe, salvato nel JSON delle metriche

### 4.1 Gotcha CatBoost + `CalibratedClassifierCV` — smoke test obbligatorio

Il wrapper sklearn di CatBoost accetta `cat_features` al `fit`, **non** al costruttore. `CalibratedClassifierCV.fit(X, y)` rifitta il base estimator su ogni fold di calibrazione e, se non riceve `cat_features` in `fit_params`, CatBoost tratta le categoriche come numeriche in silenzio. L'f1_macro del modello calibrato cala di colpo senza altro segnale.

- [ ] Passa `cat_features` con `fit_params={"cat_features": CATBOOST_CATEGORICAL_COLS_STATION_AWARE}` a `CalibratedClassifierCV.fit` — oppure usa un wrapper che memorizza `cat_features` internamente (`class CatBoostClsWithCatFeatures(CatBoostClassifier): def fit(self, X, y, **kw): return super().fit(X, y, cat_features=self._cat_features, **kw)`)
- [ ] **Smoke test** prima del run lungo: addestra CatBoost station-aware non calibrato, calcola `f1_macro` su un holdout piccolo; poi calibra e ricalcola `f1_macro` sullo **stesso holdout** usando uno dei classificatori rifittati interni al calibratore, recuperato tramite l'API/sklearn object model disponibile nella versione in uso (ad esempio dalla struttura `calibrated.calibrated_classifiers_`), **senza assumere nomi di attributi interni specifici**. Se il valore non calibrato prima e dopo il wrapping cambia di piu' di `0.01`, `cat_features` non sta passando. Fix e ripeti.

---

## 5. Metriche da salvare (uguali per tutti i modelli, altrimenti non sono comparabili)

Aggiorna `evaluate.py` per produrre in `classification_metrics.json` e `regression_metrics.json` i seguenti campi **per ogni modello** (xgboost, random_forest, logistic_regression, xgboost_te, catboost_clean, catboost_station):

### 5.1 Regressione

- [ ] `rmse`, `mae`, `r2` globali
- [ ] `rmse_per_fascia`, `mae_per_fascia` (verde/giallo/arancio/rosso) — stessa definizione gia' in uso
- [ ] `rmse_near_thresholds`: RMSE calcolato solo sui campioni con `y_true` in `[t - δ, t + δ]` per ogni `t ∈ ALERT_THRESHOLDS` e `δ = 5.0` (default)

### 5.2 Classificazione

- [ ] `f1_macro`, `accuracy`
- [ ] `precision`, `recall`, `f1` per classe
- [ ] `severe_error_rate` = `mean(|y_pred - y_true| >= 2)` (gia' definito, mantieni)
- [ ] `recall_rosso`, `precision_rosso` esplicitati in campo dedicato (leggibilita')
- [ ] `confusion_matrix` (4x4)
- [ ] `ece_per_classe` (solo per modelli calibrati)

### 5.3 Meta

- [ ] `train_date_range`, `test_date_range`, `n_test_samples`, `n_stations_*` (vedi §1)
- [ ] `cat_features_used` (lista) per documentare ogni variante CatBoost

---

## 6. Strategia ibrida `regressione -> soglie + correzione classificatore`

### 6.1 Baseline strutturale — `regressione -> soglie` pura  `[Fase 1]`

- [ ] Implementa `shared/regression_to_class.py`: `pm10 -> classe` usando `ALERT_THRESHOLDS = [20, 35, 50]`
- [ ] **Fase 1 (obbligatorio)**: valuta questa pipeline con il best regressor di Step 4 (attualmente `xgboost`, vedi `RISULTATI_MODELLI_ATTUALI.md`) e salva le metriche di §5.2 con chiave `regression_to_class_xgboost`
- [ ] **Fase 3 (solo se gate superato)**: ripeti con `xgboost_te` — **solo** se §3.3 ha prodotto `xgboost_regressor_te.joblib` (gate: CatBoost station-aware classifier non ha chiuso il gap in Fase 2)
- [ ] `catboost_regressor_*` **NON** entra in `regression→soglie`: e' benchmark puro di §3.4, non candidato per la strategia finale

### 6.2 Strategia ibrida con zona-soglia e regola prudenziale  `[Fase 4]`

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

### 6.3 Regola prudenziale — chiusa e verificabile  `[Fase 4]`

Quando il regressore e il classificatore sono discordi **in zona-soglia**:

- sia `class_severe = max(class_reg, class_cls)` (quella piu' grave tra le due)
- sia `p_severe = proba_cls[class_severe]`
- se `p_severe >= p_threshold`: `class_final = class_severe` (bias prudenziale)
- altrimenti: `class_final = class_reg` (default conservativo sul regressore, che e' piu' stabile)

Il bias prudenziale si attiva **solo** in zona-soglia + discordanza + alta confidenza sul classe grave. Non e' un override globale.

### 6.4 Tuning di `δ` e `p_threshold`  `[Fase 4]`

- [ ] Su `X_train` (mai su test): grid search su `HYBRID_DELTA_GRID × HYBRID_PROB_GRID` usando `make_temporal_cv_splits`
- [ ] Metric obiettivo composta: `score = f1_macro - α * severe_error_rate - β * (1 - recall_rosso)` con `α = 1.0`, `β = 0.5` (esplicitati in `config.py` come `HYBRID_ALPHA`, `HYBRID_BETA`)
- [ ] Salva `(δ*, p*)` in `artifacts/hybrid_params.json` insieme a curva di `score` vs `(δ, p)`
- [ ] Ripeti la search **solo** per le 2 migliori coppie `(regressore, classificatore calibrato)` uscite da Fase 2/3 — le soglie ottime possono differire

### 6.5 Valutazione finale della strategia ibrida  `[Fase 4]`

- [ ] Regressore ammesso: `xgboost` (sempre), `xgboost_te` (solo se Fase 3 gate superato). `catboost_regressor_*` esplicitamente escluso (§3.4).
- [ ] Classificatore ammesso: il migliore calibrato da §4 (CatBoost clean / station-aware / XGBoost / XGBoost+TE — dipende dall'esito di Fase 2/3).
- [ ] Su test set (ignoto al tuning): calcola tutte le metriche di §5.2 per ogni combinazione `(regressore, classificatore, δ*, p*)`
- [ ] Salva in `artifacts/hybrid_metrics.json`
- [ ] Confronta contro: `regression_to_class_xgboost` puro, classificatore puro, baseline XGBoost classifier attuale

---

## 7. Protocollo di confronto finale

**Tabella A — Classificazione** in `RISULTATI_MODELLI_ATTUALI.md`. La tabella si popola incrementalmente: le righe vengono aggiunte solo man mano che le rispettive fasi vengono eseguite.

| Strategia | Fase | f1_macro | severe_error_rate | recall_rosso | precision_rosso | RMSE_rosso |
|---|---|---|---|---|---|---|
| `logistic_regression` (stato attuale) | — | ... | ... | ... | ... | — |
| `random_forest` classifier (stato attuale) | — | ... | ... | ... | ... | — |
| `xgboost` classifier (stato attuale) | — | ... | ... | ... | ... | — |
| `xgboost` regression→soglie | 1 | ... | ... | ... | ... | ... |
| `catboost_clean` classifier | 2 | ... | ... | ... | ... | — |
| `catboost_station` classifier | 2 | ... | ... | ... | ... | — |
| `xgboost_te` classifier | 3 | ... | ... | ... | ... | — |
| `xgboost_te` regression→soglie | 3* | ... | ... | ... | ... | ... |
| **Ibrido** (best_reg, best_cls_calib, δ\*, p\*) | 4 | ... | ... | ... | ... | ... |
| `xgboost_ordinal` | 5 | ... | ... | ... | ... | — |

`*` Solo se il gate di Fase 3 lo giustifica (vedi §3.3). `catboost_regressor_*` **non** compare in questa tabella: e' benchmark puro di regressione, riportato separatamente in Tabella B.

**Tabella B — Benchmark regressione station-aware** (informativa, non decisionale per la classificazione):

| Modello | RMSE | MAE | R² | RMSE_rosso |
|---|---|---|---|---|
| `xgboost` (stato attuale) | ... | ... | ... | ... |
| `catboost_regressor_station` | ... | ... | ... | ... |

Se CatBoostRegressor batte XGBoost sui criteri di §3.4, apri Step 4.b in PR separata.

- [ ] Script `scripts/compare_classification_strategies.py` genera la Tabella A automaticamente leggendo i JSON degli artifact
- [ ] Script `scripts/compare_regression_benchmarks.py` (piu' piccolo) genera la Tabella B
- [ ] Aggiorna `RISULTATI_MODELLI_ATTUALI.md` con entrambe le tabelle e un commento di 5-8 righe sulla lettura

---

## 8. Criterio di selezione production-ready

Regola di scelta del modello finale — decisa **prima** di vedere i numeri, per evitare cherry-picking:

1. **Vincolo hard**: `severe_error_rate <= 0.025` (oggi `random_forest` ha `0.0211`, `xgboost` `0.0237`). Qualsiasi strategia che supera 2.5% e' scartata.
2. **Vincolo hard**: `recall_rosso >= 0.65` (oggi `xgboost` ha `0.6429`). Sotto questa soglia il sistema non e' un allerta utile.
3. Tra le strategie che rispettano entrambi i vincoli: sceglie quella con `f1_macro` massimo.
4. A parita' di `f1_macro` (± 0.005): preferisci la strategia piu' semplice (regressione pura > ibrido > CatBoost station-aware).

- [ ] Criterio documentato in `technical_doc.md` (sezione "Selezione del modello di allerta")
- [ ] Risultato della selezione scritto in `artifacts/final_model_selection.json` con: strategia scelta, metriche, motivazione (quale vincolo ha scartato le alternative)

---

## 9. Side benchmark opzionale — ordinal classification

Per chiudere il cerchio sul fatto che le classi sono ordinate (gli errori di §5.7 sono tra classi adiacenti), vale la pena includere un confronto ordinale. Basso costo, alto valore informativo.

- [ ] Implementa `OrdinalClassifier` wrapper che scompone il problema 4-classi in 3 classificatori binari `P(y > k)` (pattern Frank & Hall 2001)
- [ ] Wrappa `XGBClassifier` e salva `xgboost_ordinal.joblib`
- [ ] Valuta con le metriche di §5.2 e aggiungi alla tabella di §7 come riga separata
- [ ] Se `severe_error_rate` scende significativamente rispetto al multiclass piatto, documentalo come finding

---

## 10. Documentazione

- [ ] `technical_doc.md`: nuova sezione "Step 5.9 — CatBoost e strategia ibrida" che riassume §6 e §8 (decisione finale + motivazione)
- [ ] `README.md`: una riga sotto "Modelli" — "La classificazione finale usa una strategia ibrida regressione→soglie con correzione calibrata vicino alle soglie di allerta"
- [ ] `implementation_steps.md`: spunta i sei punti `[ ] Esperimento aggiuntivo` dello Step 5.7 con riferimento a questo documento
- [ ] `RISULTATI_MODELLI_ATTUALI.md`: aggiorna con la tabella di §7 e il risultato di §8

---

## 11. Verifica end-to-end

La verifica e' incrementale: si esegue alla fine di ogni fase, non tutta insieme alla fine.

**Fine Fase 1**:
- [ ] `python -m step_5_classification.evaluate` produce `classification_metrics.json` con la riga `regression_to_class_xgboost` popolata
- [ ] Guard-rail di §1 tutti verdi (date range, stazioni, random_state, CV identico)

**Fine Fase 2**:
- [ ] `python -m step_5_classification.train` esteso a CatBoost classifier (clean + station-aware)
- [ ] Smoke test di §4.1 eseguito (`cat_features` preservato attraverso eventuale wrapping)
- [ ] Riga `catboost_clean` e `catboost_station` popolate in Tabella A di §7
- [ ] Gate §0.1 valutato: se non passa, si ferma qui e si documenta l'esito

**Fine Fase 3** (solo se Fase 2 gate passato):
- [ ] TargetEncoder Pipeline di §3.3 con sanity check anti-leakage superato
- [ ] Riga `xgboost_te` classifier popolata
- [ ] Gate §0.1 valutato (TE chiude ≥ 70% del gap contro CatBoost?)

**Fine Fase 4** (solo per le 2 migliori coppie):
- [ ] `python -m step_5_classification.calibrate` estende la calibrazione ai modelli vincenti di Fase 2/3
- [ ] `python -m step_5_classification.tune_hybrid` esegue la grid search di §6.4 e scrive `hybrid_params.json`
- [ ] Riga `Ibrido` popolata in Tabella A di §7

**Fine progetto**:
- [ ] `python -m scripts.compare_classification_strategies` genera la Tabella A
- [ ] `python -m scripts.select_final_model` applica il criterio di §8 e scrive `final_model_selection.json`
- [ ] `api/services/predictor.py`: se la strategia selezionata e' diversa da quella attuale, aggiorna `predict()` (caricando `hybrid_params.json` al startup se serve)

---

## 12. Fuori scope — esplicitamente escluso

Per evitare scope creep:

- nessuna riqualificazione dello Step 4 (regressione) **dentro questo documento**: il best regressor resta quello selezionato dalla pipeline attuale. CatBoostRegressor e' benchmark puro (§3.4), non candidato per la strategia finale; se batte XGBoost sui criteri di §3.4, apre uno Step 4.b **in PR separata**, non modifica Step 5.9
- nessun tuning aggressivo di CatBoost oltre i 50 iter di RandomizedSearch — se il gap contro XGBoost e' marginale, non giustifica la complessita'
- nessun cambio delle soglie di allerta `[20, 35, 50]`: sono definite dal dominio (ARPA), non sono tunable
- nessun modello ensemble "stacking" tra regressore e classificatore piu' complesso della regola §6.3 — la regola deve restare spiegabile
- nessuna calibrazione di modelli che non entrano in Fase 4 (i.e. non si calibra tutto "per completezza"): la calibrazione costa 5× il training e si fa solo sui candidati selezionati
- nessun target encoding di `idstazione` dentro la pipeline CatBoost — CatBoost gestisce le categoriche nativamente, sovrapporre TE vanifica il confronto di §3.3
