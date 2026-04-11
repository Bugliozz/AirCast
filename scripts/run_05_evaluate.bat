@echo off
setlocal
cd /d "%~dp0.."

echo === Step 5: Classification Evaluation ===

if not exist "step_3_eda\daily_dataset_clean.parquet" (
    echo ERRORE: step_3_eda\daily_dataset_clean.parquet non trovato.
    echo         Esegui prima run_03_eda.bat
    if not defined PIPELINE_MODE pause
    exit /b 1
)

if not exist "step_5_classification\artifacts\best_model.joblib" (
    echo ERRORE: artifacts non trovati.
    echo         Esegui prima run_05_train.bat
    if not defined PIPELINE_MODE pause
    exit /b 1
)

call venv\Scripts\activate.bat

python -m step_5_classification.evaluate
if errorlevel 1 (
    echo ERRORE durante la valutazione.
    if not defined PIPELINE_MODE pause
    exit /b 1
)

echo === Step 5 valutazione completata ===
echo     Metriche: step_5_classification/artifacts/classification_metrics.json
echo     Plot:     step_5_classification/artifacts/plots/
if not defined PIPELINE_MODE pause
