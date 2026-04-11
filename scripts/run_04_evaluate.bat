@echo off
setlocal
cd /d "%~dp0.."

echo === Step 4: Regression Evaluation ===

if not exist "step_3_eda\daily_dataset_clean.parquet" (
    echo ERRORE: step_3_eda\daily_dataset_clean.parquet non trovato.
    echo         Esegui prima run_03_eda.bat
    if not defined PIPELINE_MODE pause
    exit /b 1
)

if not exist "step_4_regression\artifacts\elasticnet_best.joblib" (
    echo ERRORE: artifacts non trovati.
    echo         Esegui prima run_04_train.bat
    if not defined PIPELINE_MODE pause
    exit /b 1
)

call venv\Scripts\activate.bat

python -m step_4_regression.evaluate
if errorlevel 1 (
    echo ERRORE durante la valutazione.
    if not defined PIPELINE_MODE pause
    exit /b 1
)

echo === Step 4 valutazione completata ===
echo     Metriche: step_4_regression/artifacts/regression_metrics.json
echo     Plot:     step_4_regression/artifacts/plots/
if not defined PIPELINE_MODE pause
