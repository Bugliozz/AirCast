@echo off
setlocal
cd /d "%~dp0.."

echo === Step 4: Regression Training ===

if not exist "step_3_eda\daily_dataset_clean.parquet" (
    echo ERRORE: step_3_eda\daily_dataset_clean.parquet non trovato.
    echo         Esegui prima run_03_eda.bat
    if not defined PIPELINE_MODE pause
    exit /b 1
)

call .venv\Scripts\activate.bat

python -m step_4_regression.train
if errorlevel 1 (
    echo ERRORE durante il training.
    if not defined PIPELINE_MODE pause
    exit /b 1
)

echo === Step 4 training completato ===
echo     Artifacts: step_4_regression/artifacts/
if not defined PIPELINE_MODE pause
