@echo off
setlocal
cd /d "%~dp0.."

echo === Step 5: Classification Training ===

if not exist "step_3_eda\daily_dataset_clean.parquet" (
    echo ERRORE: step_3_eda\daily_dataset_clean.parquet non trovato.
    echo         Esegui prima run_03_eda.bat
    if not defined PIPELINE_MODE pause
    exit /b 1
)

call venv\Scripts\activate.bat

python -m step_5_classification.train
if errorlevel 1 (
    echo ERRORE durante il training.
    if not defined PIPELINE_MODE pause
    exit /b 1
)

echo === Step 5 training completato ===
echo     Artifacts: step_5_classification/artifacts/
if not defined PIPELINE_MODE pause
