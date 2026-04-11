@echo off
setlocal
cd /d "%~dp0.."

echo === Step 5: Classification Calibration ===

if not exist "step_5_classification\artifacts\best_model.joblib" (
    echo ERRORE: best_model.joblib non trovato.
    echo         Esegui prima run_05_train.bat
    if not defined PIPELINE_MODE pause
    exit /b 1
)

if not exist "step_5_classification\artifacts\train_data.joblib" (
    echo ERRORE: train_data.joblib non trovato.
    echo         Esegui prima run_05_train.bat
    if not defined PIPELINE_MODE pause
    exit /b 1
)

call venv\Scripts\activate.bat

python -m step_5_classification.calibrate
if errorlevel 1 (
    echo ERRORE durante la calibrazione.
    if not defined PIPELINE_MODE pause
    exit /b 1
)

echo === Step 5 calibrazione completata ===
echo     Plot: step_5_classification/artifacts/plots/
if not defined PIPELINE_MODE pause
