@echo off
setlocal
cd /d "%~dp0.."

echo === Step 3: EDA ===

call venv\Scripts\activate.bat

python -m step_3_eda.eda
if errorlevel 1 (
    echo ERROR during EDA.
    if not defined PIPELINE_MODE pause
    exit /b 1
)

echo === Step 3 completed ===
echo     Output: step_3_eda/daily_dataset_clean.parquet
if not defined PIPELINE_MODE pause
