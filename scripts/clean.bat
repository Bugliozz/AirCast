@echo off
setlocal
cd /d "%~dp0.."

echo === Pulizia dati generati ===
echo Cartella progetto: %CD%
echo.
echo ATTENZIONE: verranno eliminati MySQL volume, parquet EDA e artifacts regression.
pause

echo [1/3] Fermo container Docker e rimuovo volume MySQL...
docker compose -f step_2_ingestion/compose.yaml down -v

echo [2/3] Rimuovo file generati da step_3_eda...
if exist "step_3_eda\daily_dataset.parquet"       del /f "step_3_eda\daily_dataset.parquet"
if exist "step_3_eda\daily_dataset_clean.parquet" del /f "step_3_eda\daily_dataset_clean.parquet"
if exist "step_3_eda\plots\"                      del /f /q "step_3_eda\plots\*"

echo [3/3] Rimuovo artifacts di step_4_regression...
if exist "step_4_regression\artifacts\"           del /f /q "step_4_regression\artifacts\*.joblib"
if exist "step_4_regression\artifacts\"           del /f /q "step_4_regression\artifacts\*.json"
if exist "step_4_regression\artifacts\plots\"     del /f /q "step_4_regression\artifacts\plots\*"

echo.
echo === Pulizia completata. Esegui run_pipeline.bat per ricostruire tutto. ===
pause
