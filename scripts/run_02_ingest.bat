@echo off
setlocal
cd /d "%~dp0.."

echo === Step 2: Ingestion ===

call .venv\Scripts\activate.bat

echo [1/2] Avvio container MySQL...
docker compose -f step_2_ingestion/compose.yaml up -d mysql

echo      Attendo che MySQL sia healthy (max 90s)...
set attempts=0
:wait_loop
set /a attempts+=1
for /f %%i in ('docker inspect --format={{.State.Health.Status}} exam_mysql 2^>nul') do set STATUS=%%i
if "%STATUS%"=="healthy" goto mysql_ready
if %attempts% geq 30 (
    echo ERRORE: MySQL non e' diventato healthy in tempo.
    if not defined PIPELINE_MODE pause
    exit /b 1
)
timeout /t 3 /nobreak >nul
goto wait_loop

:mysql_ready
echo      MySQL e' healthy.
echo [2/2] Esecuzione ingest.py...
python -m step_2_ingestion.ingest
if errorlevel 1 (
    echo ERRORE durante l'ingestion.
    if not defined PIPELINE_MODE pause
    exit /b 1
)

echo === Step 2 completato ===
if not defined PIPELINE_MODE pause
