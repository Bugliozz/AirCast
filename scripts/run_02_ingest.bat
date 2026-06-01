@echo off
setlocal
cd /d "%~dp0.."

echo === Step 2: Ingestion ===

call venv\Scripts\activate.bat

echo [1/2] Starting MySQL container...
docker compose -f step_2_ingestion/compose.yaml up -d mysql

echo      Waiting for MySQL to become healthy (max 90s)...
set attempts=0
:wait_loop
set /a attempts+=1
for /f %%i in ('docker inspect --format={{.State.Health.Status}} exam_mysql 2^>nul') do set STATUS=%%i
if "%STATUS%"=="healthy" goto mysql_ready
if %attempts% geq 30 (
    echo ERROR: MySQL did not become healthy in time.
    if not defined PIPELINE_MODE pause
    exit /b 1
)
timeout /t 3 /nobreak >nul
goto wait_loop

:mysql_ready
echo      MySQL is healthy.
echo [2/2] Running ingest.py...
python -m step_2_ingestion.ingest
if errorlevel 1 (
    echo ERROR during ingestion.
    if not defined PIPELINE_MODE pause
    exit /b 1
)

echo === Step 2 completed ===
if not defined PIPELINE_MODE pause
