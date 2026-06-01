@echo off
setlocal
cd /d "%~dp0.."

echo === Download JSON from GCS to data/raw/ ===
echo.

call venv\Scripts\activate.bat

set /p START_DATE=Start date (YYYY-MM-DD):
set /p END_DATE=End date   (YYYY-MM-DD):

if "%START_DATE%"=="" (
    echo ERROR: start date is required.
    pause
    exit /b 1
)
if "%END_DATE%"=="" (
    echo ERROR: end date is required.
    pause
    exit /b 1
)

python scripts\sync_gcs.py --into-raw --start %START_DATE% --end %END_DATE%
if errorlevel 1 if not errorlevel 2 (
    echo ERROR occurred during download.
    pause
    exit /b 1
)

echo.
echo === Download complete ===
echo.
set /p DO_PIPELINE=Do you want to run ingest + EDA + training (regression + classification)? [y/N]:
if /i not "%DO_PIPELINE%"=="y" (
    echo Skipped. Run scripts\run_pipeline.bat manually whenever you want.
    pause
    exit /b 0
)

echo.
echo === Launching full pipeline ===
set PIPELINE_MODE=1
call scripts\run_pipeline.bat
exit /b %errorlevel%
