@echo off
setlocal
cd /d "%~dp0.."

echo === Download JSON da GCS in data/raw/ ===
echo.

call venv\Scripts\activate.bat

set /p START_DATE=Data inizio (YYYY-MM-DD):
set /p END_DATE=Data fine   (YYYY-MM-DD):

if "%START_DATE%"=="" (
    echo ERRORE: data inizio obbligatoria.
    pause
    exit /b 1
)
if "%END_DATE%"=="" (
    echo ERRORE: data fine obbligatoria.
    pause
    exit /b 1
)

python scripts\sync_gcs.py --into-raw --start %START_DATE% --end %END_DATE%
if errorlevel 1 if not errorlevel 2 (
    echo ERRORE durante il download.
    pause
    exit /b 1
)

echo.
echo === Download completato ===
echo.
set /p DO_PIPELINE=Vuoi eseguire ingest + EDA + training (regression + classification)? [y/N]:
if /i not "%DO_PIPELINE%"=="y" (
    echo Saltato. Esegui scripts\run_pipeline.bat manualmente quando vuoi.
    pause
    exit /b 0
)

echo.
echo === Avvio pipeline completa ===
set PIPELINE_MODE=1
call scripts\run_pipeline.bat
exit /b %errorlevel%
