@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

echo === Avvio PM10 Forecast API ===
echo.

rem --- 1) venv ------------------------------------------------------------
set "VENV_PY=%CD%\venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo ERRORE: venv non trovato: %VENV_PY%
    echo         Crealo con: python -m venv venv
    echo         Poi:        venv\Scripts\python -m pip install -r requirements.txt -r api\requirements.txt
    pause
    exit /b 1
)

rem Attivazione "manuale": niente dipendenza da activate.bat (a volte fallisce
rem silenziosamente). Forziamo PATH e VIRTUAL_ENV cosi' python/pip sono sempre
rem quelli del venv.
set "VIRTUAL_ENV=%CD%\venv"
set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"

rem Sanity check: il python del venv deve essere eseguibile
"%VENV_PY%" -c "import sys; sys.exit(0)"
if errorlevel 1 (
    echo ERRORE: impossibile eseguire %VENV_PY%
    echo         Il venv potrebbe essere corrotto. Ricrealo.
    pause
    exit /b 1
)
echo Python venv: %VENV_PY%

rem --- 2) dipendenze chiave ----------------------------------------------
"%VENV_PY%" -c "import fastapi, uvicorn, joblib, pandas, google.cloud.storage" 2>nul
if errorlevel 1 (
    echo Dipendenze mancanti nel venv. Installo api\requirements.txt ...
    "%VENV_PY%" -m pip install -r api\requirements.txt
    if errorlevel 1 (
        echo ERRORE: installazione dipendenze fallita.
        pause
        exit /b 1
    )
)

rem --- 3) artefatti modello ----------------------------------------------
set MISSING=0
for %%F in (
    "step_4_regression\artifacts\best_model.joblib"
    "step_5_classification\artifacts\best_model.joblib"
    "step_3_eda\daily_dataset_clean.parquet"
    "data\raw\sensors_registry.json"
) do (
    if not exist %%F (
        echo ERRORE: artefatto mancante: %%F
        set MISSING=1
    )
)
if "%MISSING%"=="1" (
    echo.
    echo Esegui prima scripts\run_pipeline.bat per generare modelli e dataset.
    pause
    exit /b 1
)

rem --- 4) credenziali GCS (warning soft) ---------------------------------
set GCS_OK=0
if defined GOOGLE_APPLICATION_CREDENTIALS (
    if exist "%GOOGLE_APPLICATION_CREDENTIALS%" set GCS_OK=1
)
if "%GCS_OK%"=="0" (
    call gcloud auth application-default print-access-token >nul 2>&1
    if not errorlevel 1 set GCS_OK=1
)
if "%GCS_OK%"=="0" (
    echo.
    echo ATTENZIONE: nessuna credenziale GCP rilevata.
    echo  - finestra 7gg recente da GCS non sara' disponibile
    echo  - previsioni funzioneranno solo su date coperte dal parquet
    echo  Per abilitare: gcloud auth application-default login
    echo         oppure: set GOOGLE_APPLICATION_CREDENTIALS=C:\path\sa.json
    echo.
) else (
    echo Credenziali GCP: OK
)

rem --- 5) libera la porta se occupata ------------------------------------
set HOST=127.0.0.1
set PORT=8000
set URL=http://%HOST%:%PORT%/

for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /r /c:"[:.]%PORT% .*LISTENING"') do (
    if not "%%P"=="0" (
        echo Porta %PORT% occupata dal PID %%P — la chiudo...
        taskkill /F /PID %%P >nul 2>&1
        if errorlevel 1 echo ATTENZIONE: impossibile terminare il PID %%P.
    )
)

rem attesa breve per lasciare tempo al processo di chiudersi davvero
ping -n 4 127.0.0.1 >nul

set "PORT_BUSY="
set "PORT_PID="
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /r /c:"[:.]%PORT% .*LISTENING"') do (
    if not "%%P"=="0" (
        set "PORT_BUSY=1"
        set "PORT_PID=%%P"
    )
)
if defined PORT_BUSY (
    echo ERRORE: la porta %PORT% e' ancora occupata dopo il tentativo di rilascio. PID: %PORT_PID%
    echo         Chiudi il processo che la usa oppure cambia PORT nello script.
    pause
    exit /b 1
)

rem --- 6) avvia uvicorn + apri browser -----------------------------------
echo.
echo Avvio server su %URL%
echo (Ctrl+C per terminare)
echo.

start "" /b powershell -NoProfile -Command "Start-Sleep -Seconds 3; Start-Process '%URL%'"

"%VENV_PY%" -m uvicorn api.main:app --host %HOST% --port %PORT%

echo.
echo === Server terminato ===
pause
endlocal
