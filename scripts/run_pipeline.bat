@echo off
setlocal
cd /d "%~dp0.."
set PIPELINE_MODE=1

echo ==========================================
echo   AirPulita -- Full Pipeline
echo ==========================================
echo.

call "%~dp0run_02_ingest.bat"
if errorlevel 1 goto error
echo.

call "%~dp0run_03_eda.bat"
if errorlevel 1 goto error
echo.

call "%~dp0run_04_train.bat"
if errorlevel 1 goto error
echo.

call "%~dp0run_04_evaluate.bat"
if errorlevel 1 goto error
echo.

echo ==========================================
echo   Pipeline completa!
echo   Risultati in step_4_regression/artifacts/
echo ==========================================
pause
exit /b 0

:error
echo.
echo ==========================================
echo   ERRORE: pipeline interrotta.
echo ==========================================
pause
exit /b 1
