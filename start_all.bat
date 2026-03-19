@echo off
setlocal enabledelayedexpansion

echo =================================================
echo   HandPoseEstimation - AI QA Assistent Start
echo =================================================

:: Setze den PYTHONPATH auf das src-Verzeichnis
set PYTHONPATH=%CD%\src

:: 1. PRÜFUNG: Ist Poetry installiert?
where poetry >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [FEHLER] Poetry wurde nicht gefunden. Bitte installieren Sie Poetry global.
    pause
    exit /b
)

:: 2. VERSIONS-MANAGEMENT: Erzwinge eine kompatible Python-Version (z.B. 3.11)
echo [CHECK] Validiere Python-Version (erfordert ^>=3.10, ^<3.12)...
call poetry env use 3.11 >nul 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [HINWEIS] Python 3.11 wurde nicht direkt im Pfad gefunden.
    echo Versuche, die aktuell aktive Poetry-Umgebung zu prüfen...

    :: Version der aktuellen Poetry-Umgebung abfragen
    for /f "tokens=2" %%v in ('poetry run python --version') do set PY_VER=%%v
    echo Aktuelle Umgebung nutzt Python !PY_VER!

    :: Prüfung: Wenn Version 3.12.x erkannt wird, erfolgt ein Abbruch gemäß pyproject.toml
    echo !PY_VER! | findstr "3.12" >nul
    if !ERRORLEVEL! EQU 0 (
        echo [FEHLER] Die aktuelle Python-Version 3.12 wird vom Projekt nicht erlaubt.
        echo Bitte installieren Sie Python 3.11 und fuehren Sie 'poetry env use 3.11' aus.
        pause
        exit /b
    )
)

:: 3. BACKEND STARTEN
echo [1/2] Installiere Abhängigkeiten und starte Backend (FastAPI)...
:: Startet uvicorn mit der backend_api App
start "Backend - FastAPI" cmd /k "poetry install && poetry run uvicorn handpose.backend.backend_api:app --host 0.0.0.0 --port 8000"

:: Wartezeit für die Initialisierung (Erstellung von Verzeichnissen und Laden der Modelle)
echo Warte auf Backend-Initialisierung (ca. 8 Sekunden)...
timeout /t 8 /nobreak > nul

:: 4. FRONTEND STARTEN
echo [2/2] Installiere npm-Pakete und starte Frontend (React)...
:: Navigiert zum Frontend-Pfad und startet den Dev-Server
start "Frontend - React" cmd /k "cd src/handpose/frontend && npm install && npm run dev"

echo.
echo =================================================
echo Dienste werden in separaten Fenstern gestartet.
echo Backend (API Docs): http://localhost:8000/docs
echo Frontend (App):      http://localhost:5173
echo =================================================
echo.
pause
