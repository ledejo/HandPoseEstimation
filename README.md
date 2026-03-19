# HandPoseEstimation

Ein System zur Echtzeit-Erkennung und ML-basierten Analyse von Handbewegungen in Montageprozessen. Es extrahiert Hand-Keypoints aus Videos und wertet diese automatisch auf Prozessschritte und Anomalien aus.

## Tech-Stack

- Backend: Python (3.10/3.11), FastAPI, MediaPipe, Poetry
- Frontend: React, Vite, Tailwind CSS, npm
- ML-Pipeline: HMM (Phasenerkennung), DBSCAN (Teile-Erkennung), VAE (Anomalieerkennung)

## Voraussetzungen

- **Python** `3.10` bis `<3.12`
- **Poetry** (Dependency-Management für Backend/ML)
- **Node.js** + **npm** für das React-Frontend
- Optional: Webcam für Live-Aufnahme

## Quick-Start (Windows)
Um das System inklusive Kamera-Setup initial aufzusetzen, folgen Sie diesen Schritten:

Voraussetzungen: Stellen Sie sicher, dass Python 3.11, Node.js und Poetry auf Ihrem System installiert sind.

Hardware: Schließen Sie das Kamera-Setup an Ihren USB-Port an.

Start: Führen Sie die Datei start_all.bat im Hauptverzeichnis per Doppelklick aus.

Das Skript installiert automatisch alle Abhängigkeiten und startet zwei Fenster (Backend & Frontend).

Browser: Öffnen Sie http://localhost:5173

## Entwicklung (lokal)

### 1. Repository klonen

```bash
git clone [https://github.com/Lu1sTV/HandPoseEstimation.git](https://github.com/Lu1sTV/HandPoseEstimation.git)
cd HandPoseEstimation
```

### 2. Abhängigkeiten installieren

#### 2.1. Python-Abhängigkeiten installieren

```bash
poetry install --with dev
```

#### 2.2. Frontend-Abhängigkeiten installieren

```bash
cd src/handpose/frontend
npm install
```

### 3. Backend starten

```bash
poetry run uvicorn handpose.backend.backend_api:app --reload --host 0.0.0.0 --port 8000
```


### 4. Frontend starten

```bash
cd src/handpose/frontend
npm run dev

```

Frontend erreichbar unter:

- `http://localhost:5173`



## Qualitätssicherung
In der PreCommit config sind qualitätssichernde Konfigurationen festgelegt, jedoch können ebenfalls die folgenden Commands manuell ausgeführt werden.

### Python Lint

```bash
poetry run ruff check .
```

### Python Format-Check

```bash
poetry run black --check .
```

### Python Tests

```bash
poetry run pytest
```

### Frontend Lint

```bash
cd src/handpose/frontend
npm run lint
```

### Frontend Tests

```bash
cd src/handpose/frontend
npm test
```
