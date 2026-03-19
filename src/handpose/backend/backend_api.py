import asyncio
import logging
import os
import shutil
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator

import cv2
import numpy as np
import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from handpose import config_loader as cfg
from handpose.backend.backend_controller import BackendController
from handpose.config_loader import Settings, get_settings

logger = logging.getLogger(__name__)

app = FastAPI(title="Hand Pose Estimation API")


# ===== DEPENDENCY INJECTION =====


@lru_cache()
def get_backend() -> BackendController:
    """Gibt gecachte BackendController-Instanz zurück."""
    return BackendController(settings=get_settings())


def _setup_cors(app_instance: FastAPI) -> None:
    """Initialisiert CORS mit aktuellen Settings."""
    settings = get_settings()
    allowed_origins = settings.pipeline.app.allowed_origins
    if not isinstance(allowed_origins, list):
        allowed_origins = []

    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


_setup_cors(app)


# ===== HELPER FUNCTIONS =====


def sanitize_types(obj: Any) -> Any:
    """Konvertiert NumPy/Pandas-Typen rekursiv zu Python-Typen (für JSON)."""
    if isinstance(obj, dict):
        return {k: sanitize_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_types(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return sanitize_types(obj.to_dict(orient="index"))
    elif isinstance(obj, pd.Series):
        return sanitize_types(obj.to_dict())
    return obj


# ===== PYDANTIC MODELS =====


class ConfigUpdate(BaseModel):
    """Konfiguration Update."""

    hmm_states: int
    dbscan_clusters: int
    cluster_names: str
    camera_type: int | None = None


class RecordMode(str, Enum):
    """Recording Modi."""

    train = "train"
    production = "production"


class RecordAction(str, Enum):
    """Recording Aktionen."""

    start = "start"
    stop = "stop"


class RecordingRequest(BaseModel):
    """Recording Anfrage."""

    mode: str


# ===== API ENDPOINTS =====


@app.get("/api/video/{mode}")
async def video_feed(
    mode: str,
    show_rois: bool = False,
    settings: Settings = Depends(get_settings),
    backend: BackendController = Depends(get_backend),
) -> StreamingResponse:
    """Video-Stream vom Backend (MJPEG)."""
    return StreamingResponse(
        generate_frames(backend, settings, show_rois),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/recordings")
async def start_recording(
    request: RecordingRequest,
    settings: Settings = Depends(get_settings),
    backend: BackendController = Depends(get_backend),
) -> dict[str, str]:
    """Startet Videoaufnahme im angegebenen Modus."""
    mode = request.mode

    if mode not in ["train", "production"]:
        raise HTTPException(
            status_code=400,
            detail=f"Ungültiger Modus: {mode}. Erlaubt: 'train', 'production'",
        )

    if mode == "train":
        target_dir = Path(settings.pipeline.camera.recording_dir_training)
    else:
        target_dir = Path(settings.pipeline.camera.recording_dir_production)

    target_dir.mkdir(parents=True, exist_ok=True)
    backend.recorder_controller.recorder.video_dir = target_dir

    backend.start_recording()
    logger.info(f"Aufnahme gestartet: {mode}")
    return {"status": "started", "mode": mode}


@app.patch("/api/recordings/current")
async def stop_recording(
    backend: BackendController = Depends(get_backend),
) -> dict[str, str]:
    """Stoppt aktuelle Videoaufnahme."""
    filename = backend.stop_recording()
    if filename is None:
        logger.error("Aufnahme stopfen fehlgeschlagen")
        raise HTTPException(status_code=500, detail="Aufnahme stopfen fehlgeschlagen")
    logger.info(f"Aufnahme gestoppt: {filename}")
    return {"status": "stopped", "filename": filename}


@app.post("/api/analysis")
async def run_analysis_restful(
    settings: Settings = Depends(get_settings),
    backend: BackendController = Depends(get_backend),
) -> dict[str, str]:
    """Startet ML-Pipeline (VAE, HMM, GMM)."""
    logger.info("Starte ML-Pipeline...")
    success = backend.run_mediapipe_pipeline()
    if not success:
        logger.error("MediaPipe Analyse fehlgeschlagen")
        raise HTTPException(status_code=400, detail="MediaPipe Analyse fehlgeschlagen")

    csv_dir = Path(settings.pipeline.preprocessing.keypoints_dir_production)
    csv_files = sorted(csv_dir.glob("recording_*.csv"))

    if not csv_files or os.path.getsize(csv_files[-1]) < 100:
        logger.error("Keine validen Keypoints gefunden")
        raise HTTPException(status_code=400, detail="Keine validen Daten gefunden")

    latest_csv = str(csv_files[-1])
    try:
        backend.process_data(latest_csv)
        run_id = Path(latest_csv).stem
        logger.info(f"ML-Pipeline erfolgreich: {run_id}")
        return {"message": "Analyse abgeschlossen", "run_id": run_id}
    except Exception as e:
        logger.exception(f"ML-Pipeline Fehler: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analysis/training")
async def trigger_training_analysis_restful(
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    backend: BackendController = Depends(get_backend),
) -> dict[str, str]:
    """Startet Training-Daten-Verarbeitung im Hintergrund."""
    background_tasks.add_task(_process_training_video_task, settings, backend)
    logger.info("Training-Verarbeitung im Hintergrund gestartet")
    return {"message": "Training-Verarbeitung im Hintergrund gestartet"}


@app.post("/api/recordings/{run_id}/frames")
async def extract_frames_restful(
    run_id: str,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Extrahiert Frames aus Video nach run_id."""
    video_path = (
        Path(settings.pipeline.postprocessing.analysed_videos_dir) / f"{run_id}.mp4"
    )

    if not video_path.exists():
        video_path = (
            Path(settings.pipeline.camera.recording_dir_production) / f"{run_id}.mp4"
        )

    if not video_path.exists():
        logger.error(f"Video nicht gefunden: {run_id}")
        raise HTTPException(status_code=404, detail="Video nicht gefunden")

    output_dir = Path(settings.pipeline.postprocessing.extracted_frames_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = output_dir / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            frame_count += 1

        cap.release()
        logger.info(f"{frame_count} Frames extrahiert: {run_id}")
        return {"message": f"{frame_count} Frames extrahiert", "count": frame_count}
    except Exception as e:
        logger.exception(f"Frame-Extraktion Fehler: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
async def get_config(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Gibt aktuelle Prozesskonfiguration zurück."""
    try:
        return {
            "hmm_states": settings.process.hmm_n_states,
            "dbscan_clusters": settings.process.dbscan_n_clusters,
            "cluster_names": settings.process.cluster_names,
            "camera_type": settings.pipeline.camera.camera_type,
            "enabled_models": settings.pipeline.ml_pipeline.models,
        }
    except Exception:
        logger.exception("Fehler beim Lesen Konfiguration")
        raise HTTPException(status_code=500, detail="Fehler beim Lesen Konfiguration")


@app.post("/api/config")
async def update_config(
    config: ConfigUpdate,
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    """Aktualisiert Prozesskonfiguration (Speicherung auf Disk)."""
    try:
        new_process_config = cfg.ProcessConfig(
            hmm_n_states=config.hmm_states,
            dbscan_n_clusters=config.dbscan_clusters,
            cluster_names=config.cluster_names,
        )
        cfg.save_process_config(new_process_config)
        if config.camera_type is not None:
            cfg.save_camera_type(config.camera_type)
        get_settings.cache_clear()
        get_backend.cache_clear()

        logger.info(f"Konfiguration aktualisiert: {config}")
        return {"message": "Erfolgreich gespeichert"}
    except Exception as e:
        logger.exception("Fehler beim Speichern Konfiguration")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results")
async def list_results(settings: Settings = Depends(get_settings)) -> list[str]:
    """Listet alle verfügbaren Analyse-Ergebnisse auf."""
    csv_dir = Path(settings.pipeline.postprocessing.output_csv_dir)
    if not csv_dir.exists():
        return []
    files = sorted(csv_dir.glob("recording_*.csv"), key=os.path.getmtime, reverse=True)
    return [f.stem for f in files]


@app.get("/api/results/{run_id}")
async def get_run_results(
    run_id: str,
    backend: BackendController = Depends(get_backend),
) -> dict[str, Any]:
    """Lädt berechnete Metriken für einen Run."""
    df = backend.get_results_for_run(run_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Keine Ergebnisse gefunden")

    metrics = backend.calculate_metrics(df)
    return sanitize_types(metrics)


@app.post("/api/record/{mode}/{action}")
async def control_recording(
    mode: RecordMode,
    action: RecordAction,
    settings: Settings = Depends(get_settings),
    backend: BackendController = Depends(get_backend),
) -> dict[str, str]:
    """Steuert Video-Aufnahme (Start/Stop)."""
    if action == RecordAction.start:
        if mode == RecordMode.train:
            target_dir = Path(settings.pipeline.camera.recording_dir_training)
        else:
            target_dir = Path(settings.pipeline.camera.recording_dir_production)

        target_dir.mkdir(parents=True, exist_ok=True)
        backend.recorder_controller.recorder.video_dir = target_dir

        backend.start_recording()
        logger.info(f"Aufnahme gestartet: {mode.value}")
        return {"status": "started", "mode": mode.value}

    else:  # Stop
        filename = backend.stop_recording()
        if filename is None:
            logger.error("Aufnahme stopfen fehlgeschlagen")
            raise HTTPException(
                status_code=500, detail="Aufnahme stopfen fehlgeschlagen"
            )
        logger.info(f"Aufnahme gestoppt: {filename}")
        return {"status": "stopped", "filename": filename}


def _process_training_video_task(
    settings: Settings,
    backend: BackendController,
) -> None:
    """Verarbeitet Training-Video im Hintergrund (kopiert, analysiert, verschiebt)."""
    logger.info("Starte Training-MediaPipe im Hintergrund...")
    try:
        train_raw_dir = Path(settings.pipeline.camera.recording_dir_training)
        train_kp_dir = Path(settings.pipeline.preprocessing.keypoints_dir_training)
        train_kp_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted(train_raw_dir.glob("*.mp4"))
        if not videos:
            logger.warning("Kein Training-Video gefunden")
            return

        latest_video = videos[-1]
        prod_raw_dir = Path(settings.pipeline.camera.recording_dir_production)
        prod_kp_dir = Path(settings.pipeline.preprocessing.keypoints_dir_production)

        # Kopieren → Analysieren → Verschieben
        temp_video_path = prod_raw_dir / latest_video.name
        shutil.copy(latest_video, temp_video_path)

        success = backend.process_single_video_mediapipe(latest_video.name)

        if temp_video_path.exists():
            temp_video_path.unlink()

        if success:
            csv_file = prod_kp_dir / f"{latest_video.stem}.csv"
            if csv_file.exists():
                shutil.move(str(csv_file), str(train_kp_dir / csv_file.name))
            logger.info(f"✅ Training-Daten verarbeitet: {latest_video.name}")
        else:
            logger.error("❌ Training-Analyse fehlgeschlagen")

    except Exception as e:
        logger.exception(f"Training-Fehler: {e}")


@app.post("/api/analyze/train")
async def trigger_training_analysis(
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    backend: BackendController = Depends(get_backend),
) -> dict[str, str]:
    """Startet Training-Analyse als Background Task."""
    background_tasks.add_task(_process_training_video_task, settings, backend)
    logger.info("Training-Analyse im Hintergrund gestartet")
    return {"message": "Training-Analyse im Hintergrund gestartet"}


@app.post("/api/analyze")
async def run_analysis_pipeline(
    settings: Settings = Depends(get_settings),
    backend: BackendController = Depends(get_backend),
) -> dict[str, str | dict[str, bool]]:
    """Startet vollständige ML-Pipeline (MediaPipe → VAE → HMM → GMM).
    Außer die Modelle wurden über die Konfiguration deaktiviert.

    """
    logger.info("Starte ML-Pipeline...")
    success = backend.run_mediapipe_pipeline()
    if not success:
        logger.error("MediaPipe Fehler")
        raise HTTPException(status_code=400, detail="MediaPipe Fehler")

    csv_dir = Path(settings.pipeline.preprocessing.keypoints_dir_production)
    csv_files = sorted(csv_dir.glob("recording_*.csv"))

    if not csv_files or os.path.getsize(csv_files[-1]) < 100:
        logger.error("Keine gültigen Keypoints")
        raise HTTPException(status_code=400, detail="Keine gültigen Daten")

    latest_csv = str(csv_files[-1])
    try:
        backend.process_data(latest_csv)
        run_id = Path(latest_csv).stem

        enabled_models = settings.pipeline.ml_pipeline.models

        logger.info(f"ML-Pipeline erfolgreich: {run_id}")
        enable_gmm = enabled_models.get(
            "enable_gmm", enabled_models.get("enable_dbscan")
        )
        logger.info(
            f"Aktivierte Modelle: VAE={enabled_models.get('enable_vae')}, HMM={enabled_models.get('enable_hmm')}, GMM={enable_gmm}"
        )

        return {
            "message": "Analyse abgeschlossen",
            "run_id": run_id,
            "enabled_models": enabled_models,
        }
    except Exception as e:
        logger.exception(f"ML-Pipeline Fehler: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract-frames/{run_id}")
async def extract_frames_endpoint(
    run_id: str,
    settings: Settings = Depends(get_settings),
    backend: BackendController = Depends(get_backend),
) -> dict[str, int]:
    """Extrahiert alle Frames aus einem Video."""
    try:
        video_dir = Path(settings.pipeline.postprocessing.analysed_videos_dir)
        video_path = video_dir / f"{run_id}.mp4"

        if not video_path.exists():
            video_path = (
                Path(settings.pipeline.camera.recording_dir_production)
                / f"{run_id}.mp4"
            )

        output_dir = (
            Path(settings.pipeline.postprocessing.extracted_frames_dir) / run_id
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        count = backend.extract_all_frames_from_video(str(video_path), str(output_dir))
        logger.info(f"{count} Frames extrahiert: {run_id}")
        return {"count": count}
    except Exception as e:
        logger.exception(f"Extraktion Fehler: {e}")
        raise HTTPException(status_code=500, detail="Extraktion Fehler")


@app.get("/api/frames/{run_id}")
async def get_frames_list(
    run_id: str,
    settings: Settings = Depends(get_settings),
) -> list[str]:
    """Listet alle extrahierten Frames für einen Run auf."""
    try:
        frames_dir = (
            Path(settings.pipeline.postprocessing.extracted_frames_dir) / run_id
        )
        if not frames_dir.exists():
            return []
        files = sorted(frames_dir.glob("frame_*.png")) + sorted(
            frames_dir.glob("frame_*.jpg")
        )
        return [f.name for f in files]
    except Exception as e:
        logger.exception(f"Fehler beim Auflisten Frames: {e}")
        return []


@app.get("/api/frame/{run_id}/{frame_name}")
async def get_frame_image(
    run_id: str,
    frame_name: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Liefert einzelnes Frame-Bild."""
    image_path = (
        Path(settings.pipeline.postprocessing.extracted_frames_dir)
        / run_id
        / frame_name
    )
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Frame nicht gefunden")
    return FileResponse(path=image_path, media_type="image/png")


# ===== VIDEO STREAMING =====


def _draw_roi_overlay(frame: np.ndarray, settings: Settings) -> np.ndarray:
    """Zeichnet ROI-Ellipsen und Namen direkt in das Kamerabild."""
    rois = settings.hmm.get_rois_as_numpy()
    radius_x = settings.hmm.roi_rad_x
    radius_y = settings.hmm.roi_rad_y

    if not rois:
        return frame

    frame_with_overlay = frame.copy()
    height, width = frame_with_overlay.shape[:2]

    for roi_name, center in rois.items():
        center_x = int(float(center[0]) * width)
        center_y = int(float(center[1]) * height)
        axis_x = max(1, int(float(radius_x) * width))
        axis_y = max(1, int(float(radius_y) * height))

        cv2.ellipse(
            frame_with_overlay,
            (center_x, center_y),
            (axis_x, axis_y),
            0,
            0,
            360,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame_with_overlay,
            str(roi_name),
            (center_x - axis_x, max(20, center_y - axis_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return frame_with_overlay


async def generate_frames(
    backend: BackendController,
    settings: Settings,
    show_rois: bool = False,
) -> AsyncGenerator[bytes, None]:
    """Generator für MJPEG Video-Streaming (mit Perspektivtransformation)."""
    recorder = backend.get_raw_recorder_instance()
    if not hasattr(recorder, "cap") or recorder.cap is None:
        backend.initialize_recorder()

    while True:
        frame = getattr(recorder, "latest_ui_frame", None)

        if frame is not None:
            frame_to_encode = _draw_roi_overlay(frame, settings) if show_rois else frame
            ret, buffer = cv2.imencode(".jpg", frame_to_encode)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
        await asyncio.sleep(0.03)


@app.post("/api/camera/alignment")
async def align_camera(
    backend: BackendController = Depends(get_backend),
) -> dict[str, Any]:
    """Kalibriert Kamera mittels ArUco-Markern zur Perspektivtransformation."""
    recorder = backend.get_raw_recorder_instance()

    frame = getattr(recorder, "latest_raw_frame", None)
    if frame is None:
        frame = getattr(recorder, "latest_ui_frame", None)

    if frame is None:
        raise HTTPException(status_code=400, detail="Kamera liefert kein Bild")

    result = backend.align_camera_with_aruco(frame)
    return result


@app.delete("/api/camera/reset-alignment")
async def reset_camera_alignment(
    backend: BackendController = Depends(get_backend),
) -> dict[str, str]:
    """Setzt Kamera-Ausrichtung zurück."""
    backend.reset_alignment()
    logger.info("Kamera-Ausrichtung zurückgesetzt")
    return {"message": "Ausrichtung zurückgesetzt"}


@app.post("/api/cleanup")
async def cleanup_session_data(
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    """Löscht alle Production-Daten beim Session-Ende (Tab-Schließen)."""
    try:

        prod_raw_dir = Path(settings.pipeline.camera.recording_dir_production)
        prod_kp_dir = Path(settings.pipeline.preprocessing.keypoints_dir_production)
        prod_features_dir = Path(
            settings.pipeline.preprocessing.features_dir_production
        )
        results_dir = Path(settings.pipeline.postprocessing.output_csv_dir)

        deleted_count = 0

        # Lösche Production-Videos
        if prod_raw_dir.exists():
            for video_file in prod_raw_dir.glob("*.mp4"):
                try:
                    video_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Konnte Video nicht löschen: {e}")

        # Lösche Production-Keypoints
        if prod_kp_dir.exists():
            for csv_file in prod_kp_dir.glob("*.csv"):
                try:
                    csv_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Konnte CSV nicht löschen: {e}")

        # Lösche Production-Features
        if prod_features_dir.exists():
            for feature_file in prod_features_dir.glob("*.pkl"):
                try:
                    feature_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Konnte Feature nicht löschen: {e}")

        # Lösche die Results
        if results_dir.exists():
            for result_file in results_dir.glob("recording_*.csv"):
                try:
                    result_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Konnte Resultat nicht löschen: {e}")

        logger.info(f"Session-Cleanup abgeschlossen: {deleted_count} Dateien gelöscht")
        return {"message": "Cleanup abgeschlossen", "deleted_count": str(deleted_count)}
    except Exception as e:
        logger.exception(f"Cleanup-Fehler: {e}")
        return {"message": "Cleanup mit Fehler abgeschlossen"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
