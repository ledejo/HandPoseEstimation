import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from handpose.config_loader import get_settings

logger = logging.getLogger(__name__)


class RecorderStatus(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    ERROR = "error"


@dataclass
class RecorderState:
    status: RecorderStatus
    is_recording: bool
    frame_count: int
    current_filename: str | None
    elapsed_time: float
    fps: int
    resolution: tuple
    message: str


class RawVideoRecorder:
    def __init__(self, cam_index: int = 0, target_fps: int = 30, video_dir: str = None):
        self.cam_index = cam_index
        self.target_fps = target_fps
        self.cap: cv2.VideoCapture | None = None
        self.video_writer: cv2.VideoWriter | None = None
        self.is_recording = False
        self.frame_count = 0
        self._frame_count_lock = threading.Lock()
        self._writer_lock = threading.Lock()

        self.latest_ui_frame = None
        self.latest_raw_frame = None
        self.perspective_matrix = None

        # Dynamische Größen für den Zuschnitt
        self.crop_width = 0
        self.crop_height = 0

        self.video_dir = (
            Path(video_dir)
            if video_dir
            else Path(get_settings().pipeline.camera.recording_dir_production)
        )
        self.current_filename: str | None = None
        self.record_start_real: float | None = None
        self.recording_start_time: float | None = None

        # Originalauflösung der Kamera
        self.width = 0
        self.height = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_camera_initialized = False  # Flag für persistente Kamera-Verbindung

    def init_camera(self) -> bool:
        """
        Initialisiert die Kamera einmalig beim Start - persistente Verbindung.
        """
        if self._is_camera_initialized and self.cap is not None and self.cap.isOpened():
            logger.debug("Kamera bereits initialisiert - keine Neu-Initialisierung")
            return True

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Dynamisches Auslesen der tatsächlichen Kameraauflösung
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Starte Capture-Thread nur wenn noch nicht laufen
        if not self._thread or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

        self._is_camera_initialized = True
        logger.debug(
            f"Kamera initialisiert: {self.width}x{self.height} @ {self.target_fps} FPS"
        )
        return True

    def set_top_down_perspective(
        self, top_left, top_right, bottom_right, bottom_left, margin=0
    ):
        """
        Berechnet die Matrix für eine gerade Draufsicht.
        Der Bereich der Marker wird exakt aus dem Bild ausgeschnitten und um einen Rand (margin) erweitert.
        """
        # 1. Quellpunkte (die Marker im Originalbild)
        src_points = np.float32([top_left, top_right, bottom_right, bottom_left])

        # 2. Reale Breite und Höhe des durch die Marker markierten Bereichs in Pixeln berechnen
        width_top = np.linalg.norm(np.array(top_right) - np.array(top_left))
        width_bottom = np.linalg.norm(np.array(bottom_right) - np.array(bottom_left))
        marker_width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
        height_right = np.linalg.norm(np.array(bottom_right) - np.array(top_right))
        marker_height = int(max(height_left, height_right))

        # 3. Zielauflösung definieren (Marker-Fläche + Rand auf allen 4 Seiten)
        self.crop_width = marker_width + (2 * margin)
        self.crop_height = marker_height + (2 * margin)

        # 4. Zielpunkte: Die Marker rücken um den "margin" nach innen
        dst_points = np.float32(
            [
                [margin, margin],  # Oben Links
                [margin + marker_width - 1, margin],  # Oben Rechts
                [margin + marker_width - 1, margin + marker_height - 1],  # Unten Rechts
                [margin, margin + marker_height - 1],  # Unten Links
            ]
        )

        # 5. Matrix berechnen und speichern
        self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        logger.info(
            f"Zuschnitt-Matrix mit {margin}px Rand erstellt. Neue Video-Größe: {self.crop_width}x{self.crop_height}"
        )

    def _capture_loop(self) -> None:
        """
        Haupt-Capture-Loop für kontinuierliche Frame-Erfassung.
        """
        while not self._stop_event.is_set():
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Auskommentiert für korrekte Koordinaten-Übereinstimmung der Marker
                    frame = cv2.flip(frame, 1)

                    self.latest_raw_frame = frame.copy()

                    # Wenn eine Matrix gesetzt wurde, wende sie an
                    if self.perspective_matrix is not None:
                        try:
                            # Schneidet exakt auf Marker-Bereich zu
                            frame = cv2.warpPerspective(
                                frame,
                                self.perspective_matrix,
                                (self.crop_width, self.crop_height),
                            )
                        except Exception as e:
                            # GIBT NUN EINEN FEHLER AUS, STATT IHN ZU IGNORIEREN!
                            logger.error(f"Fehler beim Warping (Zuschneiden): {e}")

                    self.latest_ui_frame = frame.copy()

                    with self._writer_lock:
                        if self.is_recording and self.video_writer is not None:
                            try:
                                self.video_writer.write(frame)
                                with self._frame_count_lock:
                                    self.frame_count += 1
                            except cv2.error as e:
                                logger.error(f"Fehler beim Frame schreiben: {e}")
                                self.is_recording = False
                                if self.video_writer is not None:
                                    self.video_writer.release()
                                    self.video_writer = None
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def start_recording(self) -> dict:
        """Startet die Video-Aufnahme."""
        if self.is_recording:
            return {"success": False, "message": "Läuft bereits"}
        try:
            self.current_filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S")
            self.video_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.video_dir / f"{self.current_filename}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            # Nutzt die zugeschnittene Auflösung, falls ein Zuschnitt aktiv ist
            size = (
                (self.crop_width, self.crop_height)
                if self.perspective_matrix is not None
                else (self.width, self.height)
            )

            self.video_writer = cv2.VideoWriter(
                str(out_path), fourcc, self.target_fps, size
            )
            if self.video_writer is None or not self.video_writer.isOpened():
                raise RuntimeError(
                    f"VideoWriter konnte nicht geöffnet werden: {out_path}"
                )
            with self._frame_count_lock:
                self.frame_count = 0
            with self._writer_lock:
                self.is_recording = True

            self.record_start_real = time.time()
            self.recording_start_time = time.time()
            return {
                "success": True,
                "message": "Aufnahme gestartet",
                "filename": self.current_filename,
            }
        except Exception as e:
            return {"success": False, "message": str(e)}

    def stop_recording(self) -> dict:
        """Beendet die Aufnahme und gibt statistische Daten zurück."""
        with self._writer_lock:
            if not self.is_recording:
                return {"success": False, "message": "Keine aktive Aufnahme"}

            self.is_recording = False
            filename = self.current_filename

            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

        with self._frame_count_lock:
            frames = self.frame_count

        duration = time.time() - (self.record_start_real or time.time())
        actual_fps = (frames / duration) if duration > 0 else 0

        return {
            "success": True,
            "filename": filename,
            "frames": frames,
            "duration": round(duration, 2),
            "actual_fps": round(actual_fps, 2),
        }

    def get_status(self) -> RecorderState:
        """Gibt den aktuellen Recorder-Status zurück."""
        elapsed = time.time() - (self.recording_start_time or time.time())
        return RecorderState(
            status=(
                RecorderStatus.RECORDING if self.is_recording else RecorderStatus.IDLE
            ),
            is_recording=self.is_recording,
            frame_count=self.frame_count,
            current_filename=self.current_filename,
            elapsed_time=round(elapsed, 2),
            fps=self.target_fps,
            resolution=(
                (self.crop_width, self.crop_height)
                if self.perspective_matrix is not None
                else (self.width, self.height)
            ),
            message="Aufnahme läuft" if self.is_recording else "Bereit",
        )

    def reset_camera(self) -> bool:
        """Setzt die Kamera zurück und initialisiert sie neu."""
        self._is_camera_initialized = False
        return self.init_camera()

    def list_recordings(self) -> dict:
        """Listet alle verfügbaren Videos im Video-Verzeichnis auf.

        Returns:
            dict: Status-Dictionary mit Erfolgsflag, Nachricht und Anzahl der Videos.
        """
        try:
            if not self.video_dir.exists():
                return {
                    "success": True,
                    "message": "Keine Videos vorhanden",
                    "count": 0,
                }

            videos = list(self.video_dir.glob("recording_*.mp4"))
            count = len(videos)

            return {
                "success": True,
                "message": f"{count} Videos gefunden",
                "count": count,
                "videos": [v.name for v in sorted(videos)],
            }
        except Exception as e:
            logger.error(f"Fehler beim Auflisten von Aufnahmen: {e}")
            return {"success": False, "message": str(e), "count": 0}

    def __del__(self) -> None:
        """Beendet den Capture-Thread und gibt Kamera-Ressourcen frei."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()

    def run(self):
        """Hauptschleife für interaktiven Modus"""
        if not self.init_camera():
            return

        logger.debug(
            "\n"
            + "=" * 60
            + "\n"
            + "RAW VIDEO RECORDER (30 FPS)\n"
            + "=" * 60
            + "\n"
            + "Steuerung:\n"
            + "  R = Start/Stop Aufnahme\n"
            + "  Q = Beenden\n"
            + "=" * 60
            + "\n"
        )

        while True:
            frame = self.latest_ui_frame
            if frame is None:
                time.sleep(0.01)
                continue
            display_frame = frame.copy()

            # Overlay für das Live-Display
            status = "REC" if self.is_recording else "kein REC"
            color = (0, 0, 255) if self.is_recording else (200, 200, 200)

            cv2.putText(
                display_frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
            )
            cv2.putText(
                display_frame,
                f"Frames: {self.frame_count}",
                (15, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                f"{self.target_fps} FPS (roh)",
                (15, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                "R=Start/Stop  Q=Quit",
                (15, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 200, 255),
                2,
            )

            try:
                cv2.imshow("Video Recorder", display_frame)
            except Exception:
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                if self.is_recording:
                    self.stop_recording()
                else:
                    self.start_recording()

        if self.is_recording:
            self.stop_recording()

        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
