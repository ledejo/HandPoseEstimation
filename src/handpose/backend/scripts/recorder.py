"""
Hand-Pose Recorder mit MediaPipe Hand-Tracking.

Funktionalität:
- Kamera-Feed mit MediaPipe Hand-Tracking (bis zu 2 Hände)
- Start/Stop Aufzeichnung per Tastendruck
- Speichert Videos und Keypoints-CSV mit gleichem Namen
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import pandas as pd

logger = logging.getLogger(__name__)


class HandPoseRecorder:
    """Hauptklasse für die Hand-Pose-Aufzeichnung mit MediaPipe Tracking."""

    def __init__(self) -> None:
        """Initialisiert Recorder mit Kamera und MediaPipe Setup."""
        # MediaPipe Hand-Tracking
        self.mp_hands: Any = mp.solutions.hands
        self.mp_drawing: Any = mp.solutions.drawing_utils
        self.hands: Any = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Recording Status
        self.is_recording: bool = False
        self.current_filename: str | None = None
        self.video_writer: Any | None = None
        self.keypoints_data: list[dict[str, Any]] = []

        # Pfad-Konfiguration
        self.base_path: Path = Path(__file__).parent.parent.parent.parent
        self.video_path: Path = self.base_path / "data" / "01_raw" / "production"
        self.csv_path: Path = (
            self.base_path / "data" / "03_processed" / "keypoints" / "production"
        )

        # Kamera-Eigenschaften
        self.cap: Any | None = None
        self.fps: int = 30
        self.width: int = 640
        self.height: int = 480

        logger.debug("HandPoseRecorder initialisiert")

    def initialize_camera(self) -> bool:
        """Initialisiert Kamera mit Fehlerbehandlung. Returns True bei Erfolg."""
        logger.info("Versuche Kamera zu öffnen...")

        # Kamera öffnen
        self.cap = cv2.VideoCapture(0)

        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Eigenschaften auslesen
                self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(
                    f"✅ Kamera gefunden: {self.width}x{self.height} @ {self.fps} FPS"
                )
                return True
            else:
                self.cap.release()
                self.cap = None
        else:
            self.cap = None

        # Fehlerbehandlung
        logger.error(
            "❌ Keine funktionierende Kamera gefunden!\n   1. Prüfe ob Kamera nicht von anderer App verwendet wird\n   2. Überprüfe Kamera-Berechtigungen in Systemeinstellungen\n   3. Versuche USB-Kamera"
        )
        return False

    def generate_filename(self) -> str:
        """Generiert eindeutigen Dateinamen mit Timestamp (Format: recording_YYYYMMDD_HHMMSS)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"recording_{timestamp}"

    def start_recording(self) -> None:
        """Startet neue Aufnahme mit VideoWriter und Keypoint-Speicher."""
        if not self.is_recording:
            self.current_filename = self.generate_filename()

            # Video Writer erstellen
            video_file = self.video_path / f"{self.current_filename}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                str(video_file), fourcc, self.fps, (self.width, self.height)
            )

            self.keypoints_data = []
            self.is_recording = True
            logger.info(f"📹 Aufnahme gestartet: {self.current_filename}")

    def stop_recording(self) -> None:
        """Stoppt Aufnahme und speichert Video + Keypoints-CSV."""
        if self.is_recording:
            self.is_recording = False

            # Video schließen
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            # CSV speichern
            if self.keypoints_data:
                df = pd.DataFrame(self.keypoints_data)
                csv_file = self.csv_path / f"{self.current_filename}.csv"
                df.to_csv(csv_file, index=False, sep=";", decimal=".")
                logger.info(
                    f"Gespeichert: {len(self.keypoints_data)} Frames zu {csv_file}"
                )

            self.current_filename = None

    def extract_hand_keypoints(self, hand_landmarks: Any) -> dict[str, list[float]]:
        """Extrahiert x, y, z Koordinaten aller 21 Keypoints einer Hand."""
        keypoints = {"x": [], "y": [], "z": []}
        for landmark in hand_landmarks.landmark:
            keypoints["x"].append(landmark.x)
            keypoints["y"].append(landmark.y)
            keypoints["z"].append(landmark.z)
        return keypoints

    def process_frame(self, frame: Any) -> Any:
        """Verarbeitet Frame: Spiegelung, Hand-Tracking, Keypoint-Extraktion, Zeichnung."""
        # Frame horizontal spiegeln
        frame = cv2.flip(frame, 1)

        # MediaPipe Hand-Tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Datenzeile initialisieren
        row_data = {"timestamp": time.time()}

        # Alle 126 Keypoint-Spalten mit None initialisieren
        for prefix in ["l", "r"]:
            for coord in ["x", "y", "z"]:
                for i in range(21):
                    row_data[f"{prefix}_{coord}_{i}"] = None

        # Erkannte Hände verarbeiten
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Skelett zeichnen
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # Handedness bestimmen
                label = handedness.classification[0].label
                confidence = handedness.classification[0].score
                prefix = "r" if label == "Left" else "l"

                # Debug-Text zeichnen
                wrist = hand_landmarks.landmark[0]
                cx, cy = int(wrist.x * self.width), int(wrist.y * self.height)
                cv2.putText(
                    frame,
                    f"{label} ({confidence:.2f})",
                    (cx - 40, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                # Keypoints speichern
                keypoints = self.extract_hand_keypoints(hand_landmarks)
                for i in range(21):
                    row_data[f"{prefix}_x_{i}"] = keypoints["x"][i]
                    row_data[f"{prefix}_y_{i}"] = keypoints["y"][i]
                    row_data[f"{prefix}_z_{i}"] = keypoints["z"][i]
                row_data[f"{prefix}_handedness_confidence"] = confidence

        # Daten speichern (nur wenn aufgezeichnet)
        if self.is_recording:
            self.keypoints_data.append(row_data)

        return frame

    def draw_ui(self, frame: Any) -> Any:
        """Zeichnet UI-Elemente: Status, Frame-Counter, Anleitung."""
        # Status-Indikator
        status = "RECORDING" if self.is_recording else "BEREIT"
        color = (0, 0, 255) if self.is_recording else (255, 255, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Frame-Counter
        if self.is_recording:
            frames = len(self.keypoints_data)
            cv2.putText(
                frame,
                f"Frames: {frames}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Anleitung anzeigen
        instructions = ["Steuerung:", "S - Start", "T - Stop", "Q - Beenden"]
        for i, text in enumerate(instructions):
            cv2.putText(
                frame,
                text,
                (10, self.height - 120 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        return frame

    def run(self) -> None:
        """Hauptschleife: Kamera-Feed mit Hand-Tracking und Aufnahme."""

        if not self.initialize_camera():
            return

        logger.info("Steuerung: S=Start, T=Stop, Q=Beenden")

        # Hauptschleife
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Kamera-Fehler")
                break

            frame = self.process_frame(frame)
            frame = self.draw_ui(frame)

            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)

            cv2.imshow("Hand Pose Recorder", frame)

            # Tastatur-Steuerung
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") or key == ord("S"):
                self.start_recording()
            elif key == ord("t") or key == ord("T"):
                self.stop_recording()
            elif key == ord("q") or key == ord("Q"):
                break

        # Cleanup
        if self.is_recording:
            self.stop_recording()

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        logger.info("✅ Programm beendet")


def main() -> None:
    """Einstiegspunkt der Anwendung."""
    recorder = HandPoseRecorder()
    recorder.run()


if __name__ == "__main__":
    main()
