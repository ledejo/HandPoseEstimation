import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

from handpose.config_loader import get_settings

logger = logging.getLogger(__name__)


class ProcessorStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class ProcessorState:
    status: ProcessorStatus
    is_processing: bool
    current_file: str | None
    progress: float
    message: str


class MediaPipeVideoProcessor:
    """
    Verarbeitet Videos mit MediaPipe Hands zur Keypoint-Extraktion.

    Extrahiert Hand-Keypoints aus Video-Frames und speichert diese als CSV-Dateien.
    """

    def __init__(
        self,
        video_dir: Path | None = None,
        csv_dir: Path | None = None,
        preview: bool = False,
    ) -> None:
        """
        Initialisiert den MediaPipeVideoProcessor.

        Args:
            video_dir (Path | None): Verzeichnis mit Input-Videos. Falls None, Standard-Pfad.
            csv_dir (Path | None): Verzeichnis für Output-CSV-Dateien. Falls None, Standard-Pfad.
            preview (bool): Zeige Video-Preview während Verarbeitung. Standard: False.
        """
        self.preview = preview

        # Lade Pfade aus Config
        settings = get_settings()
        self.video_dir = video_dir or Path(
            settings.pipeline.camera.recording_dir_production
        )
        self.csv_dir = csv_dir or Path(
            settings.pipeline.preprocessing.keypoints_dir_production
        )
        self.analyzed_video_dir = Path(
            settings.pipeline.postprocessing.analysed_videos_dir
        )

        for d in [self.csv_dir, self.analyzed_video_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.is_processing = False
        self.current_file = None

    def parse_start_timestamp(self, stem: str) -> float:
        """
        Parsed den Timestamp aus der Datei-Bezeichnung.

        Erwartet Format: recording_YYYYMMDD_HHMMSS

        Args:
            stem (str): Dateiname ohne Extension (z.B. 'recording_20260225_125121').

        Returns:
            float: Unix-Timestamp oder aktuelle Zeit falls Parsing fehlschlägt.
        """
        match = re.search(r"recording_(\d{8}_\d{6})", stem)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").timestamp()

        return time.time()

    def process_video(self, video_path: Path, show_preview: bool = False) -> bool:
        """
        Verarbeitet ein einzelnes Video und extrahiert Hand-Keypoints.

        Args:
            video_path (Path): Pfad zur Eingabe-Video-Datei.
            show_preview (bool): Zeige Live-Preview während Verarbeitung. Standard: False.

        Returns:
            bool: True wenn erfolgreich verarbeitet, False sonst.
        """
        if not video_path.exists():
            logger.error(f"Datei nicht gefunden: {video_path}")
            return False

        self.is_processing = True
        self.current_file = video_path.name

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = self.analyzed_video_dir / f"{video_path.stem}.mp4"
        video_writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

        rows = []
        frame_idx = 0
        base_ts = self.parse_start_timestamp(video_path.stem)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            current_row = {"timestamp": base_ts + (frame_idx / fps)}
            for side in ["l", "r"]:
                for i in range(21):
                    for ax in ["x", "y", "z"]:
                        current_row[f"{side}_{ax}_{i}"] = 0.0

            if results.multi_hand_landmarks:
                for idx, landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label
                    p = "l" if handedness == "Left" else "r"
                    for i, lm in enumerate(landmarks.landmark):
                        (
                            current_row[f"{p}_x_{i}"],
                            current_row[f"{p}_y_{i}"],
                            current_row[f"{p}_z_{i}"],
                        ) = (lm.x, lm.y, lm.z)
                    self.mp_drawing.draw_landmarks(
                        frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )

            rows.append(current_row)
            video_writer.write(frame)
            frame_idx += 1

        cap.release()
        video_writer.release()
        pd.DataFrame(rows).to_csv(
            self.csv_dir / f"{video_path.stem}.csv", index=False, sep=";"
        )

        self.is_processing = False
        return True

    def process_latest_video(self) -> bool:
        """
        Sucht das neueste Video, verarbeitet es und löscht das Original.

        Returns:
            bool: True wenn erfolgreich verarbeitet und gelöscht, False sonst.
        """
        videos = sorted(self.video_dir.glob("recording_*.mp4"))
        if not videos:
            logger.warning(f"Keine Videos in {self.video_dir} gefunden.")
            return False

        latest = videos[-1]
        logger.info(f"Verarbeite neuestes Video: {latest.name}")
        success = self.process_video(latest)

        if success:
            try:
                latest.unlink()  # Lösche Raw Video nach Verarbeitung
                logger.info(f"Originalvideo gelöscht: {latest.name}")
            except Exception as e:
                logger.error(f"Fehler beim Löschen des Videos: {e}")
        return success

    def get_status(self) -> ProcessorState:
        """
        Gibt den aktuellen Verarbeitungsstatus zurück.

        Returns:
            ProcessorState: Status mit Informationen zur Verarbeitung.
        """
        return ProcessorState(
            status=(
                ProcessorStatus.IDLE
                if not self.is_processing
                else ProcessorStatus.PROCESSING
            ),
            is_processing=self.is_processing,
            current_file=self.current_file,
            progress=0.0,
            message="Bereit",
        )
