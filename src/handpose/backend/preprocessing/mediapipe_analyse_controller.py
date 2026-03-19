import logging
from pathlib import Path

from handpose.config_loader import get_settings

from . import mediapipe_analyse
from .mediapipe_analyse import ProcessorState

logger = logging.getLogger(__name__)


class MediaPipeAnalyseController:
    """
    Controller für MediaPipe-Videoanalyse.
    """

    def __init__(self, mode: str = "prod", preview: bool = False) -> None:
        """
        Initialisiert den MediaPipeAnalyseController.

        Args:
            mode (str): Modus "train" oder "prod". Standard: "prod".
            preview (bool): Zeige Preview während der Verarbeitung. Standard: False.
        """
        # Pfade aus Konfiguration laden
        settings = get_settings()
        if mode == "train":
            v_dir = Path(settings.pipeline.camera.recording_dir_training)
            c_dir = Path(settings.pipeline.preprocessing.keypoints_dir_training)
        else:
            v_dir = Path(settings.pipeline.camera.recording_dir_production)
            c_dir = Path(settings.pipeline.preprocessing.keypoints_dir_production)

        logger.info(f"MediaPipeAnalyseController initialisiert für {mode}")
        self.processor = mediapipe_analyse.MediaPipeVideoProcessor(
            video_dir=v_dir, csv_dir=c_dir, preview=preview
        )

    def process_single_video(self, video_filename: str) -> bool:
        """
        Verarbeitet ein einzelnes Video mit MediaPipe.

        Args:
            video_filename (str): Nur Dateiname (z.B. 'recording_20260225_125121.mp4').

        Returns:
            bool: True wenn erfolgreich verarbeitet, False sonst.
        """
        logger.debug(f">>> Verarbeite Video: {video_filename}")
        video_path = self.processor.video_dir / video_filename
        return self.processor.process_video(video_path)

    def process_latest_video(self) -> bool:
        """
        Verarbeitet das neueste Video aus Production-Verzeichnis.

        Returns:
            bool: True wenn erfolgreich verarbeitet, False sonst.
        """
        logger.debug(">>> Verarbeite neuestes Video")
        result = self.processor.process_latest_video()
        if result:
            logger.info("✅ Neuestes Video verarbeitet und gelöscht")
        else:
            logger.error("❌ Fehler beim Verarbeiten des neuesten Videos")
        return result

    def process_all_videos(self) -> dict:
        """
        Verarbeitet alle Videos im Verzeichnis.

        Returns:
            dict: Status-Dictionary mit Success-Flag, Anzahl verarbeiteter Videos und Total.
        """
        logger.debug(">>> Verarbeite alle Videos")
        result = self.processor.process_all_videos()
        if result["success"]:
            logger.info(
                f"✅ {result['processed']}/{result['total']} Videos verarbeitet"
            )
        return result

    def get_status(self) -> ProcessorState:
        """
        Gibt den aktuellen Verarbeitungsstatus zurück.

        Returns:
            ProcessorState: Status mit Informationen zur Verarbeitung.
        """
        return self.processor.get_status()
