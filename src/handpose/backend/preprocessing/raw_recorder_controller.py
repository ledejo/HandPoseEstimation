import logging

from handpose.config_loader import get_settings

from .raw_recorder import RawVideoRecorder, RecorderState


logger = logging.getLogger(__name__)


class RawRecorderController:
    """
    Controller-Klasse für die Verwaltung der Rohvideoaufnahme.

    Beispiel:
        controller = RawRecorderController()
        controller.initialize_camera()
        controller.start_recording()
        controller.stop_recording()
    """

    def __init__(
        self, cam_index: int = 0, target_fps: int = 30, video_dir: str | None = None
    ) -> None:
        """
        Initialisiert den RawRecorderController.

        Args:
            cam_index (int): Index der Kamera. Standard: 0.
            target_fps (int): Ziel-FPS für die Aufnahme. Standard: 30.
            video_dir (str | None): Zielverzeichnis für Videos. Falls None, wird aus Konfiguration geladen.
        """
        logger.debug(f"RawRecorderController initialisiert - Target FPS: {target_fps}")
        if video_dir is None:
            video_dir = get_settings().pipeline.camera.recording_dir_production
        self.recorder = RawVideoRecorder(
            cam_index=cam_index, target_fps=target_fps, video_dir=video_dir
        )

    def initialize_camera(self) -> dict:
        """
        Initialisiert die Kamera.

        Returns:
            dict: Status-Dictionary mit Erfolgsflag, Nachricht, Auflösung und FPS.
        """
        logger.debug(">>> Initialisiere Kamera")
        if self.recorder.init_camera():
            result = {
                "success": True,
                "message": "Kamera initialisiert",
                "resolution": (self.recorder.width, self.recorder.height),
                "fps": self.recorder.target_fps,
            }
            logger.info(
                f"✅ Kamera OK: {self.recorder.width}x{self.recorder.height} @ {self.recorder.target_fps} FPS"
            )
            return result
        return {"success": False, "message": "Kamera konnte nicht initialisiert werden"}

    def start_recording(self) -> dict:
        """
        Startet die Aufnahme.

        Returns:
            dict: Status-Dictionary mit Erfolgsflag, Nachricht und Dateinamen.
        """
        logger.debug(">>> Starte Aufnahme")
        result = self.recorder.start_recording()
        if result["success"]:
            logger.info(f"✅ {result['message']}: {result['filename']}")
        return result

    def stop_recording(self) -> dict:
        """
        Beendet die Aufnahme.

        Returns:
            dict: Status-Dictionary mit Erfolgsflag, Nachricht, Frame-Count, Dauer und FPS.
        """
        logger.debug(">>> Beende Aufnahme")
        result = self.recorder.stop_recording()
        if result["success"]:
            logger.info(
                f"✅ Aufnahme beendet: {result['frames']} Frames in {result['duration']}s ({result['actual_fps']} FPS)"
            )
        return result

    def get_status(self) -> RecorderState:
        """
        Gibt den aktuellen Status der Aufnahme zurück.

        Returns:
            RecorderState: Struktur mit Statusdaten.
        """
        return self.recorder.get_status()

    def list_recordings(self) -> dict:
        """
        Listet alle verfügbaren Videos auf.

        Returns:
            dict: Status-Dictionary mit Erfolgsflag, Nachricht und Anzahl der Videos.
        """
        logger.debug(">>> Listet Aufnahmen auf")
        result = self.recorder.list_recordings()
        if result["success"]:
            logger.info(f"✅ {result['count']} Videos gefunden")
        return result

    def run_interactive(self) -> None:
        """
        Startet den interaktiven Modus für Live-Aufnahme.
        """
        logger.debug(">>> Starte interaktiven Modus")
        self.recorder.run()


if __name__ == "__main__":
    controller = RawRecorderController(target_fps=30)
    controller.initialize_camera()
    controller.run_interactive()
