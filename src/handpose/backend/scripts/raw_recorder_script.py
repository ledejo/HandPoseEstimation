"""
Raw Recorder Script - Manuelle Kamera-Steuerung via Konsole
Nutzt den BackendController zur Kamera-Aufnahme

Beispiele:
    # Kamera initialisieren & interaktiven Modus starten
    python raw_recorder_script.py

    # Mit spezifischem Kamera-Index
    python raw_recorder_script.py --cam 0

    # Mit spezifische FPS
    python raw_recorder_script.py --cam 0 --fps 30
"""

import argparse
import logging
import sys
from pathlib import Path

from handpose.backend.backend_controller import BackendController

# Pfad adjustieren
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    """
    CLI-Einstiegspunkt für manuelle Kamera-Steuerung.

    Initialisiert die Kamera und startet interaktiven Modus für Video-Aufnahme.
    Tastatursteuerung im Video-Fenster:
    - R = Start/Stop Aufnahme
    - Q = Beenden

    Returns:
        int: Exit-Code (0=Erfolg, 1=Fehler).
    """
    parser = argparse.ArgumentParser(
        description="Raw Recorder - Manuelle Kamera-Steuerung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Bedienung im interaktiven Modus:
                R = Start/Stop Aufnahme
                Q = Beenden

                Beispiele:
                python raw_recorder_script.py                (Default: cam 0, 30 FPS)
                python raw_recorder_script.py --cam 1 --fps 60
        """,
    )

    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="Kamera-Index (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS (default: 30)",
    )

    args = parser.parse_args()

    logger.info(f"Initialisiere Recorder: Kamera={args.cam}, FPS={args.fps}")

    try:
        # Erstelle Backend Controller
        controller = BackendController()

        # Initialisiere Kamera
        success = controller.initialize_recorder(
            cam_index=args.cam, target_fps=args.fps
        )
        if not success:
            logger.error("❌ Kamera konnte nicht initialisiert werden")
            return 1

        logger.info("✅ Kamera initialisiert")
        logger.info("\n" + "=" * 60)
        logger.info("INTERAKTIVER MODUS GESTARTET")
        logger.info("=" * 60)
        logger.info("Bedienung:")
        logger.info("  R = Start/Stop Aufnahme")
        logger.info("  Q = Beenden")
        logger.info("=" * 60 + "\n")

        # Nutze die Recorder-Instanz für die Hauptschleife
        # (OpenCV-Objekte sind nicht serialisierbar)
        recorder = controller.get_raw_recorder_instance()
        recorder.run()

        logger.info("✅ Recorder beendet")
        return 0

    except Exception as e:
        logger.error(f"❌ Fehler: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
