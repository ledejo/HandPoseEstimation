"""
Process Videos Script - MediaPipe Video-Analyse via Konsole
Nutzt den BackendController für MediaPipe-Verarbeitung

Beispiele:
    # Alle Videos analysieren (überspringt bereits verarbeitete)
    python process_videos_script.py --all

    # Alle Videos ERNEUT analysieren
    python process_videos_script.py --all-force

    # Einzelnes Video analysieren
    python process_videos_script.py --file recording_20260225_125121.mp4

    # Neuestes Video analysieren (Production: löscht Original nach Erfolg)
    python process_videos_script.py --latest
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
    CLI-Einstiegspunkt für MediaPipe Video-Analyse.

    Unterstützte Modi:
    - --file: Einzelnes Video verarbeiten
    - --all: Alle Videos verarbeiten (überspringt bereits verarbeitete)
    - --all-force: Alle Videos erneut verarbeiten
    - --latest: Neuestes Video verarbeiten und Original löschen

    Returns:
        int: Exit-Code (0=Erfolg, 1=Fehler).
    """
    parser = argparse.ArgumentParser(
        description="Process Videos - MediaPipe Video-Analyse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python process_videos_script.py --all              (überspringt bereits verarbeitete)
  python process_videos_script.py --all-force        (verarbeitet ALLES erneut)
  python process_videos_script.py --file VIDEO.mp4   (einzelne Datei)
  python process_videos_script.py --latest           (neuestes Video, Production-Modus)
        """,
    )

    parser.add_argument("--file", type=str, help="Einzelnes Video verarbeiten")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Alle Videos verarbeiten (überspringt bereits verarbeitete)",
    )
    parser.add_argument(
        "--all-force",
        action="store_true",
        help="Alle Videos erneut verarbeiten (auch bereits verarbeitete)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Neuestes Video verarbeiten + löschen (Production)",
    )

    args = parser.parse_args()

    logger.info("Initialisiere BackendController...")

    try:
        controller = BackendController()

        # ===== FILE MODE =====
        if args.file:
            logger.info(f"📹 Verarbeite einzelnes Video: {args.file}")
            success = controller.process_single_video_mediapipe(args.file)
            if success:
                logger.info("✅ Video verarbeitet")
                return 0
            else:
                logger.error("❌ Fehler beim Verarbeiten")
                return 1

        # ===== ALL MODE =====
        elif args.all or args.all_force:
            if args.all_force:
                logger.info("📹 Force-Mode: Verarbeite ALLE Videos erneut...")
            else:
                logger.info(
                    "📹 Verarbeite ALLE Videos (überspringt bereits verarbeitete)..."
                )

            result = controller.process_all_videos_mediapipe()

            if result.get("success"):
                logger.info(
                    f"✅ Abgeschlossen: {result['processed']}/{result['total']} verarbeitet, "
                    f"{result['failed']} Fehler"
                )
                return 0 if result["failed"] == 0 else 1
            else:
                logger.error(f"❌ Fehler: {result.get('message')}")
                return 1

        # ===== LATEST MODE =====
        elif args.latest:
            logger.info("📹 Production-Modus: Analysiere neuestes Video...")
            success = controller.run_mediapipe_pipeline()
            if success:
                logger.info("✅ Neuestes Video analysiert")
                return 0
            else:
                logger.error("❌ Fehler beim Analysieren")
                return 1

        # ===== HELP =====
        else:
            parser.print_help()
            return 0

    except Exception as e:
        logger.error(f"❌ Fehler: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
