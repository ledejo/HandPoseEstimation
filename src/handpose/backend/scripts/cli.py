import argparse
import logging
import sys
from pathlib import Path

from handpose.backend.ML.HMM.bic_analyse import run_bic_analysis
from handpose.backend.backend_controller import BackendController

src_path = Path(__file__).resolve().parent.parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# Setup Logging für das CLI
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    CLI-Einstiegspunkt für zentrale Backend-Steuerung ohne Frontend.

    Unterstützte Befehle:
    - record: Video-Aufnahme mit Kamera
    - preprocess: MediaPipe Analyse auf Videos (single/all/latest)
    - train: ML-Modelle trainieren (HMM, VAE, GMM, all)
    - hmm-bic: HMM-Training mit BIC-State-Suche und BIC-Kurven-Output
    - tune: Hyperparameter-Optimierung für VAE
    - analyze: Komplette ML-Pipeline auf Keypoint-CSV

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="HandPose CLI - Zentrale Steuerung des Backends ohne Frontend.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
--------------------------------------------------------------------------------
SCHNELLÜBERSICHT & BEISPIELE:

  1. RECORD (Aufnahme)
     Optionen: [--cam INDEX] [--fps FPS]
     Beispiel: python cli.py record --fps 60

  2. PREPROCESS (MediaPipe Analyse)
     Optionen: [--mode {all,single,latest}] [--file PFAD_ZUR_MP4]
     Beispiel: python cli.py preprocess --mode single --file video.mp4

  3. TRAIN (Modelle trainieren)
    Optionen: {hmm,vae,gmm,dbscan,all}
     Beispiel: python cli.py train hmm

  4. TUNE (Hyperparameter optimieren)
     Optionen: {vae}
     Beispiel: python cli.py tune vae

    5. HMM-BIC (BIC-Analyse und Training)
      Optionen: [--min-states N] [--max-states N]
      Beispiel: python cli.py hmm-bic --min-states 2 --max-states 30

  6. ANALYZE (Gesamte Pipeline ausführen)
     Optionen: --file PFAD_ZUR_CSV
     Beispiel: python cli.py analyze --file data/02_intermediate/.../test.csv
--------------------------------------------------------------------------------
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Befehle")

    # --- 1. RECORD ---
    record_parser = subparsers.add_parser(
        "record", help="Startet Aufnahme. (Optionen: --cam, --fps)"
    )
    record_parser.add_argument("--cam", type=int, default=None, help="Kamera-Index.")
    record_parser.add_argument("--fps", type=int, default=None, help="Ziel-FPS.")

    # --- 2. PREPROCESS ---
    mp_parser = subparsers.add_parser(
        "preprocess",
        help="MediaPipe Analyse. (Optionen: --mode [all|single|latest], --file)",
    )
    mp_parser.add_argument("--mode", choices=["all", "single", "latest"], default="all")
    mp_parser.add_argument("--file", type=str, help="Dateiname (für mode='single').")

    # --- 3. TRAIN ---
    train_parser = subparsers.add_parser(
        "train",
        help="ML Modelle trainieren. (Erfordert Modellname: hmm, vae, gmm, dbscan, all)",
    )
    train_parser.add_argument("model", choices=["hmm", "vae", "gmm", "dbscan", "all"])

    # --- 4. TUNE ---
    tune_parser = subparsers.add_parser(
        "tune", help="Hyperparameter-Tuning. (Erfordert Modellname: vae)"
    )
    tune_parser.add_argument("model", choices=["vae"])

    # --- 5. HMM-BIC ---
    hmm_bic_parser = subparsers.add_parser(
        "hmm-bic",
        help="HMM mit BIC-State-Suche trainieren und BIC-Kurve speichern.",
    )
    hmm_bic_parser.add_argument(
        "--min-states",
        type=int,
        default=2,
        help="Untere Grenze der State-Suche.",
    )
    hmm_bic_parser.add_argument(
        "--max-states",
        type=int,
        default=30,
        help="Obere Grenze der State-Suche.",
    )

    # --- 6. PIPELINE ---
    pipeline_parser = subparsers.add_parser(
        "analyze", help="ML-Pipeline ausführen. (Erfordert: --file)"
    )
    pipeline_parser.add_argument(
        "--file", type=str, required=True, help="Pfad zur Keypoint-CSV."
    )

    args = parser.parse_args()

    # Hilfe anzeigen, wenn kein Befehl übergeben wurde
    if not args.command:
        parser.print_help()
        return

    # Controller initialisieren (lädt Configs und erstellt Ordner automatisch)
    controller = BackendController()

    # -------------------------------------------------------------------------
    # Command Routing
    # -------------------------------------------------------------------------

    if args.command == "record":
        logger.debug(f"Initialisiere Kamera (Cam: {args.cam}, FPS: {args.fps})...")

        if controller.initialize_recorder(cam_index=args.cam, target_fps=args.fps):
            logger.debug("✅ Kamera initialisiert")
            logger.debug("\n" + "=" * 60)
            logger.info("INTERAKTIVER MODUS GESTARTET (Video-Fenster öffnet sich)")
            logger.debug("=" * 60)
            logger.debug("Bedienung im Video-Fenster:")
            logger.debug("  R = Start/Stop Aufnahme")
            logger.debug("  Q = Beenden")
            logger.debug("=" * 60 + "\n")

            # Hol dir die rohe Instanz und starte die OpenCV-UI-Schleife
            recorder = controller.get_raw_recorder_instance()
            try:
                recorder.run()
            except Exception as e:
                logger.error(f"Fehler während der Aufnahme-Schleife: {e}")

            logger.info("✅ Aufnahme-Session beendet.")
        else:
            logger.error(
                "❌ Kamera konnte nicht initialisiert werden. Prüfe die Verbindung."
            )

    elif args.command == "preprocess":
        if args.mode == "all":
            controller.process_all_videos_mediapipe()
        elif args.mode == "latest":
            controller.run_mediapipe_pipeline()
        elif args.mode == "single":
            if not args.file:
                logger.error(
                    "❌ Für mode='single' muss --file angegeben werden (z.B. --file video.mp4)."
                )
            else:
                controller.process_single_video_mediapipe(args.file)

    elif args.command == "train":
        if args.model in ["hmm", "all"]:
            controller.train_hmm_model()
        if args.model in ["vae", "all"]:
            controller.train_vae_model()
        if args.model in ["gmm", "all"]:
            controller.train_gmm_rois()

    elif args.command == "tune":
        if args.model == "vae":
            controller.tune_vae_hyperparameters()

    elif args.command == "hmm-bic":
        exit_code = run_bic_analysis(
            min_states=args.min_states,
            max_states=args.max_states,
        )
        if exit_code != 0:
            logger.error("❌ HMM-BIC Analyse fehlgeschlagen.")

    elif args.command == "analyze":
        csv_path = Path(args.file)
        if not csv_path.exists():
            logger.error(f"❌ Datei nicht gefunden: {csv_path}")
            return

        logger.info(f"Starte volle Analyse-Pipeline für {csv_path.name}...")
        _, _, _, df_results = controller.process_data(str(csv_path))

        if df_results is not None:
            logger.info("✅ Analyse erfolgreich. Berechne KPIs...")
            metrics = controller.calculate_metrics(df_results)
            logger.info("\n" + "=" * 40)
            logger.info("KPI ERGEBNISSE:")
            logger.info("=" * 40)
            for k, v in metrics.items():
                logger.info(f"{k}: {v}")
            logger.info("=" * 40)
        else:
            logger.error("❌ Fehler während der Datenverarbeitung in process_data().")


if __name__ == "__main__":
    main()
