import logging
from pathlib import Path
from typing import Any

import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd

from handpose.config_loader import Settings, get_settings

from .ML.GMM.controller_gmm import GmmController
from .ML.HMM.controller_hmm import HmmController
from .ML.VAE.controller_vae import VaeController
from .postprocessing.metrics_postprocessing import KPICalculator
from .postprocessing.video_postprocessing import extract_all_frames, extract_frames
from .preprocessing.mediapipe_analyse_controller import MediaPipeAnalyseController
from .preprocessing.raw_recorder_controller import RawRecorderController

logger = logging.getLogger(__name__)


class BackendController:
    """Facade für Aufnahme, Preprocessing und ML-Analyse-Pipelines."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialisiert den Backend-Controller mit allen Sub-Komponenten.

        Args:
            settings: Settings-Instanz. Falls None, wird get_settings() verwendet.
        """
        # Settings per Dependency Injection
        if settings is None:
            settings = get_settings()
        self.settings = settings

        # Verzeichnisse erstellen (einmalig bei Initialisierung)
        from handpose import config_loader as cfg

        cfg._create_required_directories()

        default_cam_index = settings.pipeline.camera.camera_type
        target_fps = settings.pipeline.camera.target_fps

        # Recording & Preprocessing Controller
        self.recorder_controller = RawRecorderController(
            cam_index=default_cam_index, target_fps=target_fps
        )
        self.mediapipe_controller = MediaPipeAnalyseController(preview=False)

        # ML Controller
        self.vae_controller = VaeController()
        self.hmm_controller = HmmController()
        self.gmm_controller = GmmController()

        # Postprocessing controller
        self.kpi_calculator = KPICalculator(fps=target_fps)

        # ArUco Kalibrierung aus Config laden
        aruco_cfg = settings.pipeline.aruco_calibration
        marker_name = aruco_cfg.marker_dict
        if not marker_name.startswith("DICT_"):
            marker_name = f"DICT_{marker_name}"

        self.aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, marker_name))
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_min_markers = aruco_cfg.min_markers
        self.aruco_output_size = aruco_cfg.output_size
        self.aruco_margin = aruco_cfg.margin

        logger.debug("Backend Controller (Facade) und Sub-Komponenten initialisiert")

    # -------------------------------------------------------------------------
    # 1. RECORDING WRAPPER (Delegation an RawRecorderController)
    # -------------------------------------------------------------------------

    def initialize_recorder(
        self, cam_index: int | None = None, target_fps: int | None = None
    ) -> bool:
        """Initialisiert die Kamera über den RecorderController.

        Args:
            cam_index: Index der Kamera. Falls None, wird der Standardwert verwendet.
            target_fps: Ziel-Framerate. Falls None, wird der Standardwert verwendet.

        Returns:
            True bei erfolgreicher Initialisierung, sonst False.
        """
        if cam_index is not None:
            self.recorder_controller.recorder.cam_index = cam_index
        if target_fps is not None:
            self.recorder_controller.recorder.target_fps = target_fps

        result = self.recorder_controller.initialize_camera()
        return result.get("success", False)

    def start_recording(self) -> None:
        """Startet die Videoaufnahme."""
        self.recorder_controller.start_recording()

    def stop_recording(self) -> str | None:
        """Stoppt die laufende Videoaufnahme.

        Returns:
            Dateiname des aufgenommenen Videos oder None bei Fehler.
        """
        result = self.recorder_controller.stop_recording()
        if result.get("success"):
            return result.get("filename")
        return None

    def get_raw_recorder_instance(self) -> Any:
        """Gibt die rohe Recorder-Instanz für Frontend-Preview zurück."""
        return self.recorder_controller.recorder

    def is_recording(self) -> bool:
        """Prüft, ob aktuell eine Aufnahme läuft.

        Returns:
            True wenn eine Aufnahme aktiv ist, sonst False.
        """
        return self.recorder_controller.recorder.is_recording

    def get_current_filename(self) -> str:
        """Gibt den aktuellen Dateinamen der Aufnahme zurück.

        Returns:
            Dateiname oder "-" wenn keine Aufnahme aktiv ist.
        """
        return getattr(self.recorder_controller.recorder, "current_filename", "-")

    def align_camera_with_aruco(self, frame: np.ndarray) -> dict[str, Any]:
        """Kalibriert Kamera mittels ArUco-Markererkennung für Perspektivnormalisierung.

        Erkennt 4 ArUco-Marker und berechnet Perspektivtransformation (Homographie)
        zur geometrischen Normalisierung. Sortiert Marker geometrisch zur Vermeidung
        von Überkreuzungen. Speichert Matrix intern im Recorder für nachfolgende Frames.
        """
        if frame is None:
            return {"success": False, "message": "Kein Bild übergeben."}

        # 1. Bild in Graustufen umwandeln für robustere Erkennung
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logger.warning(f"Konvertierungsfehler zu Graustufen: {e}")
            gray_frame = frame

        # 2. Marker suchen (Normal und Gespiegelt testen)
        try:
            detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(gray_frame)

            is_flipped = False
            # Wenn normal nichts gefunden wird, testen wir gespiegelt
            if ids is None or len(ids) < self.aruco_min_markers:
                gray_frame_flipped = cv2.flip(gray_frame, 1)
                corners_f, ids_f, _ = detector.detectMarkers(gray_frame_flipped)
                if ids_f is not None and len(ids_f) >= self.aruco_min_markers:
                    corners = corners_f
                    ids = ids_f
                    is_flipped = True

        except Exception as e:
            logger.error(f"OpenCV ArUco Erkennungsfehler: {e}")
            return {
                "success": False,
                "message": "Interner Fehler bei Marker-Erkennung.",
            }

        # 3. Validierung: Mindestens 4 Marker erforderlich
        if ids is None or len(ids) < self.aruco_min_markers:
            logger.warning("Nicht genügend ArUco Marker gefunden.")
            return {
                "success": False,
                "message": "Zu wenige Marker gefunden. Bitte Spiegelungen auf dem Display vermeiden!",
            }

        # 4. Mittelpunkte der ersten 4 gefundenen Marker berechnen
        centers = []
        for i in range(4):
            marker_corners = corners[i][0]
            cx = float(marker_corners[:, 0].mean())
            cy = float(marker_corners[:, 1].mean())

            # Falls gespiegelt: X-Koordinate auf Originalbild umrechnen
            if is_flipped:
                height, width = gray_frame.shape
                cx = width - cx

            centers.append((cx, cy))

        # 5. Geometrische Sortierung zur Vermeidung von Überkreuzungen
        # A) Nach Y-Koordinate sortieren (trennt obere/untere Marker)
        centers = sorted(centers, key=lambda p: p[1])
        top_two = centers[:2]
        bottom_two = centers[2:]

        # B) Obere zwei nach X-Koordinate sortieren
        top_two = sorted(top_two, key=lambda p: p[0])
        tl = top_two[0]  # Top-Left
        tr = top_two[1]  # Top-Right

        # C) Untere zwei nach X-Koordinate sortieren
        bottom_two = sorted(bottom_two, key=lambda p: p[0])
        bl = bottom_two[0]  # Bottom-Left
        br = bottom_two[1]  # Bottom-Right

        # 6. Perspektivtransformation an den Recorder übergeben
        self.recorder_controller.recorder.set_top_down_perspective(
            top_left=tl,
            top_right=tr,
            bottom_right=br,
            bottom_left=bl,
            margin=self.aruco_margin,
        )

        gefunden_ids = ids.flatten().tolist()[:4]
        logger.info(
            f"✅ Kamera erfolgreich geometrisch ausgerichtet (IDs: {gefunden_ids})"
        )

        return {
            "success": True,
            "message": "Kamera erfolgreich ausgerichtet.",
            "detected_ids": gefunden_ids,
        }

    def reset_alignment(self) -> None:
        """Setzt Perspektivtransformation zurück."""
        self.recorder_controller.recorder.perspective_matrix = None
        logger.debug("Perspektivtransformation zurückgesetzt")

    # -------------------------------------------------------------------------
    # 2. PREPROCESSING WRAPPER (MediaPipe)
    # -------------------------------------------------------------------------

    def run_mediapipe_pipeline(self) -> bool:
        """Führt MediaPipe-Keypoint-Extraktion auf das letzte Video durch."""
        logger.debug("Starte MediaPipe Pipeline via BackendController...")
        success = self.mediapipe_controller.process_latest_video()

        if success:
            logger.info("MediaPipe Analyse erfolgreich abgeschlossen.")
        else:
            logger.warning("MediaPipe Analyse fehlgeschlagen oder Video-Probleme.")

        return success

    def process_single_video_mediapipe(self, video_filename: str) -> bool:
        """Extrahiert Keypoints aus einem spezifischen Video."""

        logger.debug(f"Analysiere einzelnes Video: {video_filename}")
        success = self.mediapipe_controller.process_single_video(video_filename)

        if success:
            logger.info(f"✅ Video analysiert: {video_filename}")
        else:
            logger.error(f"❌ Fehler bei der Analyse: {video_filename}")

        return success

    def process_all_videos_mediapipe(self) -> dict[str, Any]:
        """Batch-Analyse aller Videos mit MediaPipe (mit Duplikat-Übersprung)."""

        logger.debug("Analysiere ALLE Videos mit MediaPipe...")
        result = self.mediapipe_controller.process_all_videos()

        if result.get("success"):
            logger.info(
                f"✅ Batch-Analyse abgeschlossen: {result['processed']}/{result['total']} verarbeitet"
            )
        else:
            logger.error("❌ Fehler bei Batch-Analyse")

        return result

    def check_hands_in_latest_csv(self, csv_dir_path: Path) -> bool:
        """Prüft heuristisch ob die neueste CSV Hand-Keypoints enthält."""
        try:
            csvs = sorted(csv_dir_path.glob("recording_*.csv"))
            if not csvs:
                return False

            latest_csv = csvs[-1]
            df = pd.read_csv(latest_csv, sep=";")

            has_right = any("r_" in col for col in df.columns)
            has_left = any("l_" in col for col in df.columns)

            return has_right or has_left
        except Exception as e:
            logger.error(f"Hand-Check Fehler: {e}")
            return False

    # -------------------------------------------------------------------------
    # 3. ML PIPELINE (HMM, VAE, GMM)
    # -------------------------------------------------------------------------

    def train_hmm_model(self) -> None:
        """Trainiert HMM für Phasenerkennung."""
        logger.debug("Starte HMM-Training...")
        self.hmm_controller.train()
        logger.info("✅ HMM-Training abgeschlossen")

    def train_vae_model(self) -> None:
        """Trainiert VAE zur Anomalieerkennung."""
        logger.debug("Starte VAE-Training...")
        self.vae_controller.train()
        logger.info("VAE-Training abgeschlossen.")

    def train_gmm_rois(self) -> tuple[Any, Any, Any] | None:
        """Trainiert GMM-Clustering auf Griffposen (ROI-Extraktion)."""
        logger.debug("Starte GMM ROI-Training...")
        result = self.gmm_controller.train_rois()
        if result:
            logger.info("GMM ROI-Training abgeschlossen.")
        return result

    def train_dbscan_rois(self) -> tuple[Any, Any, Any] | None:
        """Kompatibilitaetsalias: routed DBSCAN-Aufruf auf GMM-ROI-Training."""
        return self.train_gmm_rois()

    def tune_vae_hyperparameters(self) -> None:
        """Führt Hyperparameter-Tuning für VAE mit Optuna durch."""
        logger.debug("Starte VAE Hyperparameter-Tuning...")
        self.vae_controller.tune_hyperparameter()
        logger.info("VAE Hyperparameter-Tuning abgeschlossen.")

    def process_data(
        self, path_to_keypoint_csv: str
    ) -> tuple[Any | None, Any | None, Any | None, pd.DataFrame | None]:
        """Orchestriert die ML-Analyse-Pipeline (VAE, HMM, DBSCAN) gemäß Konfiguration."""
        try:
            # Settings-Zugriffe für diese Methode
            is_anomaly_col = self.settings.pipeline.ml_pipeline.column_names[
                "is_anomaly_column"
            ]
            frames_col = self.settings.pipeline.ml_pipeline.column_names[
                "frames_column"
            ]
            frame_col = self.settings.pipeline.ml_pipeline.column_names["frame_column"]
            csv_ext = self.settings.pipeline.ml_pipeline.file_formats["csv_extension"]
            output_dir = Path(self.settings.pipeline.postprocessing.output_csv_dir)

            # Modell-Aktivierungsstatus aus Konfiguration
            enable_vae = self.settings.pipeline.ml_pipeline.models.get(
                "enable_vae", False
            )
            enable_hmm = self.settings.pipeline.ml_pipeline.models.get(
                "enable_hmm", False
            )
            enable_gmm = self.settings.pipeline.ml_pipeline.models.get(
                "enable_gmm",
                self.settings.pipeline.ml_pipeline.models.get("enable_dbscan", False),
            )

            plot_vae_anomalies = None
            plot_hmm = None
            plot_gmm = None
            df_anomalies = None
            df_hmm_analysis = None
            df_combined_analysis = None

            # 1. VAE (Anomalieerkennung)
            if enable_vae:
                logger.info("VAE ist aktiviert, starte Anomalieerkennung...")
                feature_file_path, df_features = (
                    self.vae_controller.transform_keypoints_to_features(
                        path_to_keypoint_csv, save_path=None
                    )
                )

                vae_result = self.vae_controller.predict_anomalies(df_features)
                if vae_result is None:
                    logger.error("VAE Anomalieerkennung fehlgeschlagen")
                    return None, None, None, None

                plot_vae_anomalies, anomalies_array = vae_result
                df_anomalies = pd.DataFrame({is_anomaly_col: anomalies_array})
                df_anomalies.insert(0, frames_col, range(1, 1 + len(df_anomalies)))
                logger.info("✅ VAE Anomalieerkennung abgeschlossen.")
            else:
                logger.info("VAE ist deaktiviert (enable_vae: false)")

            # 2. HMM (Phasen/Zustände)
            if enable_hmm:
                logger.info("HMM ist aktiviert, starte Phasenanalyse...")
                df_hmm_analysis = self.hmm_controller.get_detailed_analysis_for_one_csv(
                    path_to_keypoint_csv
                )
                plot_hmm = self.hmm_controller.show_result_for_single_video(
                    path_to_keypoint_csv
                )

                if df_hmm_analysis is None:
                    raise ValueError(
                        "HMM-Analyse fehlgeschlagen. Modell evtl. nicht trainiert."
                    )
                logger.info("✅ HMM Analyse abgeschlossen.")
            else:
                logger.info("HMM ist deaktiviert (enable_hmm: false)")

            # 3. GMM (Orts-Cluster)
            if enable_gmm:
                logger.info("GMM ist aktiviert, starte räumliche Clusteranalyse...")
                X, labels, centers = self.gmm_controller.analyze_single_file(
                    path_to_keypoint_csv
                )
                plot_gmm = self.gmm_controller.show_results(X, labels, centers)
                logger.info("✅ GMM Analyse abgeschlossen.")
            else:
                logger.info("GMM ist deaktiviert (enable_gmm/enable_dbscan: false)")

            # 4. Ergebnisse zusammenführen (nur wenn mindestens ein Modell aktiviert ist)
            if enable_vae or enable_hmm or enable_gmm:
                # Starte mit HMM falls aktiviert
                if enable_hmm:
                    df_combined_analysis = df_hmm_analysis.copy()
                elif enable_vae:
                    # Falls nur VAE aktiviert, starte damit
                    df_combined_analysis = df_anomalies.copy()
                else:
                    # Falls nur GMM, erstelle leere DataFrame (GMM hat keine Feature-Spalte wie HMM)
                    logger.info(
                        "Nur GMM aktiviert - begrenzte Ergebnis-Zusammenführung"
                    )

                # Merge mit VAE-Ergebnissen (falls HMM und VAE beide aktiviert)
                if enable_vae and enable_hmm and df_combined_analysis is not None:
                    df_combined_analysis = pd.merge(
                        df_combined_analysis,
                        df_anomalies,
                        left_on=frame_col,
                        right_on=frames_col,
                        how="left",
                    )
                    if frames_col in df_combined_analysis.columns:
                        df_combined_analysis = df_combined_analysis.drop(
                            columns=[frames_col]
                        )

                # Speichere Ergebnisse
                if df_combined_analysis is not None:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_filename = Path(path_to_keypoint_csv).stem + csv_ext
                    output_path = output_dir / output_filename
                    df_combined_analysis.to_csv(output_path, index=False)
                    logger.info(f"✅ Analyse-Ergebnisse gespeichert: {output_path}")
            else:
                logger.warning(
                    "⚠️  Keine Modelle aktiviert (enable_vae, enable_hmm, enable_gmm/enable_dbscan alle false)"
                )

            plots_dir = Path(self.settings.pipeline.postprocessing.plots_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)

            run_id = Path(path_to_keypoint_csv).stem
            plot_filenames = self.settings.pipeline.postprocessing.plot_filenames

            if plot_vae_anomalies is not None:
                filename = plot_filenames.get("vae", "vae_{run_name}.jpg").replace(
                    "{run_name}", run_id
                )
                plot_vae_anomalies.savefig(plots_dir / filename, bbox_inches="tight")

            if plot_hmm is not None:
                filename = plot_filenames.get("hmm", "hmm_{run_name}.jpg").replace(
                    "{run_name}", run_id
                )
                plot_hmm.savefig(plots_dir / filename, bbox_inches="tight")

            if plot_gmm is not None:
                filename = plot_filenames.get(
                    "gmm", plot_filenames.get("dbscan", "gmm_{run_name}.jpg")
                ).replace("{run_name}", run_id)
                plot_gmm.savefig(plots_dir / filename, bbox_inches="tight")

            return plot_vae_anomalies, plot_hmm, plot_gmm, df_combined_analysis

        except Exception as e:
            logger.exception(f"Kritischer Fehler in process_data: {e}")
            return None, None, None, None

    def analyze_single_file_hmm(self, file_path: str) -> pd.DataFrame | None:
        """Führt HMM-KPI-Analyse auf einer einzelnen Datei aus.

        Args:
            file_path: Pfad zur Keypoint-CSV-Datei.

        Returns:
            DataFrame mit HMM-Analyse oder None bei Fehler.
        """
        return self.hmm_controller.analyze_single_file(file_path)

    def analyze_directory_hmm(self, dir_path: str) -> pd.DataFrame | None:
        """Führt HMM-Analyse auf allen Dateien in einem Verzeichnis aus.

        Args:
            dir_path: Pfad zum Verzeichnis mit Keypoint-CSV-Dateien.

        Returns:
            Aggregiertes DataFrame mit allen Analysen oder None bei Fehler.
        """
        return self.hmm_controller.analyze_all_files_in_directory(dir_path)

    # -------------------------------------------------------------------------
    # 4. POSTPROCESSING (KPIs & Video Export)
    # -------------------------------------------------------------------------

    def calculate_metrics(self, df_analysis: pd.DataFrame) -> dict[str, Any]:
        """Berechnet KPIs aus ML-Analyseergebnissen basierend auf aktivierten Modellen."""
        if df_analysis is None or df_analysis.empty:
            logger.warning("Keine Daten für KPI-Berechnung vorhanden.")
            return {}

        # Modell-Aktivierungsstatus aus Konfiguration
        enable_vae = self.settings.pipeline.ml_pipeline.models.get("enable_vae", False)
        enable_hmm = self.settings.pipeline.ml_pipeline.models.get("enable_hmm", False)

        metrics = {}

        # HMM-basierte KPIs (Zeiten und Teile)
        if enable_hmm:
            metrics["total_duration"] = self.kpi_calculator.calculate_overall_time(
                df_analysis
            )
            metrics["assembly_time_net"] = (
                self.kpi_calculator.calculate_assembly_time_without_state(df_analysis)
            )
            metrics["parts_summary"] = self.kpi_calculator.calculate_kpi_summary(
                df_analysis
            )
        else:
            logger.info(
                "HMM ist deaktiviert - HMM-basierte KPIs (Zeiten, Teile) werden nicht berechnet"
            )

        # VAE-basierte KPIs (Anomalien)
        if enable_vae:
            metrics["anomaly_count"] = (
                self.kpi_calculator.calculate_amount_of_anomalies(df_analysis)
            )
        else:
            logger.info("VAE ist deaktiviert - Anomalie-Zählung wird nicht berechnet")

        # DBSCAN wird in parts_summary (HMM) integriert, daher keine separate KPI

        return metrics

    def extract_best_frames_for_video(self, video_path: str, output_dir: str) -> int:
        """Extrahiert qualitätsgeprüfte Frames aus einem Video."""
        try:
            count = extract_frames(video_path, output_dir)
            logger.info(f"{count} Best Frames extrahiert nach {output_dir}")
            return count
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren der Frames: {e}")
            return 0

    def extract_all_frames_from_video(
        self, video_path: str, output_dir: str | None = None
    ) -> int:
        """Extrahiert alle Frames eines Videos als PNG-Bilder."""
        try:
            count = extract_all_frames(video_path, output_dir)
            logger.info(
                f"{count} Frames extrahiert nach {output_dir or 'Default-Pfad'}"
            )
            return count
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren aller Frames: {e}")
            return 0

    def get_results_for_run(self, run_id: str) -> pd.DataFrame | None:
        """Lädt die KPI-Ergebnisse für einen Recording-Run."""
        output_dir = Path(self.settings.pipeline.postprocessing.output_csv_dir)
        csv_path = output_dir / f"{run_id}.csv"
        if not csv_path.exists():
            logger.warning(f"KPI-CSV nicht gefunden: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            logger.error(f"Fehler beim Laden der KPI-CSV ({csv_path}): {e}")
            return None
