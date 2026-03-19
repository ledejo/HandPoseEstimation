import logging
from pathlib import Path
from typing import Any

import pandas as pd

from handpose.config_loader import get_settings

from .analysis_hmm import KPICalculator, analyze_frames_and_structure
from .features_hmm import HmmFeatureExtractor
from .model_hmm import HmmWrapper
from .predict_hmm import HmmPredictor
from .train_hmm import HmmTrainer
from .visualize_hmm import show_hmm_result

logger = logging.getLogger(__name__)


class HmmController:
    def __init__(self) -> None:
        """
        Initialisierung des HMM-Controllers.
        """
        hmm_cfg = get_settings().hmm
        self.trainer = HmmTrainer()
        self.extractor = HmmFeatureExtractor()
        self.kpi_calc = KPICalculator(
            grasp_state=hmm_cfg.grasp_state,
            rois=hmm_cfg.rois,
            roi_rad_x=hmm_cfg.roi_rad_x,
            roi_rad_y=hmm_cfg.roi_rad_y,
            min_y=hmm_cfg.min_y_for_grasp,
        )
        self.predictor = HmmPredictor()
        self.model_wrapper = HmmWrapper()
        self.is_loaded = False
        logger.debug("HMM Controller initialisiert")

    def train(self) -> None:
        """
        Führt das Training des HMM-Modells durch.
        """
        logger.debug("Starte Training des HMM Modells...")
        self.trainer.run_training()
        self.is_loaded = False

    def train_with_bic(
        self, min_s: int = 2, max_s: int = 30
    ) -> tuple[list[int], list[float]]:
        """
        Additiver Trainingspfad mit BIC-basierter State-Suche.

        Bestehende train()-Methode bleibt unveraendert.
        """
        logger.info(f"Starte BIC-Training mit State-Suche im Bereich {min_s}-{max_s}.")
        result = self.trainer.run_training_with_bic(min_states=min_s, max_states=max_s)
        self.is_loaded = False
        return result

    def load_model_strict(self) -> None:
        """
        Laedt das Modell ohne automatisches Nachtrainieren.
        """
        self.load_model(auto_train=False)

    def load_model(self, auto_train: bool = True) -> None:
        """
        Lädt trainiertes HMM-Modell und Scaler.

        Args:
            auto_train: Falls True, trainiert automatisch falls Modell fehlt.

        Raises:
            FileNotFoundError: Wenn Modell fehlt und auto_train=False.
        """
        model_dir = Path(get_settings().hmm.paths.get("model_dir"))
        model_path = model_dir / "hmm_model.pkl"
        scaler_path = model_dir / "scaler.pkl"
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.model_wrapper.load(model_path, scaler_path)
            self.is_loaded = True
            logger.debug(f"HMM-Modell geladen: {model_path}")
        except FileNotFoundError:
            if auto_train:
                logger.warning(
                    "Modell nicht gefunden. Starte automatisches Training..."
                )
                self.train()
                self.model_wrapper.load(model_path, scaler_path)
                self.is_loaded = True
                logger.debug("Modell erfolgreich trainiert und geladen.")
            else:
                logger.error(f"Modell nicht gefunden: {model_path}")
                raise

    def analyze_single_file(self, file_path: str) -> pd.DataFrame | None:
        """
        Analysiert einzelne CSV-Datei und berechnet KPIs.

        Args:
            file_path: Pfad zur CSV-Datei mit Keypoint-Daten.

        Returns:
            DataFrame mit KPI-Ergebnissen.

        Raises:
            FileNotFoundError: Wenn Datei nicht existiert.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Datei nicht gefunden: {file_path}")
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        if not self.is_loaded:
            self.load_model()

        df = self.extractor.process_csv(file_path)
        if df is None:
            raise ValueError(f"Keine gültigen Daten in CSV: {file_path}")

        X, _ = self.extractor.calculate_features(df)
        states = self.model_wrapper.predict(X)
        kpi_results = self.kpi_calc.calculate_kpis(states, df)

        return kpi_results

    def analyze_all_files_in_directory(self, dir_path: str) -> pd.DataFrame | None:
        """
        Analysiert alle CSV-Dateien in einem Verzeichnis.

        Args:
            dir_path: Pfad zum Verzeichnis mit CSV-Dateien.

        Returns:
            DataFrame mit KPI-Ergebnissen aller Dateien.

        Raises:
            FileNotFoundError: Wenn Verzeichnis nicht existiert.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            logger.error(f"Verzeichnis nicht gefunden: {dir_path}")
            raise FileNotFoundError(f"Verzeichnis nicht gefunden: {dir_path}")

        all_kpis = []
        # Nutze Path.glob() statt glob.glob() - moderner und robuster
        csv_files = list(dir_path.glob("*.csv"))

        for file_path in csv_files:
            try:
                logger.debug(f"Analysiere: {file_path}")
                kpi_results = self.analyze_single_file(file_path)
                kpi_results["file_name"] = Path(file_path).name
                all_kpis.append(kpi_results)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Überspringen {file_path}: {e}")

        if not all_kpis:
            raise ValueError(f"Keine CSV-Dateien in {dir_path} verarbeitet")

        return pd.concat(all_kpis, ignore_index=True)

    def show_result_for_single_video(self, file_path: str) -> Any | None:
        """
        Zeigt HMM-Analyse-Ergebnisse als Plot.

        Args:
            file_path: Pfad zur CSV-Datei.

        Raises:
            FileNotFoundError: Wenn Datei nicht existiert.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Datei nicht gefunden: {file_path}")
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        if not self.is_loaded:
            self.load_model()

        return show_hmm_result(file_path)

    def get_detailed_analysis_for_one_csv(self, file_path: str) -> pd.DataFrame | None:
        """
        Führt detaillierte Frame-basierte Analyse durch.

        Args:
            file_path: Pfad zur CSV-Datei.

        Returns:
            DataFrame mit detaillierten Frame-Informationen.

        Raises:
            FileNotFoundError: Wenn Datei nicht existiert.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Datei nicht gefunden: {file_path}")
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        if not self.is_loaded:
            self.load_model()

        states, coords, _ = self.predictor.predict(file_path)
        if states is None:
            raise ValueError(f"HMM-Vorhersage fehlgeschlagen für {file_path}")

        df_analysis = analyze_frames_and_structure(states, coords, file_path)
        return df_analysis
