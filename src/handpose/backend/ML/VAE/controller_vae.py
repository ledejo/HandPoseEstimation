import logging

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from handpose.config_loader import get_settings

from . import predict_vae, train_vae, tuning_vae
from .features_vae import FeatureExtractionController

logger = logging.getLogger(__name__)


class VaeController:
    def __init__(self) -> None:
        """
        Initialisierung des VAE Controllers.
        """
        device = get_settings().vae.actual_device
        self.inference_engine = predict_vae.VaeInference()
        self.feature_extractor = FeatureExtractionController()
        logger.debug(f"VAE Controller initialisiert auf: {device}")

    def train(self) -> None:
        """
        Führt das Training des VAE-Modells durch.
        """
        logger.debug("Starte Training des VAE Modells...")
        train_vae.train()

    def tune_hyperparameter(self) -> None:
        """
        Führt das Hyperparameter-Tuning mit Optuna durch.
        """
        logger.debug("Starte Hyperparameter-Tuning für das VAE Modell...")
        tuning_vae.run_optuna_study()

    def predict_anomalies(
        self, df_features: pd.DataFrame
    ) -> tuple[Figure, np.ndarray] | None:
        """
        Führt die Anomalieerkennung auf einer Datei durch.

        Args:
            df_features: DataFrame mit den Feature-Daten.

        Returns:
            Tuple of (plot_figure, anomalies_array) or None if prediction fails.
        """

        logger.info("Starte Inference mit dem VAE Modell...")

        result = self.inference_engine.predict_anomalies(df_features)
        if result is None:
            return None

        plot_VAE_Anomalies, anomalies_array = result
        return plot_VAE_Anomalies, anomalies_array

    def transform_keypoints_to_features(
        self, path_to_keypoints: str, save_path: str | None
    ) -> tuple[str | None, pd.DataFrame | None]:
        """
        Transformiert Keypoint-Daten in Feature-Daten.

        Args:
            path_to_keypoints (str): Pfad zu den Keypoint-Daten im CSV-Format.
            save_path (str): Pfad zum Speichern der extrahierten Features.

        Returns:
            Tuple of (feature_file_path, features_df)
        """
        result = self.feature_extractor.process_keypoints(path_to_keypoints, save_path)

        return result
