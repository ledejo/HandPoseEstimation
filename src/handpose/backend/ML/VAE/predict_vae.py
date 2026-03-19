import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from handpose.config_loader import get_settings

from .model_vae import HandPoseVae

logger = logging.getLogger(__name__)


class VaeInference:
    def __init__(self, device: str | None = None):
        """
        Initialisierung des VAE Inference Engines mit DI für device.

        Args:
            device: PyTorch Device ('cpu' oder 'cuda'). Default: aus Config.
        """
        vae_cfg = get_settings().vae
        self.device = device if device is not None else vae_cfg.actual_device

        self.model = HandPoseVae(
            input_dim=vae_cfg.input_dim,
            hidden_dim=vae_cfg.hidden_dim,
            latent_dim=vae_cfg.latent_dim,
            seq_len=vae_cfg.window_size,
        ).to(self.device)

        try:
            self.model.load_state_dict(
                torch.load(vae_cfg.model_save_path, map_location=self.device)
            )
            logger.debug("VAE-Modell erfolgreich geladen!")
        except Exception as e:
            logger.error(f"Modell-Ladefehler: {e}")

    def predict_anomalies(
        self,
        df_features: pd.DataFrame,
        window_size: int | None = None,
        threshold: float | None = None,
    ):
        """
        Führt die Anomalieerkennung auf einer Datei mittels dem VAE Modell durch.
        Args:
            df_features: DataFrame mit den Feature-Daten.
            window_size: Größe des Sliding Windows
            threshold: Schwellenwert für Anomalieerkennung
        Returns:
            fig: Matplotlib Figure mit den Ergebnissen.
        """
        vae_cfg = get_settings().vae
        window_size = window_size if window_size is not None else vae_cfg.window_size
        threshold = threshold if threshold is not None else vae_cfg.threshold

        df_features = df_features.copy()

        # Überprüfen ob das Modell geladen ist
        if self.model is None:
            logger.error("Modell nicht geladen, Abbruch der Analyse.")
            return

        # Entferne unnötige Spalten (Timestamp & Confidence)
        cols_to_drop = [
            c
            for c in df_features.columns
            if "timestamp" in c.lower() or "confidence" in c.lower()
        ]

        if cols_to_drop:
            df_features = df_features.drop(columns=cols_to_drop)

        # Cleaning und Normalisierung
        df_features = df_features.fillna(0.0).select_dtypes(include=[np.number])
        angle_cols = [c for c in df_features.columns if "angle" in c or "abd" in c]
        if angle_cols:
            df_features[angle_cols] = df_features[angle_cols] / 180.0

        data: np.ndarray = df_features.values.astype(np.float32)

        # Modell laden
        model = self.model
        model.eval()

        # 3. Analyse (Sliding Window)
        criterion: nn.MSELoss = nn.MSELoss(reduction="none")
        errors_list: list[float] = []
        indices: list[int] = []

        # Prüfe ob genügend Daten für mindestens ein Fenster vorhanden sind
        if len(data) < window_size:
            logger.debug("Datei zu kurz.")
            return

        # Sliding Window mit Schrittweite 1 durchlaufen
        with torch.no_grad():
            for i in range(0, len(data) - window_size + 1):
                window = data[i : i + window_size]
                inp = torch.from_numpy(window).unsqueeze(0).to(self.device)
                recon, _, _ = model(inp)
                loss = torch.mean(criterion(recon, inp)).item()
                errors_list.append(loss)
                indices.append(i + window_size // 2)

        # Ergebnisse auswerten
        errors_arr: np.ndarray = np.array(errors_list)
        anomalies_array: np.ndarray = errors_arr > threshold

        # Vollständiges Anomalie-Array für alle Datenpunkte erstellen
        # -> Ersten 15 und die letzten 15 Punkte sind nie Anomalien -> Wird mit False gefüllt
        full_anomalies_array = np.zeros(len(data), dtype=bool)
        full_anomalies_array[indices] = anomalies_array

        plot_VAE_Anomalies = self.plot_VAE_results(
            indices, errors_arr, anomalies_array, THRESHOLD=threshold
        )

        return plot_VAE_Anomalies, full_anomalies_array

    def plot_VAE_results(self, indices, errors, anomalies, THRESHOLD: float):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(indices, errors, label="MSE Loss")
        ax.axhline(y=THRESHOLD, color="r", linestyle="--")
        if np.any(anomalies):
            ax.fill_between(
                indices,
                THRESHOLD,
                errors,
                where=anomalies,
                color="red",
                alpha=0.3,
            )
        ax.set_title("Anomalieerkennung mit VAE")
        return fig
