import logging
import os
from typing import Any

import joblib
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class HmmWrapper:
    def __init__(
        self, n_states: int = 4, n_iter: int = 500, random_state: int = 42
    ) -> None:
        """
        Initialisierung des HMM-Wrapper-Modells.

        Args:
            n_states (int): Anzahl der versteckten Zustände im HMM.
            n_iter (int): Maximale Anzahl der Trainingsiterationen.
            random_state (int): Seed für Reproduzierbarkeit.
        """
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            verbose=True,
            random_state=random_state,
            init_params="kmeans",
        )
        self.scaler = StandardScaler()

    def train(self, X_concat: Any, lengths: list[int]) -> None:
        """
        Trainiert das HMM-Modell auf den übergebenen Daten.

        Args:
            X_concat: Konkatenierte Feature-Matrix aller Sequenzen.
            lengths: Liste mit Längen der einzelnen Sequenzen.
        """
        logger.debug("Skaliere Daten...")
        X_scaled = self.scaler.fit_transform(X_concat)
        logger.debug("Trainiere HMM...")
        self.model.fit(X_scaled, lengths)

    def predict(self, X: Any) -> Any:
        """
        Führt Vorhersage der versteckten Zustände durch.

        Args:
            X: Feature-Matrix für die Vorhersage.

        Returns:
            Array mit vorhergesagten Zuständen.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, model_path: str, scaler_path: str) -> None:
        """
        Speichert Modell und Scaler.

        Args:
            model_path (str): Pfad zum Speichern des Modells.
            scaler_path (str): Pfad zum Speichern des Scalers.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.debug(f"Model saved to {model_path}")

    def load(self, model_path: str, scaler_path: str) -> None:
        """
        Lädt Modell und Scaler von der Festplatte.

        Args:
            model_path (str): Pfad zum gespeicherten Modell.
            scaler_path (str): Pfad zum gespeicherten Scaler.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        logger.debug("Model loaded successfully.")

    def get_bic(
        self, X_scaled: Any, lengths: list[int], penalty_factor: float = 1.0
    ) -> float:
        """
        Berechnet das Bayesian Information Criterion (BIC) fuer das aktuelle Modell.

        Args:
            X_scaled: Bereits skalierte Feature-Matrix.
            lengths: Sequenzlaengen fuer hmmlearn.
            penalty_factor: Optionaler Multiplikator fuer den Komplexitaetsterm.

        Returns:
            float: BIC-Score (niedriger ist besser).
        """
        log_likelihood = self.model.score(X_scaled, lengths)
        n_samples = max(int(X_scaled.shape[0]), 1)
        n_features = int(X_scaled.shape[1])
        n_states = int(self.model.n_components)

        # Freiheitsgrade fuer GaussianHMM mit diagonaler Kovarianz.
        n_params = (
            (n_states - 1) + (n_states * (n_states - 1)) + (2 * n_states * n_features)
        )

        return float(
            (penalty_factor * np.log(n_samples) * n_params) - 2 * log_likelihood
        )
