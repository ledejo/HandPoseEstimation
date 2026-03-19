from pathlib import Path
from typing import Any

from handpose.config_loader import get_settings

from .features_hmm import HmmFeatureExtractor
from .model_hmm import HmmWrapper


class HmmPredictor:
    def __init__(self) -> None:
        """
        Initialisierung des HMM-Predictors.
        Lädt Feature-Extraktor und Modell-Wrapper für Inferenz.
        """
        self.extractor = HmmFeatureExtractor()
        self.wrapper = HmmWrapper()
        self.is_model_loaded = False

    def _ensure_model_loaded(self) -> None:
        """
        Stellt sicher, dass das Modell geladen ist.
        Lädt Modell und Scaler falls noch nicht geschehen.
        """
        if not self.is_model_loaded:
            model_dir = Path(get_settings().hmm.paths.get("model_dir"))
            model_path = model_dir / "hmm_model.pkl"
            scaler_path = model_dir / "scaler.pkl"
            model_dir.mkdir(parents=True, exist_ok=True)

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Modell nicht gefunden unter {model_path}. Bitte erst trainieren!"
                )

            self.wrapper.load(model_path, scaler_path)
            self.is_model_loaded = True

    def predict(self, file_path: str) -> tuple[Any, Any, Any] | None:
        """
        Führt Vorhersage auf einer CSV-Datei durch.

        Args:
            file_path (str): Pfad zur CSV-Datei mit Keypoint-Daten.

        Returns:
            tuple: (states, coords, timestamps)
                - states: Array mit vorhergesagten HMM-Zuständen
                - coords: Liste mit Koordinaten-Dictionaries
                - timestamps: Timestamp-Serie oder None
        """
        self._ensure_model_loaded()

        df = self.extractor.process_csv(file_path)
        if df is None:
            return None, None, None

        X, coords = self.extractor.calculate_features(df)
        states = self.wrapper.predict(X)

        return states, coords, df.get("timestamp", None)
