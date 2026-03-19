import logging
import numpy as np

from handpose.config_loader import get_absolute_path, get_settings

from .features_hmm import HmmFeatureExtractor
from .model_hmm import HmmWrapper

logger = logging.getLogger(__name__)


class HmmTrainer:
    def __init__(self) -> None:
        """
        Initialisierung des HMM-Trainers.

        Lädt die Konfiguration und initialisiert Feature-Extraktor und Modell-Wrapper.
        """
        hmm_cfg = get_settings().hmm
        self.extractor = HmmFeatureExtractor()
        self.wrapper = HmmWrapper(
            n_states=hmm_cfg.n_states,
            n_iter=hmm_cfg.n_iter,
            random_state=hmm_cfg.random_state,
        )

    def run_training(self) -> None:
        """
        Führt das vollständige Training des HMM-Modells durch.

        Liest CSV-Dateien aus dem Trainingsverzeichnis, extrahiert Features
        und trainiert das HMM-Modell.
        """
        hmm_cfg = get_settings().hmm
        raw_data_dir = get_absolute_path(hmm_cfg.paths.get("raw_data_dir"))
        files = list(raw_data_dir.glob("*.csv"))

        if not files:
            logger.warning(f"Keine Dateien in {raw_data_dir} gefunden.")
            return

        logger.info(f"Starte Training mit {len(files)} Dateien...")
        X_all = []
        lengths = []

        for f in files:
            df = self.extractor.process_csv(f)
            if df is not None:
                feats, _ = self.extractor.calculate_features(df)
                if len(feats) > 0:
                    X_all.append(feats)
                    lengths.append(len(feats))

        if not X_all:
            logger.error("Fehler: Keine validen Features extrahiert.")
            return

        X_concat = np.concatenate(X_all)
        self.wrapper.train(X_concat, lengths)

        hmm_cfg = get_settings().hmm
        model_dir = get_absolute_path(hmm_cfg.paths.get("model_dir"))
        model_path = model_dir / "hmm_model.pkl"
        scaler_path = model_dir / "scaler.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.wrapper.save(model_path, scaler_path)

        # Nach erfolgreichem HMM-Training den Grasp-State automatisch neu bestimmen
        # und direkt in configs/models/hmm.yaml persistieren.

        try:
            from ..DBSCAN.auto_grasp_state import detect_and_persist_grasp_state

            detected_state = detect_and_persist_grasp_state()
            if detected_state is None:
                logger.warning(
                    "Grasp-State konnte nach HMM-Training nicht automatisch aktualisiert werden."
                )
            else:
                logger.info(
                    "Grasp-State nach HMM-Training automatisch gesetzt: %s",
                    detected_state,
                )
        except Exception as e:
            logger.warning(
                "Auto-Grasp-State nach HMM-Training fehlgeschlagen: %s",
                e,
            )

    def run_training_with_bic(
        self, min_states: int = 2, max_states: int = 30
    ) -> tuple[list[int], list[float]]:
        """
        Trainiert ein HMM mit automatischer State-Suche per BIC.

        Bestehende Trainingslogik bleibt unberuehrt; diese Methode ist additiv.

        Args:
            min_states: Untere Grenze der zu pruefenden State-Anzahlen.
            max_states: Obere Grenze der zu pruefenden State-Anzahlen.

        Returns:
            tuple[list[int], list[float]]: Gepruefte State-Werte und BIC-Historie.
        """
        if min_states < 2:
            raise ValueError("min_states muss >= 2 sein")
        if max_states < min_states:
            raise ValueError("max_states muss >= min_states sein")

        hmm_cfg = get_settings().hmm
        raw_data_dir = get_absolute_path(hmm_cfg.paths.get("raw_data_dir"))
        files = list(raw_data_dir.glob("*.csv"))

        if not files:
            logger.warning(f"Keine Dateien in {raw_data_dir} gefunden.")
            return [], []

        X_all: list[np.ndarray] = []
        lengths: list[int] = []

        for file_path in files:
            df = self.extractor.process_csv(file_path)
            if df is None:
                continue
            feats, _ = self.extractor.calculate_features(df)
            if len(feats) == 0:
                continue
            X_all.append(feats)
            lengths.append(len(feats))

        if not X_all:
            logger.error("Fehler: Keine validen Features extrahiert.")
            return [], []

        X_concat = np.concatenate(X_all)
        X_scaled = self.wrapper.scaler.fit_transform(X_concat)

        state_range = list(range(min_states, max_states + 1))
        bic_values: list[float] = []
        best_bic = np.inf
        best_n_states = min_states

        logger.info(
            f"Suche optimale State-Anzahl zwischen {min_states} und {max_states}..."
        )

        for n_states in state_range:
            try:
                candidate_wrapper = HmmWrapper(
                    n_states=n_states,
                    n_iter=hmm_cfg.n_iter,
                    random_state=hmm_cfg.random_state,
                )
                candidate_wrapper.model.fit(X_scaled, lengths)
                current_bic = candidate_wrapper.get_bic(X_scaled, lengths)
                bic_values.append(float(current_bic))
                logger.info(f"States: {n_states} | BIC: {current_bic:.2f}")

                if current_bic < best_bic:
                    best_bic = current_bic
                    best_n_states = n_states
            except Exception as exc:
                logger.error(f"Fehler bei n_states={n_states}: {exc}")
                bic_values.append(float("nan"))

        logger.info(f"Optimale Anzahl gefunden: {best_n_states} States.")
        logger.info("BIC-Analyse abgeschlossen. Es wird kein Modell gespeichert.")

        return state_range, bic_values
