import logging
from pathlib import Path
from typing import Any

import numpy as np

from handpose.config_loader import get_settings

from ..HMM.model_hmm import HmmWrapper
from .features_dbscan import DbscanFeatureExtractor
from .model_dbscan import DbscanWrapper

logger = logging.getLogger(__name__)


class DbscanTrainer:
    def __init__(self) -> None:
        """
        Initialisierung des DBSCAN-Trainers.
        Lädt DBSCAN-Parameter und HMM-Modell für Vorfilterung.
        """
        settings = get_settings()
        dbscan_cfg = settings.dbscan
        hmm_cfg = settings.hmm

        self.extractor = DbscanFeatureExtractor()
        self.dbscan = DbscanWrapper(
            eps=dbscan_cfg.eps, min_samples=dbscan_cfg.min_samples
        )
        self.frames_to_take = dbscan_cfg.frames_to_take
        self.grasp_state = hmm_cfg.grasp_state
        self.raw_data_dir = Path(dbscan_cfg.paths.get("raw_data_dir"))

        self.hmm = HmmWrapper()
        hmm_model_dir = Path(hmm_cfg.paths.get("model_dir"))
        model_path = hmm_model_dir / "hmm_model.pkl"
        scaler_path = hmm_model_dir / "scaler.pkl"
        hmm_model_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.hmm.load(model_path, scaler_path)
            self.hmm_loaded = True
        except Exception:
            logger.warning(
                "WARNUNG: HMM Modell nicht gefunden. DBSCAN kann keine Punkte vorfiltern."
            )
            self.hmm_loaded = False

    def collect_points(self) -> list[Any] | None:
        """
        Sammelt relevante Punkte aus allen Trainingsdateien.
        Verwendet exakte Block-Logik:
        - Nur Grasp-State
        - Nur die letzten N Frames eines Griffs
        - Auswahl der aktiven Hand (tieferes Y) pro Griff

        Returns:
            np.ndarray: Array mit gesammelten Punkten.
        """
        if not self.hmm_loaded:
            return np.array([])

        files = list(self.raw_data_dir.glob("*.csv"))
        all_points = []

        logger.info(f"Sammle Punkte aus {len(files)} Dateien...")

        for f in files:
            try:
                df = self.extractor.process_csv(f)
                if df is None:
                    continue

                X_hmm, _ = self.extractor.calculate_features(df)
                states = self.hmm.predict(X_hmm)

                states_padded = np.append(states, -1)
                block_start = 0
                current_state = states[0]

                l_x = df["l_x_8"].values
                l_y = df["l_y_8"].values
                r_x = df["r_x_8"].values
                r_y = df["r_y_8"].values

                for j in range(1, len(states_padded)):
                    if states_padded[j] != current_state:
                        block_end = j

                        if current_state == self.grasp_state:
                            frames_to_take = self.frames_to_take
                            start_extract = max(block_start, block_end - frames_to_take)

                            l_pts = np.column_stack(
                                (
                                    l_x[start_extract:block_end],
                                    l_y[start_extract:block_end],
                                )
                            )
                            r_pts = np.column_stack(
                                (
                                    r_x[start_extract:block_end],
                                    r_y[start_extract:block_end],
                                )
                            )

                            if np.mean(l_pts[:, 1]) > np.mean(r_pts[:, 1]):
                                active = l_pts
                            else:
                                active = r_pts

                            valid = active[np.linalg.norm(active, axis=1) > 0.1]

                            if len(valid) > 0:
                                all_points.extend(valid)

                        current_state = states_padded[j]
                        block_start = j

            except Exception as e:
                logger.error(f"Skipping {f}: {e}")

        return np.array(all_points)

    def run(self) -> tuple[Any, Any, Any] | None:
        """
        Führt das vollständige ROI-Training durch.
        Sammelt Punkte, führt Clustering aus und gibt ROI-Zentren zurück.

        Returns:
            tuple | None: (X, labels, centers) oder None wenn keine Punkte gefunden.
        """
        X = self.collect_points()
        if len(X) == 0:
            logger.warning("Keine validen Punkte gefunden.")
            return None

        logger.info(f"Starte Clustering mit {len(X)} Punkten...")
        labels = self.dbscan.fit(X)
        centers = self.dbscan.get_centers(X)

        logger.info("\n--- GEFUNDENE ROIS (für configloader_HMM.py) ---")
        process_cfg = get_settings().process
        roi_names = [
            name.strip().lower() for name in process_cfg.cluster_names.split(",")
        ]
        logger.info("ROIS = {")
        for i, c in enumerate(centers):
            name = roi_names[i] if i < len(roi_names) else f"roi_{i}"
            logger.info(f"    '{name}': np.array([{c[0]:.4f}, {c[1]:.4f}]),")
        logger.info("}")

        return X, labels, centers
