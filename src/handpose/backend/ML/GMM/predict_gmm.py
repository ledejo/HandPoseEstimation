from typing import Any
from pathlib import Path

import numpy as np

from handpose.config_loader import get_absolute_path, get_settings

from ..HMM.model_hmm import HmmWrapper
from .features_gmm import GmmFeatureExtractor
from .model_gmm import GmmWrapper
from .train_gmm import _load_runtime_config


class GmmInference:
    def __init__(self) -> None:
        """Initialisierung der GMM-Inferenz fuer einzelne Dateien."""
        cfg = _load_runtime_config()
        settings = get_settings()
        self.extractor = GmmFeatureExtractor()
        self.wrapper = GmmWrapper(
            n_components=int(cfg["n_components"]),
            covariance_type=str(cfg["covariance_type"]),
            random_state=int(cfg["random_state"]),
            n_init=int(cfg["n_init"]),
            max_iter=int(cfg["max_iter"]),
            reg_covar=float(cfg["reg_covar"]),
            init_params=str(cfg["init_params"]),
        )
        self.frames_to_take = int(cfg["frames_to_take"])
        self.grasp_state = int(cfg["grasp_state"])

        self.hmm = HmmWrapper()
        model_dir = Path(get_absolute_path(settings.hmm.paths.get("model_dir")))
        self.hmm.load(model_dir / "hmm_model.pkl", model_dir / "scaler.pkl")

    def run_analysis(self, file_path: str) -> tuple[Any | None, Any | None, Any | None]:
        """Fuehrt GMM-Analyse auf einer einzelnen Datei durch.

        Die Punktauswahl ist identisch zur DBSCAN/GMM-Trainingslogik:
        nur Grasp-State-Blocks, letzte N Frames, aktive Hand, Norm-Filter.
        """
        df = self.extractor.process_csv(file_path)
        if df is None:
            return None, None, None

        X_hmm, _ = self.extractor.calculate_features(df)
        if len(X_hmm) == 0:
            return None, None, None

        states = self.hmm.predict(X_hmm)
        if len(states) == 0:
            return None, None, None

        points = []
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
                    start_extract = max(block_start, block_end - self.frames_to_take)

                    l_pts = np.column_stack(
                        (l_x[start_extract:block_end], l_y[start_extract:block_end])
                    )
                    r_pts = np.column_stack(
                        (r_x[start_extract:block_end], r_y[start_extract:block_end])
                    )

                    active = (
                        l_pts if np.mean(l_pts[:, 1]) > np.mean(r_pts[:, 1]) else r_pts
                    )
                    valid = active[np.linalg.norm(active, axis=1) > 0.1]
                    if len(valid) > 0:
                        points.extend(valid)

                current_state = states_padded[j]
                block_start = j

        if not points:
            return None, None, None

        X_valid = np.array(points)
        labels = self.wrapper.fit(X_valid)
        centers = self.wrapper.get_centers()

        return X_valid, labels, centers
