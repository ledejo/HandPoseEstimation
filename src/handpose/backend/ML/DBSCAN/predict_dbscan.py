from typing import Any

import numpy as np
import pandas as pd

from handpose.config_loader import get_settings

from .features_dbscan import DbscanFeatureExtractor
from .model_dbscan import DbscanWrapper


class DbscanInference:
    def __init__(self) -> None:
        """
        Initialisierung der DBSCAN-Inferenz.
        Erstellt neuen Wrapper für Ad-hoc-Analysen einzelner Dateien.
        """
        dbscan_cfg = get_settings().dbscan
        self.extractor = DbscanFeatureExtractor()
        self.wrapper = DbscanWrapper(
            eps=dbscan_cfg.eps, min_samples=dbscan_cfg.min_samples
        )

    def run_analysis(self, file_path: str) -> tuple[Any | None, Any | None, Any | None]:
        """
        Führt DBSCAN-Analyse auf einer einzelnen Datei aus.
        Extrahiert Punkte und führt Clustering durch (ohne HMM-Filter).

        Args:
            file_path (str): Pfad zur CSV-Datei mit Keypoint-Daten.

        Returns:
            tuple | None: (X_valid, labels, centers) oder (None, None, None) bei Fehler.
        """
        df = self.extractor.process_csv(file_path)
        if df is None:
            return None, None, None

        lx = pd.to_numeric(df["l_x_8"], errors="coerce").fillna(0)
        ly = pd.to_numeric(df["l_y_8"], errors="coerce").fillna(0)
        points = np.column_stack((lx, ly))

        mask = (points[:, 0] > 0.01) & (points[:, 1] > 0.01)
        X_valid = points[mask]

        if len(X_valid) == 0:
            return None, None, None

        labels = self.wrapper.fit(X_valid)
        centers = self.wrapper.get_centers(X_valid)

        return X_valid, labels, centers
