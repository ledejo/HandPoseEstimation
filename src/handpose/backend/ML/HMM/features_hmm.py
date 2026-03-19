import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HmmFeatureExtractor:
    def __init__(self) -> None:
        """
        Initialisierung des Feature-Extractors für HMM.
        """
        pass

    def process_csv(self, file_path: str) -> pd.DataFrame | None:
        """
        Liest und bereinigt eine CSV-Datei mit Keypoint-Daten.

        Args:
            file_path (str): Pfad zur CSV-Datei.

        Returns:
            pd.DataFrame | None: Bereinigtes DataFrame oder None bei Fehler.
        """
        try:
            df = pd.read_csv(file_path, sep=";")
            cols_to_fix = [c for c in df.columns if "x" in c or "y" in c]
            for col in cols_to_fix:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = (
                df.interpolate(method="linear", limit_direction="both")
                .bfill()
                .ffill()
                .fillna(0)
            )
            return df
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None

    def calculate_features(self, df: pd.DataFrame) -> tuple[Any, list[dict[str, Any]]]:
        """
        Berechnet Features aus den Keypoint-Daten.

        Args:
            df (pd.DataFrame): DataFrame mit Keypoint-Daten.

        Returns:
            tuple: (X, raw_coords)
                - X: NumPy-Array mit berechneten Features
                - raw_coords: Liste mit Koordinaten-Dictionaries pro Frame
        """
        temp_data = []
        raw_coords = []

        for _, row in df.iterrows():
            l_index = np.array([row["l_x_8"], row["l_y_8"]])
            r_index = np.array([row["r_x_8"], row["r_y_8"]])
            l_thumb = np.array([row["l_x_4"], row["l_y_4"]])
            r_thumb = np.array([row["r_x_4"], row["r_y_4"]])

            row_dict = {}
            row_dict["hand_dist"] = np.linalg.norm(l_index - r_index)

            pinch_l = np.linalg.norm(l_index - l_thumb)
            pinch_r = np.linalg.norm(r_index - r_thumb)
            row_dict["mean_pinch"] = (pinch_l + pinch_r) / 2

            row_dict["max_reach_y"] = max(l_index[1], r_index[1])
            temp_data.append(row_dict)

            raw_coords.append({"l": l_index, "r": r_index})

        feat_df = pd.DataFrame(temp_data)
        feat_df["delta"] = (
            feat_df["hand_dist"]
            .diff()
            .fillna(0)
            .rolling(window=3, center=True)
            .mean()
            .fillna(0)
        )

        X = feat_df[["hand_dist", "mean_pinch", "max_reach_y", "delta"]].values
        return X, raw_coords
