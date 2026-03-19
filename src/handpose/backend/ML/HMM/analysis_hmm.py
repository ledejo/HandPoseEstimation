import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from handpose.config_loader import get_settings

logger = logging.getLogger(__name__)


def get_ellipse_dist(
    point: np.ndarray, center: list[float], rx: float, ry: float
) -> float:
    """
    Berechnet die normalisierte Distanz eines Punkts zum Ellipsen-Zentrum.

    Args:
        point: 2D-Koordinate des Punkts.
        center: 2D-Koordinate des Ellipsen-Zentrums.
        rx: Radius der Ellipse in x-Richtung.
        ry: Radius der Ellipse in y-Richtung.

    Returns:
        float: Normalisierte Distanz (<=1.0 bedeutet innerhalb der Ellipse).
    """
    term_x = ((point[0] - center[0]) / rx) ** 2
    term_y = ((point[1] - center[1]) / ry) ** 2
    return term_x + term_y


def analyze_frames_and_structure(
    states: np.ndarray, coords: list[dict[str, np.ndarray]], file_path_in: str
) -> pd.DataFrame | None:
    """
    Führt eine detaillierte Frame-basierte Analyse durch.
    Identifiziert welche ROIs in welchen Frames getroffen wurden.

    Args:
        states: Array mit HMM-Zuständen pro Frame.
        coords: Liste mit Koordinaten-Dictionaries pro Frame.
        file_path_in (str): Pfad zur analysierten Datei.

    Returns:
        pd.DataFrame: DataFrame mit detaillierten Frame-Informationen.
    """
    # Load configuration
    hmm_cfg = get_settings().hmm
    grasp_state = hmm_cfg.grasp_state
    rois = hmm_cfg.rois
    roi_rad_x = hmm_cfg.roi_rad_x
    roi_rad_y = hmm_cfg.roi_rad_y
    min_y_for_grasp = hmm_cfg.min_y_for_grasp

    num_frames = len(states)
    best_hit_per_roi = {name: {"frame": -1, "score": np.inf} for name in rois.keys()}
    grasp_indices = np.where(states == grasp_state)[0]

    for idx in grasp_indices:
        pos = coords[idx]
        ly = pos["l"][1]
        ry = pos["r"][1]

        if max(ly, ry) < min_y_for_grasp:
            continue

        for name, center in rois.items():
            dist_l = get_ellipse_dist(pos["l"], center, roi_rad_x, roi_rad_y)
            dist_r = get_ellipse_dist(pos["r"], center, roi_rad_x, roi_rad_y)
            min_dist = min(dist_l, dist_r)

            if min_dist <= 1.0:
                if min_dist < best_hit_per_roi[name]["score"]:
                    best_hit_per_roi[name]["score"] = min_dist
                    best_hit_per_roi[name]["frame"] = idx

    df_analysis = pd.DataFrame(
        {
            "Frame": np.arange(1, num_frames + 1),
            "State": states,
            "Teil": "NONE",
            "Best_Frame": False,
        }
    )

    for idx in grasp_indices:
        frame_num_one_based = idx + 1
        pos = coords[idx]
        ly = pos["l"][1]
        ry = pos["r"][1]

        if max(ly, ry) < min_y_for_grasp:
            continue

        best_hit_in_frame = {"name": "NONE", "score": 1.0}

        for name, center in rois.items():
            dist_l = get_ellipse_dist(pos["l"], center, roi_rad_x, roi_rad_y)
            dist_r = get_ellipse_dist(pos["r"], center, roi_rad_x, roi_rad_y)
            min_dist = min(dist_l, dist_r)

            if min_dist <= 1.0:
                if min_dist < best_hit_in_frame["score"]:
                    best_hit_in_frame["score"] = min_dist
                    best_hit_in_frame["name"] = name.upper()

        if best_hit_in_frame["name"] != "NONE":
            df_analysis.loc[df_analysis["Frame"] == frame_num_one_based, "Teil"] = (
                best_hit_in_frame["name"]
            )

        is_best_frame = False
        for name in rois.keys():
            if idx == best_hit_per_roi[name]["frame"]:
                is_best_frame = True
                break

        df_analysis.loc[df_analysis["Frame"] == frame_num_one_based, "Best_Frame"] = (
            is_best_frame
        )

    df_analysis.attrs["file_name"] = os.path.basename(file_path_in)
    return df_analysis


class KPICalculator:
    def __init__(
        self,
        grasp_state: int | None = None,
        rois: dict | None = None,
        roi_rad_x: float | None = None,
        roi_rad_y: float | None = None,
        min_y: float | None = None,
    ):
        """
        Initialisierung des KPI-Calculators.

        Args:
            grasp_state: HMM-State der Griffphase.
            rois: Dictionary mit ROI-Definitionen.
            roi_rad_x: Ellipsen-Radius in x-Richtung.
            roi_rad_y: Ellipsen-Radius in y-Richtung.
            min_y: Minimale Y-Koordinate für Grifferkennung.
        """
        hmm_cfg = get_settings().hmm
        self.grasp_state = (
            grasp_state if grasp_state is not None else hmm_cfg.grasp_state
        )
        self.rois = rois if rois is not None else hmm_cfg.rois
        self.rad_x = roi_rad_x if roi_rad_x is not None else hmm_cfg.roi_rad_x
        self.rad_y = roi_rad_y if roi_rad_y is not None else hmm_cfg.roi_rad_y
        self.min_y = min_y if min_y is not None else hmm_cfg.min_y_for_grasp
        self.required_parts = {
            name.lower() for name in self.rois.keys() if name.lower() != "ablage"
        }

    def calculate_kpis(
        self, states: np.ndarray, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Berechnet KPIs basierend auf HMM-Zuständen und Keypoint-Daten.

        Args:
            states: Array mit vorhergesagten HMM-Zuständen.
            df (pd.DataFrame): DataFrame mit Keypoint-Daten.

        Returns:
            pd.DataFrame: DataFrame mit berechneten KPIs.
        """
        raw_coords = []
        for _, row in df.iterrows():
            raw_coords.append(
                {
                    "l": np.array([row.get("l_x_8", 0), row.get("l_y_8", 0)]),
                    "r": np.array([row.get("r_x_8", 0), row.get("r_y_8", 0)]),
                }
            )

        timestamps = (
            df["timestamp"] if "timestamp" in df.columns else pd.Series(range(len(df)))
        )
        try:
            metrics = self.calculate_metrics(states, raw_coords, timestamps)
            return pd.DataFrame([metrics])
        except Exception as e:
            logger.error(f"Fehler bei der KPI-Berechnung: {e}")
            return pd.DataFrame()

    def calculate_metrics(
        self,
        states: np.ndarray,
        raw_coords: list[dict[str, np.ndarray]],
        timestamps: pd.Series,
    ) -> dict[str, Any]:
        """
        Berechnet Metriken für eine Sequenz.

        Args:
            states: Array mit HMM-Zuständen.
            raw_coords: Liste mit Koordinaten pro Frame.
            timestamps: Serie mit Zeitstempeln.

        Returns:
            dict: Dictionary mit berechneten Metriken.
        """
        try:
            duration = timestamps.iloc[-1] - timestamps.iloc[0]
        except Exception as e:
            logger.warning(
                f"Fehler bei der Berechnung der Dauer: {e}. Verwende 0 als Standardwert."
            )
            duration = 0

        detected_parts = set()
        states_padded = np.append(states, -1)
        block_start = 0
        current_state = states[0]

        for i in range(1, len(states_padded)):
            if states_padded[i] != current_state:
                block_end = i
                if current_state == self.grasp_state:
                    self._check_rois_in_block(
                        raw_coords=raw_coords,
                        start_idx=block_start,
                        end_idx=block_end,
                        detected_parts=detected_parts,
                    )
                current_state = states_padded[i]
                block_start = i

        missing = self.required_parts - detected_parts
        is_complete = len(missing) == 0

        return {
            "Dauer (s)": round(float(duration), 2),
            "Vollständig": is_complete,
            "Fehlende Teile": ", ".join(sorted(missing)) if missing else "-",
            "Gefundene Teile": ", ".join(sorted(detected_parts)),
        }

    def _check_rois_in_block(
        self,
        raw_coords: list[dict[str, np.ndarray]],
        start_idx: int,
        end_idx: int,
        detected_parts: set,
    ) -> set:
        """
        Überprüft welche ROIs in einem Zustandsblock berührt wurden.

        Args:
            raw_coords: Liste mit Koordinaten.
            start_idx: Startindex des Blocks.
            end_idx: Endindex des Blocks.
            detected_parts: Set zum Sammeln erkannter Teile (wird modifiziert).
        """
        safe_start = max(0, start_idx - 5)
        safe_end = min(len(raw_coords), end_idx + 5)
        block_coords = raw_coords[safe_start:safe_end]

        y_values = [max(fc["l"][1], fc["r"][1]) for fc in block_coords]
        if not y_values or np.max(y_values) < self.min_y:
            return

        for roi_name, roi_pos in self.rois.items():
            for frame_coord in block_coords:
                in_ellipse_l = self._is_in_ellipse(frame_coord["l"], roi_pos)
                in_ellipse_r = self._is_in_ellipse(frame_coord["r"], roi_pos)
                if in_ellipse_l or in_ellipse_r:
                    detected_parts.add(roi_name)
                    break

    def _is_in_ellipse(self, point: np.ndarray, center: list[float]) -> bool:
        """
        Prüft ob ein Punkt innerhalb einer Ellipse liegt.

        Args:
            point: 2D-Koordinate des Punkts.
            center: 2D-Koordinate des Ellipsen-Zentrums.

        Returns:
            bool: True wenn Punkt in Ellipse liegt.
        """
        term_x = ((point[0] - center[0]) / self.rad_x) ** 2
        term_y = ((point[1] - center[1]) / self.rad_y) ** 2
        return (term_x + term_y) <= 1.0
