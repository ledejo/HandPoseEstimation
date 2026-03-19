import logging

import pandas as pd

from handpose.config_loader import get_settings

logger = logging.getLogger(__name__)


class KPICalculator:
    def __init__(self, fps: int = 30) -> None:
        self.fps: int = fps
        self.grasp_state = get_settings().hmm.grasp_state

    def calculate_overall_time(self, df: pd.DataFrame) -> float:
        """
        Berechnet die Gesamtzeit (in Sekunden) eines Vorgangs basierend auf den Frame-Daten.

        Args:
            df (pd.DataFrame): DataFrame mit einer Spalte 'State', die den Zustand enthält.

        Returns:
            float: Gesamtzeit in Sekunden.
        """
        try:
            state_3_frames = df[df["State"] == self.grasp_state]

            if state_3_frames.empty:
                logger.warning(
                    f"Keine Frames mit State {self.grasp_state} gefunden. Returniere 0.0 Sekunden."
                )
                return 0.0

            first_state_3 = state_3_frames["Frame"].min()
            last_state_3 = state_3_frames["Frame"].max()

            if first_state_3 is not None and last_state_3 is not None:
                gesamtmontagedauer_sec = (last_state_3 - first_state_3) / self.fps
                logger.debug(
                    f"Gesamtzeit berechnet: {gesamtmontagedauer_sec:.2f}s (Frames: {first_state_3} - {last_state_3})"
                )
                return gesamtmontagedauer_sec

            logger.warning("Frame Min/Max sind None. Returniere 0.0 Sekunden.")
            return 0.0

        except Exception as e:
            logger.error(
                f"Fehler bei der Berechnung der Gesamtmontagedauer: {e}", exc_info=True
            )
            return 0.0

    def calculate_assembly_time_without_state(self, df: pd.DataFrame) -> float:
        """
        Berechnet die Montagezeit ohne den Zustand GRASP_STATE.

        Args:
            df (pd.DataFrame): DataFrame mit einer Spalte 'State', die den Zustand enthält.

        Returns:
            float: Montagezeit ohne Zustand 3 in Sekunden.
        """
        try:
            state_3_frames = df[df["State"] == self.grasp_state]

            if state_3_frames.empty:
                logger.warning(
                    f"Keine Frames mit State {self.grasp_state} gefunden. Returniere 0.0 Sekunden."
                )
                return 0.0

            first_state_3 = state_3_frames["Frame"].min()
            last_state_3 = state_3_frames["Frame"].max()

            span = last_state_3 - first_state_3
            count_state_3 = len(state_3_frames)
            frames_not_state_3 = span - count_state_3

            assembly_time_without_state_3_sec = frames_not_state_3 / self.fps
            logger.debug(
                f"Montagezeit ohne State 3: {assembly_time_without_state_3_sec:.2f}s ({frames_not_state_3} Frames)"
            )
            return assembly_time_without_state_3_sec

        except Exception as e:
            logger.error(
                f"Fehler bei der Berechnung der Montagezeit ohne Zustand 3: {e}",
                exc_info=True,
            )
            return 0.0

    def calculate_amount_of_anomalies(self, df: pd.DataFrame) -> int:
        """
        Berechnet die Anzahl der einzigartigen Anomalien im DataFrame.

        Args:
            df (pd.DataFrame): DataFrame mit einer Spalte 'is_anomaly', die True für Anomalien enthält.

        Returns:
            int: Anzahl der einzigartigen Anomalien.
        """
        # Prüfe, ob die Spalte 'is_anomaly' existiert
        if "is_anomaly" not in df.columns:
            logger.warning(
                "Spalte 'is_anomaly' nicht im DataFrame gefunden. VAE-Anomalieerkennung ist möglicherweise deaktiviert. Returniere 0 Anomalien."
            )
            return 0

        try:
            df = df.copy()
            df["is_anomaly"] = df["is_anomaly"].astype(bool)
            is_anomaly_shifted = df["is_anomaly"].shift(fill_value=False).astype(bool)

            neue_anomalien = df["is_anomaly"] & ~is_anomaly_shifted
            unique_anomalies_count = neue_anomalien.sum()

            logger.debug(f"{unique_anomalies_count} einzigartige Anomalien gefunden")
            return unique_anomalies_count

        except Exception as e:
            logger.error(f"Fehler bei der Berechnung der Anomalien: {e}", exc_info=True)
            return 0

    def calculate_kpi_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt eine Zusammenfassung der KPIs gruppiert nach Teilen.

        Args:
            df (pd.DataFrame): DataFrame mit Teil-Informationen und Frame-Daten.

        Returns:
            pd.DataFrame: Zusammengefasste KPI-Daten mit Teilen als Index.
        """
        # 1. Filtern: Wir wollen nur Zeilen, in denen ein echtes Teil erkannt wurde (nicht "NONE")
        df_parts = df[df["Teil"] != "NONE"].copy()

        # Liste für die Ergebnisse
        summary_list = []

        # 2. Gruppieren nach 'Teil'
        # Wir iterieren durch die einzigartigen Teile, um die Metriken zu berechnen
        for part_name in df_parts["Teil"].unique():
            # Teil-Datensatz filtern
            part_data = df_parts[df_parts["Teil"] == part_name]

            # A. Anzahl Frames berechnen
            count = len(part_data)

            # B. Range bestimmen (Min Frame bis Max Frame)
            start_frame = part_data["Frame"].min()
            end_frame = part_data["Frame"].max()
            range_str = f"{start_frame}-{end_frame}"

            # C. Best Frame finden
            # Suche die Zeile, wo Best_Frame == True ist
            best_frame_row = part_data[part_data["Best_Frame"]]

            if not best_frame_row.empty:
                # Wenn vorhanden, nimm den ersten Treffer
                best_frame_val = best_frame_row["Frame"].iloc[0]
            else:
                best_frame_val = None  # Fallback, falls kein Best Frame markiert wurde

            # Ergebnisse in die Liste packen
            summary_list.append(
                {
                    "Teil": part_name,
                    "Anzahl_Frames": count,
                    "Ranges": range_str,
                    "Best_Frame": best_frame_val,
                }
            )

        # 3. DataFrame erstellen und formatieren
        summary_df = pd.DataFrame(summary_list)

        # 'Teil' als Index setzen, wie im gewünschten Output
        if not summary_df.empty:
            summary_df.set_index("Teil", inplace=True)

        return summary_df
