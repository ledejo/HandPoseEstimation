import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_the_most_recent_value(df_kpis: pd.DataFrame, columnname: str) -> float | int:
    """
    Gibt den aktuellsten Wert einer Spalte zurück.

    Args:
        df_kpis (pd.DataFrame): DataFrame mit KPI-Daten.
        columnname (str): Name der Spalte.

    Returns:
        float | int: Der aktuellste Wert, gerundet auf 2 Dezimalstellen.
    """
    if len(df_kpis[columnname]) == 0:
        return 0
    return (df_kpis[columnname].iloc[-1]).round(2)


def get_delta_between_last_two_values(
    df_kpis: pd.DataFrame, columnname: str
) -> float | int:
    """
    Berechnet die Differenz zwischen den letzten zwei Werten einer Spalte.

    Args:
        df_kpis (pd.DataFrame): DataFrame mit KPI-Daten.
        columnname (str): Name der Spalte.

    Returns:
        float | int: Die Differenz, gerundet auf 2 Dezimalstellen.
    """
    if len(df_kpis[columnname]) < 2:
        return 0
    return ((df_kpis[columnname].iloc[-1]) - (df_kpis[columnname].iloc[-2])).round(2)
