import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .features_hmm import HmmFeatureExtractor
from .predict_hmm import HmmPredictor

logger = logging.getLogger(__name__)


def show_hmm_result(file_path: str) -> Any | None:
    """
    Visualisiert die HMM-Analyse-Ergebnisse für eine Datei.
    Zeigt HMM-Zustände, Interaktions-Features und räumliche Position.

    Args:
        file_path (str): Pfad zur CSV-Datei mit Keypoint-Daten.

    Returns:
        matplotlib.figure.Figure: Figure-Objekt mit den Plots.
    """
    logger.info(f"Analysiere Datei: {file_path}")

    extractor = HmmFeatureExtractor()
    predictor = HmmPredictor()

    df = extractor.process_csv(file_path)
    if df is None:
        return

    X, _ = extractor.calculate_features(df)
    hand_dist = X[:, 0]
    mean_pinch = X[:, 1]
    reach_y = X[:, 2]

    try:
        states, _, _ = predictor.predict(file_path)
    except Exception as e:
        logger.error(f"Fehler bei Vorhersage: {e}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].step(range(len(states)), states, where="mid", color="black", linewidth=2)
    axes[0].set_ylabel("State ID")
    axes[0].set_title("HMM Zustände")
    axes[0].grid(True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i in range(len(states) - 1):
        axes[0].axvspan(i, i + 1, color=colors[states[i] % 10], alpha=0.3)

    axes[1].plot(hand_dist, color="purple", label="Hand-Abstand")
    axes[1].plot(mean_pinch, color="green", label="Pinch")
    axes[1].set_ylabel("Abstand / Pinch")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    axes[2].plot(reach_y, color="orange", label="Tiefe (Y)")
    axes[2].set_ylabel("Y-Position")
    axes[2].legend(loc="upper right")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True)

    return fig
