import logging
from typing import Any

from .predict_dbscan import DbscanInference
from .train_dbscan import DbscanTrainer
from .visualize_dbscan import show_dbscan_result

logger = logging.getLogger(__name__)


class DbscanController:
    def __init__(self) -> None:
        self.trainer = DbscanTrainer()
        self.inferencer = DbscanInference()
        logger.debug("DBSCAN Controller initialisiert")

    def train_rois(self) -> tuple[Any, Any, Any] | None:
        """
        Führt das ROI-Training durch.
        Sammelt Punkte aus allen Trainingsdateien und identifiziert Cluster-Zentren.

        Returns:
            tuple | None: (X, labels, centers) oder None bei Fehler.
        """
        logger.debug("Starte DBSCAN ROI-Training...")
        return self.trainer.run()

    def analyze_single_file(self, file_path: str) -> tuple[Any, Any, Any] | None:
        """
        Führt DBSCAN-Clustering auf einer einzelnen Datei aus.

        Args:
            file_path (str): Pfad zur CSV-Datei mit Keypoint-Daten.

        Returns:
            tuple | None: (X, labels, centers) oder None bei Fehler.
        """
        return self.inferencer.run_analysis(file_path)

    def show_results(self, X: Any, labels: Any, centers: Any) -> Any | None:
        """
        Visualisiert die DBSCAN-Clustering-Ergebnisse.

        Args:
            X: Array mit Datenpunkten.
            labels: Array mit Cluster-Zuweisungen.
            centers: Array mit Cluster-Zentren.

        Returns:
            matplotlib.figure.Figure: Figure-Objekt mit dem Plot.
        """
        return show_dbscan_result(X, labels, centers)
