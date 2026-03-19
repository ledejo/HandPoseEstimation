import logging
from typing import Any

from .predict_gmm import GmmInference
from .train_gmm import GmmTrainer
from .visualize_gmm import show_gmm_result

logger = logging.getLogger(__name__)


class GmmController:
    def __init__(self) -> None:
        self.trainer = GmmTrainer()
        self.inferencer = GmmInference()
        logger.debug("GMM Controller initialisiert")

    def train_rois(self) -> tuple[Any, Any, Any] | None:
        """Fuehrt das ROI-Training mit GMM durch."""
        logger.debug("Starte GMM ROI-Training...")
        return self.trainer.run()

    def analyze_single_file(self, file_path: str) -> tuple[Any, Any, Any] | None:
        """Fuehrt GMM-Clustering auf einer einzelnen Datei aus."""
        return self.inferencer.run_analysis(file_path)

    def show_results(self, X: Any, labels: Any, centers: Any) -> Any | None:
        """Visualisiert die GMM-Clustering-Ergebnisse."""
        return show_gmm_result(
            X,
            labels,
            centers,
            covariances=self.inferencer.wrapper.covariances_,
            probabilities=self.inferencer.wrapper.probabilities_,
        )
