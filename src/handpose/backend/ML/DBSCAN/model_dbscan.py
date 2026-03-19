import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN


class DbscanWrapper:
    """Wrapper für DBSCAN-Clustering mit ROI-Extraktion."""

    def __init__(self, eps: float = 0.01, min_samples: int = 50) -> None:
        """
        Initialisiert DBSCAN-Wrapper mit Hyperparametern.

        Args:
            eps (float): Maximale Distanz zwischen zwei Punkten in einem Cluster.
            min_samples (int): Minimale Anzahl an Punkten für einen Cluster.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_: npt.NDArray[np.int32] | None = None

    def fit(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        """
        Führt DBSCAN-Clustering auf Datenpunkten durch.

        Args:
            X: Array mit Datenpunkten der Form (n_samples, n_features).

        Returns:
            Array mit Cluster-Zuweisungen. -1 kennzeichnet Rauschen.
        """
        self.model.fit(X)
        self.labels_ = self.model.labels_
        return self.labels_

    def get_centers(
        self, X: npt.NDArray[np.float64], top_k: int = 4
    ) -> npt.NDArray[np.float64]:
        """
        Berechnet Zentren der Top-K größten Cluster.

        Args:
            X: Array mit Datenpunkten der Form (n_samples, n_features).
            top_k (int): Anzahl der zurückzugebenden größten Cluster.

        Raises:
            ValueError: Wenn fit() noch nicht aufgerufen wurde.
        """
        if self.labels_ is None:
            raise ValueError("Modell muss erst mit fit() trainiert werden")

        unique_labels = set(self.labels_) - {-1}  # Rauschen ausschließen

        if not unique_labels:
            return np.array([])

        centers = np.array(
            [X[self.labels_ == label].mean(axis=0) for label in unique_labels]
        )

        # Wenn mehr als top_k Cluster -> Nur größte behalten
        if len(centers) > top_k:
            counts = [(label, np.sum(self.labels_ == label)) for label in unique_labels]
            counts.sort(key=lambda x: x[1], reverse=True)
            top_indices = [label for label, _ in counts[:top_k]]
            centers = np.array(
                [X[self.labels_ == label].mean(axis=0) for label in top_indices]
            )

        # Sortierung nach X-Koordinate für konsistente ROI-Reihenfolge
        if len(centers) > 0:
            sorted_indices = np.argsort(centers[:, 0])
            centers = centers[sorted_indices]

        return centers
