from sklearn.mixture import GaussianMixture
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GmmWrapper:
    def __init__(
        self,
        n_components=4,
        covariance_type="full",
        random_state=42,
        n_init=20,
        max_iter=100,
        reg_covar=1e-6,
        init_params="kmeans",
    ):
        """
        Initialisierung des GMM-Wrapper.

        Args:
            n_components (int): Anzahl der zu findenden Cluster (ROIs).
            covariance_type (str): Art der Kovarianzmatrix:
                - 'full': Jede Komponente hat eigene Kovarianzmatrix (flexibel)
                - 'tied': Alle Komponenten teilen Kovarianzmatrix
                - 'diag': Diagonale Kovarianzmatrix (schneller)
                - 'spherical': Skalare Kovarianz (am schnellsten)
            random_state (int): Seed für Reproduzierbarkeit.
            n_init (int): Anzahl der Initialisierungen (beste wird gewählt).
            max_iter (int): Maximale Anzahl EM-Iterationen.
            reg_covar (float): Stabilisierungsterm auf Diagonale der Kovarianz.
            init_params (str): Initialisierungsmethode für die Mixture-Parameter.
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.init_params = init_params

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            reg_covar=reg_covar,
            init_params=init_params,
            verbose=0,
        )
        self.labels_ = None
        self.centers_ = None
        self.covariances_ = None
        self.probabilities_ = None

    def fit(self, X):
        """
        Führt das GMM-Clustering durch.

        Args:
            X: Array mit Datenpunkten (N x 2).

        Returns:
            Array mit Cluster-Zuweisungen (0 bis n_components-1).
        """
        logger.info(f"Fitting GMM with {self.n_components} components...")
        self.model.fit(X)

        # Predict hard labels
        self.labels_ = self.model.predict(X)

        # Get soft probabilities (optional, for uncertainty analysis)
        self.probabilities_ = self.model.predict_proba(X)

        # Store centers and covariances
        self.centers_ = self.model.means_
        self.covariances_ = self.model.covariances_

        # Log cluster statistics
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        logger.info(f"GMM found {len(unique_labels)} clusters:")
        for label, count in zip(unique_labels, counts):
            logger.info(
                f"  Cluster {label}: {count} points ({100 * count / len(X):.1f}%)"
            )

        # Log model quality metrics
        logger.info(f"BIC (Bayesian Information Criterion): {self.model.bic(X):.2f}")
        logger.info(f"AIC (Akaike Information Criterion): {self.model.aic(X):.2f}")
        logger.info(f"Log-likelihood: {self.model.score(X) * len(X):.2f}")

        return self.labels_

    def get_centers(self, X=None, top_k=None):
        """
        Gibt die Cluster-Zentren zurück, sortiert nach X-Koordinate.

        Args:
            X: Nicht verwendet (für API-Kompatibilität mit DBSCAN).
            top_k: Nicht verwendet bei GMM (n_components ist fix).

        Returns:
            Array mit Cluster-Zentren, sortiert nach X-Koordinate.
        """
        if self.centers_ is None:
            logger.warning("Model not fitted yet. Call fit() first.")
            return np.array([])

        centers = self.centers_.copy()

        # Sort by X coordinate (like DBSCAN for consistency)
        sorted_indices = np.argsort(centers[:, 0])
        centers = centers[sorted_indices]

        return centers

    def get_cluster_info(self):
        """
        Gibt detaillierte Cluster-Informationen zurück.

        Returns:
            dict: Dictionary mit Cluster-Statistiken.
        """
        if self.centers_ is None:
            return {}

        info = {
            "n_clusters": self.n_components,
            "centers": self.centers_,
            "covariances": self.covariances_,
            "weights": self.model.weights_,  # Mixing coefficients (prior probabilities)
            "converged": self.model.converged_,
            "n_iter": self.model.n_iter_,
        }

        return info

    def get_uncertainty_mask(self, threshold=0.7):
        """
        Identifiziert Punkte mit hoher Unsicherheit (ähnlich wie Noise bei DBSCAN).

        Args:
            threshold (float): Minimale Wahrscheinlichkeit (0-1).
                Punkte mit max_probability < threshold gelten als unsicher.

        Returns:
            Array (boolean): True für unsichere Punkte.
        """
        if self.probabilities_ is None:
            logger.warning("No probability data available.")
            return np.array([])

        max_probs = np.max(self.probabilities_, axis=1)
        uncertain_mask = max_probs < threshold

        return uncertain_mask
