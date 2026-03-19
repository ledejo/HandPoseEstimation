import os
import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Optional, Any
from pathlib import Path
import argparse
import numpy as np
import logging

logger = logging.getLogger(__name__)


def show_gmm_result(
    X: Any,
    labels: Any,
    centers: Any,
    covariances: Any = None,
    probabilities: Any = None,
    uncertainty_threshold: float = 0.7,
) -> Optional[Any]:
    """
    Visualisiert die GMM-Clustering-Ergebnisse mit Kovarianz-Ellipsen.

    Args:
        X: Array mit Datenpunkten.
        labels: Array mit Cluster-Zuweisungen.
        centers: Array mit Cluster-Zentren.
        covariances: Array mit Kovarianzmatrizen (optional).
        probabilities: Array mit Zugehörigkeitswahrscheinlichkeiten (optional).
        uncertainty_threshold: Schwellwert für unsichere Punkte (default: 0.7).

    Returns:
        matplotlib.figure.Figure: Figure-Objekt mit dem Plot.
    """
    if X is None or labels is None:
        logger.warning("GMM: Keine Daten für Visualisierung vorhanden")
        return None

    fig = plt.figure(figsize=(12, 8))

    # Identify uncertain points (like noise in DBSCAN)
    if probabilities is not None:
        max_probs = np.max(probabilities, axis=1)
        mask_uncertain = max_probs < uncertainty_threshold
        mask_certain = ~mask_uncertain
    else:
        mask_uncertain = np.zeros(len(X), dtype=bool)
        mask_certain = np.ones(len(X), dtype=bool)

    # Plot uncertain points (gray)
    if np.any(mask_uncertain):
        plt.scatter(
            X[mask_uncertain, 0],
            X[mask_uncertain, 1],
            c="gray",
            s=5,
            alpha=0.2,
            label=f"Uncertain (prob < {uncertainty_threshold})",
        )

    # Plot certain points colored by cluster
    if np.any(mask_certain):
        plt.scatter(
            X[mask_certain, 0],
            X[mask_certain, 1],
            c=labels[mask_certain],
            cmap="jet",
            s=5,
            alpha=0.5,
        )

    # Plot cluster centers
    if centers is not None and len(centers) > 0:
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            c="white",
            marker="X",
            s=200,
            edgecolors="black",
            linewidths=2,
            zorder=10,
            label="Zentren",
        )

        # Add labels to centers
        roi_names = ["kappe", "mine", "oberteil", "ablage"]
        for i, center in enumerate(centers):
            name = roi_names[i] if i < len(roi_names) else f"ROI {i}"
            plt.annotate(
                name,
                xy=center,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    # Plot covariance ellipses (uncertainty visualization)
    if covariances is not None and centers is not None:
        for i, (center, covar) in enumerate(zip(centers, covariances)):
            # Get eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(covar)

            # Calculate angle of ellipse
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            # 2 standard deviations covers ~95% of points
            width, height = 2 * 2 * np.sqrt(eigenvalues)

            # Draw ellipse
            ellipse = Ellipse(
                xy=center,
                width=width,
                height=height,
                angle=angle,
                facecolor="none",
                edgecolor="black",
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
                zorder=5,
            )
            plt.gca().add_patch(ellipse)

    plt.gca().invert_yaxis()
    plt.title("GMM ROI Analyse (mit Kovarianz-Ellipsen)")
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def compare_clustering_methods(
    X_dbscan: Any,
    labels_dbscan: Any,
    centers_dbscan: Any,
    X_gmm: Any,
    labels_gmm: Any,
    centers_gmm: Any,
    covariances_gmm: Any = None,
) -> Optional[Any]:
    """
    Vergleicht DBSCAN und GMM Ergebnisse side-by-side.

    Args:
        X_dbscan: DBSCAN Datenpunkte
        labels_dbscan: DBSCAN Labels
        centers_dbscan: DBSCAN Zentren
        X_gmm: GMM Datenpunkte
        labels_gmm: GMM Labels
        centers_gmm: GMM Zentren
        covariances_gmm: GMM Kovarianzen (optional)

    Returns:
        matplotlib.figure.Figure: Figure mit 2 Subplots
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # DBSCAN Plot (left)
    ax = axes[0]
    plt.sca(ax)

    mask_noise = labels_dbscan == -1
    plt.scatter(
        X_dbscan[mask_noise, 0],
        X_dbscan[mask_noise, 1],
        c="gray",
        s=5,
        alpha=0.1,
        label="Noise",
    )

    mask_valid = labels_dbscan != -1
    if np.any(mask_valid):
        plt.scatter(
            X_dbscan[mask_valid, 0],
            X_dbscan[mask_valid, 1],
            c=labels_dbscan[mask_valid],
            cmap="jet",
            s=5,
            alpha=0.5,
        )

    if centers_dbscan is not None and len(centers_dbscan) > 0:
        plt.scatter(
            centers_dbscan[:, 0],
            centers_dbscan[:, 1],
            c="white",
            marker="X",
            s=200,
            edgecolors="black",
            linewidths=2,
            zorder=10,
        )

    ax.invert_yaxis()
    ax.set_title("DBSCAN Clustering")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # GMM Plot (right)
    ax = axes[1]
    plt.sca(ax)

    plt.scatter(
        X_gmm[:, 0],
        X_gmm[:, 1],
        c=labels_gmm,
        cmap="jet",
        s=5,
        alpha=0.5,
    )

    if centers_gmm is not None and len(centers_gmm) > 0:
        plt.scatter(
            centers_gmm[:, 0],
            centers_gmm[:, 1],
            c="white",
            marker="X",
            s=200,
            edgecolors="black",
            linewidths=2,
            zorder=10,
        )

        # Add covariance ellipses if available
        if covariances_gmm is not None:
            for center, covar in zip(centers_gmm, covariances_gmm):
                eigenvalues, eigenvectors = np.linalg.eigh(covar)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width, height = 2 * 2 * np.sqrt(eigenvalues)

                ellipse = Ellipse(
                    xy=center,
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.7,
                    zorder=5,
                )
                ax.add_patch(ellipse)

    ax.invert_yaxis()
    ax.set_title("GMM Clustering")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_gmm_visualization(
    output_path: str | Path | None = None,
    uncertainty_threshold: float = 0.7,
    dpi: int = 180,
) -> Path:
    """
    Führt GMM-Training aus und speichert die Visualisierung als PNG.

    Args:
        output_path: Zielpfad für PNG. Default: gmm_visualization_manual.png im GMM-Ordner.
        uncertainty_threshold: Schwellwert für unsichere Punkte.
        dpi: Auflösung der Ausgabe.

    Returns:
        Path: Absoluter Pfad zur gespeicherten Datei.
    """
    # Local import to avoid mandatory training deps when only plotting helpers are used
    try:
        from .train_gmm import GmmTrainer
    except ImportError:
        # Fallback for direct script execution (without package context)
        from handpose.backend.ML.GMM.train_gmm import GmmTrainer

    trainer = GmmTrainer()
    result = trainer.run()
    if result is None:
        raise RuntimeError("GMM-Training lieferte keine Daten für die Visualisierung.")

    X, labels, centers = result
    covariances = trainer.gmm.covariances_
    probabilities = trainer.gmm.probabilities_

    fig = show_gmm_result(
        X=X,
        labels=labels,
        centers=centers,
        covariances=covariances,
        probabilities=probabilities,
        uncertainty_threshold=uncertainty_threshold,
    )
    if fig is None:
        raise RuntimeError("show_gmm_result() gab keine Figure zurück.")

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "gmm_visualization_manual.png"
    else:
        output_path = Path(output_path).expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("GMM-Visualisierung gespeichert unter: %s", output_path)
    return output_path


def main() -> None:
    """CLI-Einstiegspunkt: trainiert GMM und erzeugt direkt eine Visualisierung."""
    parser = argparse.ArgumentParser(
        description="Erzeuge GMM-Visualisierung inkl. Training."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optionaler Ausgabepfad für PNG.",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.7,
        help="Schwellwert für unsichere Punkte (default: 0.7).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Auflösung der PNG-Datei (default: 180).",
    )
    args = parser.parse_args()

    output_path = run_gmm_visualization(
        output_path=args.output,
        uncertainty_threshold=args.uncertainty_threshold,
        dpi=args.dpi,
    )
    print(output_path)


if __name__ == "__main__":
    main()
