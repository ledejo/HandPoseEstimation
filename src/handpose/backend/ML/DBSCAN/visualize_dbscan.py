import logging
from typing import Any
from pathlib import Path
import argparse
import os

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def show_dbscan_result(X: Any, labels: Any, centers: Any) -> Any | None:
    """
    Visualisiert die DBSCAN-Clustering-Ergebnisse.

    Args:
        X: Array mit Datenpunkten.
        labels: Array mit Cluster-Zuweisungen.
        centers: Array mit Cluster-Zentren.

    Returns:
        matplotlib.figure.Figure: Figure-Objekt mit dem Plot.
    """
    if X is None or labels is None:
        logger.warning("DBSCAN: Keine Daten für Visualisierung vorhanden")
        return None

    fig = plt.figure(figsize=(10, 8))

    mask_noise = labels == -1
    plt.scatter(
        X[mask_noise, 0],
        X[mask_noise, 1],
        c="gray",
        s=5,
        alpha=0.1,
        label="Noise",
    )

    mask_valid = labels != -1
    if np.any(mask_valid):
        plt.scatter(
            X[mask_valid, 0],
            X[mask_valid, 1],
            c=labels[mask_valid],
            cmap="jet",
            s=5,
            alpha=0.5,
        )

    if centers is not None and len(centers) > 0:
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            c="white",
            marker="X",
            s=200,
            edgecolors="black",
            zorder=10,
            label="Zentren",
        )

    plt.gca().invert_yaxis()
    plt.title("DBSCAN ROI Analyse")
    plt.legend()
    plt.tight_layout()

    return fig


def run_dbscan_visualization(
    output_path: str | Path | None = None,
    dpi: int = 180,
) -> Path:
    """
    Führt DBSCAN-Training aus und speichert die Visualisierung als PNG.

    Args:
        output_path: Zielpfad für PNG. Default: dbscan_visualization.png im DBSCAN-Ordner.
        dpi: Auflösung der Ausgabe.

    Returns:
        Path: Absoluter Pfad zur gespeicherten Datei.
    """
    try:
        from .train_dbscan import DbscanTrainer
    except ImportError:
        # Fallback für direkten Skript-Start ohne Paketkontext
        from handpose.backend.ML.DBSCAN.train_dbscan import DbscanTrainer

    trainer = DbscanTrainer()
    result = trainer.run()
    if result is None:
        raise RuntimeError(
            "DBSCAN-Training lieferte keine Daten für die Visualisierung."
        )

    X, labels, centers = result
    fig = show_dbscan_result(X=X, labels=labels, centers=centers)
    if fig is None:
        raise RuntimeError("show_dbscan_result() gab keine Figure zurück.")

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "dbscan_visualization.png"
    else:
        output_path = Path(output_path).expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("DBSCAN-Visualisierung gespeichert unter: %s", output_path)
    return output_path


def main() -> None:
    """CLI-Einstiegspunkt: trainiert DBSCAN und erzeugt direkt eine Visualisierung."""
    parser = argparse.ArgumentParser(
        description="Erzeuge DBSCAN-Visualisierung inkl. Training."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optionaler Ausgabepfad für PNG.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Auflösung der PNG-Datei (default: 180).",
    )
    args = parser.parse_args()

    output_path = run_dbscan_visualization(output_path=args.output, dpi=args.dpi)
    print(output_path)


if __name__ == "__main__":
    main()
