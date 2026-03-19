import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from handpose.backend.ML.HMM.controller_hmm import HmmController
from handpose.config_loader import get_absolute_path, get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_bic_analysis(min_states: int = 2, max_states: int = 30) -> int:
    """
    Trainiert HMM mit BIC-State-Suche und speichert die BIC-Kurve.

    Returns:
        int: 0 bei Erfolg, sonst 1.
    """
    logger.info("Initialisiere HMM Controller...")
    controller = HmmController()

    logger.info("Starte Training mit automatischer State-Ermittlung (BIC)...")
    try:
        state_range, bic_values = controller.train_with_bic(
            min_s=min_states,
            max_s=max_states,
        )
    except Exception as exc:
        logger.error(f"Training fehlgeschlagen: {exc}")
        return 1

    if not state_range or not bic_values:
        logger.error("BIC-Training lieferte keine Ergebnisse.")
        return 1

    hmm_cfg = get_settings().hmm
    report_dir = get_absolute_path(hmm_cfg.paths.get("report_dir"))
    report_dir.mkdir(parents=True, exist_ok=True)

    bic_array = np.asarray(bic_values, dtype=float)
    valid_mask = np.isfinite(bic_array)

    plt.figure(figsize=(10, 6))
    plt.plot(state_range, bic_values, "bo-", linewidth=2)

    if valid_mask.any():
        valid_indices = np.where(valid_mask)[0]
        min_valid_index = valid_indices[int(np.argmin(bic_array[valid_mask]))]
        plt.plot(
            state_range[min_valid_index],
            bic_values[min_valid_index],
            "ro",
            markersize=10,
            label="Minimum BIC",
        )

    plt.title('BIC-Kurve: Suche nach dem "Ellbogen"')
    plt.xlabel("Anzahl States")
    plt.ylabel("BIC Score (niedriger = besser)")
    plt.grid(True)
    plt.legend()

    save_path_bic = report_dir / "BIC_Kurve_Analyse.png"
    plt.savefig(save_path_bic)
    plt.close()
    logger.info(f"BIC-Kurve gespeichert unter: {save_path_bic}")

    logger.info("Prozess abgeschlossen.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trainiert HMM mit BIC-State-Suche und speichert die BIC-Kurve."
    )
    parser.add_argument("--min-states", type=int, default=2)
    parser.add_argument("--max-states", type=int, default=30)
    args = parser.parse_args()

    return run_bic_analysis(min_states=args.min_states, max_states=args.max_states)


if __name__ == "__main__":
    raise SystemExit(main())
