from pathlib import Path
from typing import Tuple, Any
import numpy as np
import logging
import yaml

from handpose.config_loader import ROOT_DIR, get_absolute_path, get_settings
from .model_gmm import GmmWrapper
from .features_gmm import GmmFeatureExtractor
from ..HMM.model_hmm import HmmWrapper

logger = logging.getLogger(__name__)

DEFAULT_GMM_CONFIG = {
    "n_components": 4,
    "covariance_type": "full",
    "random_state": 42,
    "n_init": 20,
    "max_iter": 100,
    "reg_covar": 1e-6,
    "init_params": "kmeans",
    "frames_to_take": 5,
}


def _persist_roi_centers_to_hmm_config(centers: np.ndarray) -> None:
    """Schreibt GMM-Clusterzentren in configs/models/hmm.yaml (roi_centers).

    Das bestehende Dateiformat bleibt erhalten: Nur der Block ``roi_centers``
    wird ersetzt und als Inline-Listen ``- [x, y]`` geschrieben.
    """
    config_path = ROOT_DIR / "configs" / "models" / "hmm.yaml"

    content = config_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    roi_lines = ["roi_centers:"]
    roi_lines.extend([f"  - [{float(c[0]):.4f}, {float(c[1]):.4f}]" for c in centers])

    updated_lines = []
    i = 0
    replaced = False

    while i < len(lines):
        line = lines[i]
        if line.startswith("roi_centers:"):
            updated_lines.extend(roi_lines)
            replaced = True
            i += 1

            # Ueberspringe alten roi_centers-Block (alle eingerueckten Folgelinien)
            while i < len(lines):
                next_line = lines[i]
                if next_line.startswith(" ") or next_line.startswith("\t"):
                    i += 1
                    continue
                break
            continue

        updated_lines.append(line)
        i += 1

    if not replaced:
        if updated_lines and updated_lines[-1] != "":
            updated_lines.append("")
        updated_lines.extend(roi_lines)

    new_content = "\n".join(updated_lines) + "\n"
    config_path.write_text(new_content, encoding="utf-8")

    get_settings.cache_clear()
    logger.info("HMM-Config aktualisiert: roi_centers aus GMM übernommen.")


def _load_runtime_config() -> dict[str, Any]:
    """
    Lädt GMM/HMM-Laufzeitkonfiguration robust.

    Priorität:
    1. defaults
    2. settings aus config_loader (hmm/dbscan)
    3. optionale overrides aus configs/models/gmm.yaml (falls vorhanden)
    """
    config = dict(DEFAULT_GMM_CONFIG)

    settings = get_settings()
    config["grasp_state"] = int(settings.hmm.grasp_state)
    config["hmm_model_dir"] = get_absolute_path(settings.hmm.paths.get("model_dir"))

    raw_data_rel = (
        settings.dbscan.paths.get("raw_data_dir")
        or settings.hmm.paths.get("raw_data_dir")
        or "data/03_processed/keypoints/training"
    )
    config["raw_data_dir"] = get_absolute_path(raw_data_rel)

    gmm_cfg_path = ROOT_DIR / "configs" / "models" / "gmm.yaml"
    if gmm_cfg_path.exists():
        with gmm_cfg_path.open("r", encoding="utf-8") as f:
            gmm_cfg = yaml.safe_load(f) or {}

        for key in (
            "n_components",
            "covariance_type",
            "random_state",
            "n_init",
            "max_iter",
            "reg_covar",
            "init_params",
            "frames_to_take",
        ):
            if key in gmm_cfg:
                config[key] = gmm_cfg[key]

        gmm_paths = gmm_cfg.get("paths", {})
        if isinstance(gmm_paths, dict) and gmm_paths.get("raw_data_dir"):
            config["raw_data_dir"] = get_absolute_path(gmm_paths["raw_data_dir"])

    return config


class GmmTrainer:
    def __init__(self) -> None:
        """
        Initialisierung des GMM-Trainers.
        Lädt GMM-Parameter und HMM-Modell für Vorfilterung.
        """
        cfg = _load_runtime_config()

        self.n_components = int(cfg["n_components"])
        self.covariance_type = str(cfg["covariance_type"])
        self.random_state = int(cfg["random_state"])
        self.n_init = int(cfg["n_init"])
        self.max_iter = int(cfg["max_iter"])
        self.reg_covar = float(cfg["reg_covar"])
        self.init_params = str(cfg["init_params"])
        self.frames_to_take = int(cfg["frames_to_take"])
        self.grasp_state = int(cfg["grasp_state"])
        self.raw_data_dir = Path(cfg["raw_data_dir"])
        self.hmm_model_dir = Path(cfg["hmm_model_dir"])

        self.gmm = GmmWrapper(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
            reg_covar=self.reg_covar,
            init_params=self.init_params,
        )
        self.extractor = GmmFeatureExtractor()

        # HMM für Vorfilterung
        self.hmm_wrapper = HmmWrapper()
        hmm_model_path = self.hmm_model_dir / "hmm_model.pkl"
        hmm_scaler_path = self.hmm_model_dir / "scaler.pkl"
        self.hmm_model_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.hmm_wrapper.load(hmm_model_path, hmm_scaler_path)
            self.hmm_loaded = True
        except Exception as e:
            logger.warning(
                f"WARNUNG: HMM Modell nicht gefunden ({e}). GMM kann keine Punkte vorfiltern."
            )
            self.hmm_loaded = False

        logger.info(
            f"GmmTrainer initialisiert: n_components={self.n_components}, "
            f"covariance_type={self.covariance_type}, n_init={self.n_init}, "
            f"max_iter={self.max_iter}, reg_covar={self.reg_covar}, "
            f"init_params={self.init_params}, grasp_state={self.grasp_state}, "
            f"raw_data_dir={self.raw_data_dir}"
        )

    def collect_points(self) -> np.ndarray:
        """
        Sammelt Trainingspunkte aus allen CSV-Dateien mit derselben
        Datengrundlage wie DBSCAN:
        - nur HMM-Grasp-Blocks
        - nur die letzten ``frames_to_take`` Frames je Block
        - pro Block aktive Hand (hoeherer mittlerer Y-Wert)
        - gueltige Punkte per Norm-Filter (> 0.1)

        Returns:
            np.ndarray: Array mit gefilterten Punkten (N x 2).
        """
        if not self.hmm_loaded:
            logger.error("HMM model not loaded. Cannot collect points.")
            return np.array([]).reshape(0, 2)

        raw_data_path = self.raw_data_dir
        csvs = sorted(raw_data_path.glob("*.csv"))

        if not csvs:
            logger.error(f"Keine CSV-Dateien gefunden in {raw_data_path}")
            return np.array([]).reshape(0, 2)

        logger.info(f"Sammle Punkte aus {len(csvs)} Dateien...")
        logger.info(
            f"GMM Vorfilter aktiv: nutze nur HMM-Grasp-State={self.grasp_state} (frames_to_take={self.frames_to_take})"
        )

        all_points = []
        blocks_total = 0
        grasp_blocks = 0
        points_before_valid = 0
        points_after_valid = 0

        for csv_path in csvs:
            try:
                df = self.extractor.process_csv(csv_path)
                if df is None:
                    continue

                # HMM Features berechnen
                X, _ = self.extractor.calculate_features(df)
                if len(X) == 0:
                    continue

                # HMM Prediction
                states = self.hmm_wrapper.predict(X)

                # Manual block detection (same as train_dbscan.py)
                states_padded = np.append(states, -1)
                block_start = 0
                current_state = states[0]

                l_x = df["l_x_8"].values
                l_y = df["l_y_8"].values
                r_x = df["r_x_8"].values
                r_y = df["r_y_8"].values

                for j in range(1, len(states_padded)):
                    if states_padded[j] != current_state:
                        blocks_total += 1
                        block_end = j

                        if current_state == self.grasp_state:
                            grasp_blocks += 1
                            frames_to_take = self.frames_to_take
                            start_extract = max(block_start, block_end - frames_to_take)

                            l_pts = np.column_stack(
                                (
                                    l_x[start_extract:block_end],
                                    l_y[start_extract:block_end],
                                )
                            )
                            r_pts = np.column_stack(
                                (
                                    r_x[start_extract:block_end],
                                    r_y[start_extract:block_end],
                                )
                            )

                            if np.mean(l_pts[:, 1]) > np.mean(r_pts[:, 1]):
                                active = l_pts
                            else:
                                active = r_pts

                            points_before_valid += len(active)
                            valid = active[np.linalg.norm(active, axis=1) > 0.1]
                            if len(valid) > 0:
                                all_points.extend(valid)
                                points_after_valid += len(valid)

                        current_state = states_padded[j]
                        block_start = j

            except Exception as e:
                logger.warning(f"Error processing {csv_path}: {e}")
                continue

        logger.info(
            f"HMM-Filter Statistik: blocks_total={blocks_total}, grasp_blocks={grasp_blocks}, "
            f"points_before_valid={points_before_valid}, points_after_valid={points_after_valid}"
        )

        if not all_points:
            logger.warning("Keine validen Punkte nach HMM-Filterung gefunden.")
            return np.array([]).reshape(0, 2)

        return np.array(all_points)

    def run(self) -> Tuple[Any, Any, Any] | None:
        """
        Führt das komplette GMM-Training durch:
        1. Sammelt Punkte aus allen Training-Dateien (mit HMM-Filter)
        2. Führt GMM-Clustering durch
        3. Gibt Ergebnisse zurück

        Returns:
            tuple | None: (X, labels, centers) oder None wenn keine Punkte gefunden.
        """
        X = self.collect_points()
        if len(X) == 0:
            logger.warning("Keine validen Punkte gefunden.")
            return None

        logger.info(f"Starte GMM-Clustering mit {len(X)} Punkten...")
        labels = self.gmm.fit(X)
        centers = self.gmm.get_centers()

        try:
            _persist_roi_centers_to_hmm_config(centers)
        except (FileNotFoundError, OSError, yaml.YAMLError) as e:
            logger.warning(f"Konnte roi_centers nicht persistieren: {e}")

        logger.info("\n--- GEFUNDENE ROIS (für hmm_config.yaml) ---")
        roi_names = ["kappe", "mine", "oberteil", "ablage"]
        logger.info("rois:")
        for i, c in enumerate(centers):
            name = roi_names[i] if i < len(roi_names) else f"roi_{i}"
            logger.info(f"  {name}: [{c[0]:.4f}, {c[1]:.4f}]")

        # Log cluster quality info
        cluster_info = self.gmm.get_cluster_info()
        logger.info(f"\nModel converged: {cluster_info.get('converged', 'unknown')}")
        logger.info(f"Iterations: {cluster_info.get('n_iter', 'unknown')}")

        # Log uncertainty statistics
        uncertain_mask = self.gmm.get_uncertainty_mask(threshold=0.7)
        n_uncertain = np.sum(uncertain_mask)
        logger.info(
            f"Uncertain points (prob < 0.7): {n_uncertain} ({100 * n_uncertain / len(X):.1f}%)"
        )

        return X, labels, centers


def main() -> None:
    """Standalone-Skript für GMM-Training."""
    trainer = GmmTrainer()
    result = trainer.run()

    if result is None:
        logger.error("GMM-Training fehlgeschlagen.")
        return

    X, labels, centers = result
    logger.info(f"GMM-Training erfolgreich: {len(centers)} ROIs gefunden.")


if __name__ == "__main__":
    main()
