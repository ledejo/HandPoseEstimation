import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml

from handpose.config_loader import get_settings, ROOT_DIR

from ..HMM.model_hmm import HmmWrapper
from .features_dbscan import DbscanFeatureExtractor

logger = logging.getLogger(__name__)


def _append_state_values(
    states: npt.NDArray[np.int_],
    max_reach_y_values: npt.NDArray[np.float64],
    state_to_values: dict[int, list[float]],
) -> None:
    """Überführt max_reach_y-Werte in den Sammelcontainer pro State.

    Args:
        states: Vorhergesagte HMM-States pro Frame.
        max_reach_y_values: max_reach_y-Feature pro Frame.
        state_to_values: Zielcontainer je State-ID.
    """
    for state in np.unique(states):
        mask = states == state
        values = max_reach_y_values[mask]
        if values.size == 0:
            continue

        state_key = int(state)
        state_to_values.setdefault(state_key, []).extend(values.tolist())


def detect_grasp_state_by_mean_max_reach_y(
    raw_data_dir: Path,
    hmm_model_dir: Path,
    feature_index_max_reach_y: int = 2,
) -> tuple[int | None, dict[int, float], int]:
    """Erkennt den Grasp-State über den höchsten Mittelwert von max_reach_y.

    Ablauf:
        1. Trainings-CSV-Dateien laden.
        2. Features erzeugen und HMM-States vorhersagen.
        3. max_reach_y je State aggregieren.
        4. State mit höchstem Mittelwert als Grasp-State wählen.

    Args:
        raw_data_dir: Verzeichnis mit Trainings-CSV-Dateien.
        hmm_model_dir: Verzeichnis mit HMM-Modell und Scaler.
        feature_index_max_reach_y: Index des max_reach_y Features im Feature-Array.

    Returns:
        Tuple mit:
            - erkanntem Grasp-State oder None,
            - Mittelwerten pro State,
            - Anzahl erfolgreich verarbeiteter Dateien.
    """
    extractor = DbscanFeatureExtractor()
    hmm_wrapper = HmmWrapper()

    model_path = hmm_model_dir / "hmm_model.pkl"
    scaler_path = hmm_model_dir / "scaler.pkl"
    hmm_wrapper.load(model_path, scaler_path)

    csv_files = sorted(raw_data_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("Keine Trainingsdateien gefunden in %s", raw_data_dir)
        return None, {}, 0

    state_to_values: dict[int, list[float]] = {}
    files_used = 0

    for csv_path in csv_files:
        try:
            dataframe = extractor.process_csv(csv_path)
            if dataframe is None:
                logger.debug("Datei übersprungen (keine Daten): %s", csv_path.name)
                continue

            features, _ = extractor.calculate_features(dataframe)
            if len(features) == 0:
                logger.debug("Datei übersprungen (leere Features): %s", csv_path.name)
                continue

            states = hmm_wrapper.predict(features)
            if len(states) != len(features):
                logger.warning(
                    "Inkonsistente Länge bei %s (states=%d, features=%d)",
                    csv_path.name,
                    len(states),
                    len(features),
                )
                continue

            max_reach_y_values = features[:, feature_index_max_reach_y]
            _append_state_values(
                states=states,
                max_reach_y_values=max_reach_y_values,
                state_to_values=state_to_values,
            )
            files_used += 1

        except (KeyError, ValueError) as error:
            logger.warning("Ungültige Daten in %s: %s", csv_path.name, error)
        except RuntimeError as error:
            logger.warning("Modellfehler bei %s: %s", csv_path.name, error)

    if not state_to_values:
        logger.warning("Keine State-Daten für Grasp-State-Erkennung gesammelt.")
        return None, {}, files_used

    state_means = {
        state: float(np.mean(values))
        for state, values in state_to_values.items()
        if values
    }
    if not state_means:
        logger.warning(
            "State-Daten vorhanden, aber keine gültigen Mittelwerte berechnet."
        )
        return None, {}, files_used

    detected_state = max(state_means, key=state_means.get)

    logger.info("Automatische Grasp-State-Erkennung (mean_max_reach_y):")
    for state, mean_value in sorted(state_means.items()):
        logger.info("  state=%s -> mean_max_reach_y=%.6f", state, mean_value)
    logger.info("Erkannter Grasp-State: %s", detected_state)

    return detected_state, state_means, files_used


def update_hmm_grasp_state(grasp_state: int) -> None:
    """Aktualisiert den Grasp-State in der HMM-Konfiguration.

    Args:
        grasp_state: Zu speichernder Grasp-State.
    """
    config_path = ROOT_DIR / "configs" / "models" / "hmm.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config["grasp_state"] = int(grasp_state)

    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    get_settings.cache_clear()
    logger.info("HMM-Config aktualisiert: grasp_state=%s", grasp_state)


def detect_and_persist_grasp_state() -> int | None:
    """Erkennt den Grasp-State und persistiert ihn in der HMM-Config.

    Returns:
        Erkannter Grasp-State oder None bei Fehlschlag.
    """
    settings = get_settings()

    raw_data_dir = Path(settings.dbscan.paths.get("raw_data_dir", ""))
    hmm_model_dir = Path(settings.hmm.paths.get("model_dir", ""))
    feature_index = settings.dbscan.feature_index_max_reach_y

    try:
        grasp_state, _, files_used = detect_grasp_state_by_mean_max_reach_y(
            raw_data_dir=raw_data_dir,
            hmm_model_dir=hmm_model_dir,
            feature_index_max_reach_y=feature_index,
        )
    except (FileNotFoundError, ValueError) as error:
        logger.error("Erkennung fehlgeschlagen: %s", error)
        return None

    if grasp_state is None:
        logger.warning(
            "Automatische Grasp-State-Erkennung fehlgeschlagen (files_used=%s).",
            files_used,
        )
        return None

    try:
        update_hmm_grasp_state(grasp_state)
    except (FileNotFoundError, OSError, yaml.YAMLError) as error:
        logger.error("Persistierung fehlgeschlagen: %s", error)
        return None

    return grasp_state
