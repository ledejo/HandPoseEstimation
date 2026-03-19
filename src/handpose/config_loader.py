import logging
import logging.config
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _get_root_dir() -> Path:
    """
    Bestimmt das Projekt-Root-Verzeichnis robust mittels Projekt-Markern.
    """
    if env_root := os.getenv("HANDPOSE_ROOT"):
        root = Path(env_root).resolve()
        logger.debug(f"ROOT_DIR aus HANDPOSE_ROOT: {root}")
        return root

    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            logger.debug(f"ROOT_DIR detektiert via pyproject.toml: {parent}")
            return parent

    fallback_root = Path(__file__).resolve().parents[2]
    logger.debug(f"ROOT_DIR aus __file__-relativ: {fallback_root}")
    return fallback_root


ROOT_DIR = _get_root_dir()


# --- Pipeline Config Models ---
class AppSettings(BaseModel):
    allowed_origins: list[str]
    logging_config: str
    logging_output_dir: str = "logs"


class CameraConfig(BaseModel):
    camera_type: int = 1
    target_fps: int
    retry_attempts: int
    retry_delay_sec: float
    recording_dir_production: str
    recording_dir_training: str


class ArucoCalibrationConfig(BaseModel):
    marker_dict: str
    min_markers: int
    output_size: int
    margin: int


class FrameExtractionConfig(BaseModel):
    every_nth_frame: int
    supported_video_formats: list[str]
    supported_frame_formats: list[str]


class PreprocessingConfig(BaseModel):
    frame_extraction: FrameExtractionConfig
    keypoints_dir_production: str
    keypoints_dir_training: str
    features_dir_production: str
    features_dir_training: str


class MLPipelineControl(BaseModel):
    models: dict[str, bool]
    column_names: dict[str, str]
    file_formats: dict[str, str]


class PostprocessingConfig(BaseModel):
    analysed_videos_dir: str
    extracted_frames_dir: str
    output_csv_dir: str
    plots_dir: str
    kpi_settings: dict[str, int]
    plot_filenames: dict[str, str]


class PipelineConfig(BaseModel):
    app: AppSettings
    camera: CameraConfig
    aruco_calibration: ArucoCalibrationConfig
    preprocessing: PreprocessingConfig
    ml_pipeline: MLPipelineControl
    postprocessing: PostprocessingConfig


# --- Process Config Model ---
class ProcessConfig(BaseModel):
    hmm_n_states: int = 6
    dbscan_n_clusters: int = 6
    cluster_names: str = "Teil1, Teil2, Teil3, Teil4"


class HmmConfig(BaseModel):
    paths: dict[str, str]
    n_states: int = 6
    n_iter: int = 500
    random_state: int = 42
    grasp_state: int = 3
    roi_centers: list[list[float]]  # Nur Koordinaten
    roi_names: list[str] = []  # Wird aus process.yaml geladen
    rois: dict[str, list[float]] = {}  # Wird dynamisch generiert
    roi_threshold: float
    roi_rad_x: float
    roi_rad_y: float
    min_y_for_grasp: float

    def get_rois_as_numpy(self) -> dict[str, np.ndarray]:
        return {k: np.array(v) for k, v in self.rois.items()}


class VaeConfig(BaseModel):
    device: str = "cuda"
    paths: dict[str, str]
    window_size: int = 30
    stride: int = 5
    validation_split: float = 0.2
    hidden_dim: int = 256
    latent_dim: int = 32
    input_dim: int = 158
    dropout: float = 0.075
    batch_size: int = 32
    epochs: int = 500
    patience: int = 30
    learning_rate: float
    kl_weight: float
    n_trials: int = 30
    threshold: float = 0.01

    @property
    def actual_device(self) -> str:
        return self.device if torch.cuda.is_available() else "cpu"

    @property
    def model_save_path(self) -> str:
        return f"{self.paths['model_save']}/vae_model_SW{self.window_size}.pth"


class DbscanConfig(BaseModel):
    paths: dict[str, str]
    eps: float = 0.01
    min_samples: int = 50
    frames_to_take: int = 5
    feature_index_max_reach_y: int = 2


class Settings(BaseModel):
    pipeline: PipelineConfig
    hmm: HmmConfig
    vae: VaeConfig
    dbscan: DbscanConfig
    process: ProcessConfig = ProcessConfig()


def load_settings() -> Settings:
    config_dir = ROOT_DIR / "configs"

    def _load(rel_path: str) -> dict:
        file = config_dir / f"{rel_path}.yaml"
        if not file.exists():
            if "process" in rel_path:
                return {}
            raise FileNotFoundError(f"Missing Config: {file}")
        with open(file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    loaded = {
        "pipeline": _load("pipeline"),
        "hmm": _load("models/hmm"),
        "vae": _load("models/vae"),
        "dbscan": _load("models/dbscan"),
        "process": _load("process"),
    }

    # ROI-Namen aus process.yaml + Koordinaten aus hmm.yaml kombinieren
    process_data = loaded.get("process", {})
    hmm_data = loaded.get("hmm", {})

    if "cluster_names" in process_data and "roi_centers" in hmm_data:
        names = [name.strip() for name in process_data["cluster_names"].split(",")]
        centers = hmm_data["roi_centers"]

        # Kombiniere Namen mit Koordinaten zu rois Dictionary
        hmm_data["roi_names"] = names
        hmm_data["rois"] = {name: center for name, center in zip(names, centers)}

    return Settings.model_validate(loaded)


@lru_cache()
def get_settings() -> Settings:
    """Gibt die gecachten Settings zurück.

    Diese Funktion ist mit @lru_cache() dekoriert für den Einsatz
    als FastAPI-Abhängigkeit (Depends(get_settings)).

    Returns:
        Gecachte Settings-Instanz.
    """
    return load_settings()


def save_process_config(process_data: ProcessConfig):
    file_path = ROOT_DIR / "configs" / "process.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(process_data.model_dump(), f, default_flow_style=False)

    get_settings.cache_clear()
    logger.info("Process Config in process.yaml gespeichert.")


def save_camera_type(camera_type: int) -> None:
    """Speichert camera_type in pipeline.yaml."""
    file_path = ROOT_DIR / "configs" / "pipeline.yaml"
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    camera = data.setdefault("camera", {})
    camera["camera_type"] = int(camera_type)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    get_settings.cache_clear()
    logger.info(f"Camera type gespeichert: {camera_type}")


def _collect_all_paths(obj: BaseModel | dict) -> list[str]:
    """
    Sammelt Pfad-Strings aus einer Konfiguration.
    Nur Feldnamen, die mit '_dir' oder '_path' enden, werden als Pfade behandelt.
    """
    paths = []
    items = obj.model_dump().items() if isinstance(obj, BaseModel) else obj.items()

    for field_name, field_value in items:
        if isinstance(field_value, str) and (
            field_name.endswith("_dir") or field_name.endswith("_path")
        ):
            paths.append(field_value)
        elif isinstance(field_value, dict):
            for sub_key, sub_value in field_value.items():
                if isinstance(sub_value, str) and (
                    sub_key.endswith("_dir") or sub_key.endswith("_path")
                ):
                    paths.append(sub_value)
    return paths


def _create_required_directories() -> None:
    """Erstellt alle erforderlichen Verzeichnisse aus der Konfiguration."""
    settings_obj = get_settings()
    all_paths = _collect_all_paths(settings_obj)

    for dir_path in set(filter(None, all_paths)):  # set() für Duplikate entfernen
        path = Path(dir_path) if isinstance(dir_path, str) else dir_path
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Verzeichnis erstellt: {path}")


def get_absolute_path(rel_path: str | Path) -> Path:
    return ROOT_DIR / rel_path


def setup_logging() -> None:
    s = get_settings()
    path = ROOT_DIR / s.pipeline.app.logging_config
    if path.exists():
        with open(path, "rt", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f.read())
                if "handlers" in config and "file" in config["handlers"]:
                    log_dir = ROOT_DIR / s.pipeline.app.logging_output_dir
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_file = log_dir / "handpose.log"
                    config["handlers"]["file"]["filename"] = str(log_file)
                logging.config.dictConfig(config)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Logging-Config: {e}")
                logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)
        logger.warning(f"Logging-Config nicht gefunden: {path}")
