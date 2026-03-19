import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


_raw_data_cache = None
_cache_key = None


class HandPoseDataset(Dataset):
    """PyTorch Dataset für Hand-Pose Sequenzen mit Sliding-Window-Mechanismus."""

    def __init__(self, features_dir: str, window_size: int, stride: int) -> None:
        """
        Initialisierung des HandPose-Datasets.

        Args:
            features_dir (str): Pfad zum Features-Verzeichnis.
            window_size (int): Länge der Sliding Windows.
            stride (int): Schrittweite für Sliding Windows.
        """
        self.features_dir = Path(features_dir)
        self.window_size = window_size
        self.stride = stride
        self.data_windows: list[np.ndarray] = []

        raw_data_list = self._load_raw_data_cached(self.features_dir)
        self._create_windows(raw_data_list)

    def preprocess_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Bereinigt und normalisiert einen DataFrame.

        Args:
            df (pd.DataFrame): Eingabe DataFrame.

        Returns:
            np.ndarray: Bereinigtes und normalisiertes numpy Array.
        """
        cols_to_drop = [
            c
            for c in df.columns
            if "timestamp" in c.lower() or "confidence" in c.lower()
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        df = df.fillna(0.0)
        df = df.replace([np.inf, -np.inf], 0.0)

        angle_cols = [c for c in df.columns if "angle" in c or "abd" in c]
        if angle_cols:
            df[angle_cols] = df[angle_cols] / 180.0

        df = df.select_dtypes(include=[np.number])
        return df.values.astype(np.float32)

    def _load_raw_data_cached(self, features_dir) -> list:
        """
        Lädt Rohdaten aus dem Features-Ordner mit Caching.

        Args:
            features_dir: Path oder str zum Features-Ordner.

        Returns:
            list: Liste von numpy Arrays mit Rohdaten.
        """
        global _raw_data_cache, _cache_key
        features_path = Path(features_dir)

        if _raw_data_cache is not None and _cache_key == features_path:
            return _raw_data_cache

        logger.info(f"Lade Dataset von: {features_path} ...")
        files = list(features_path.glob("*.csv"))

        raw_data = []
        valid_files = 0

        for f in files:
            try:
                df = pd.read_csv(f, sep=";")
                data_array = self.preprocess_dataframe(df)

                if len(data_array) > 0:
                    raw_data.append(data_array)
                    valid_files += 1

            except Exception as e:
                logger.error(f"Fehler bei Datei {f.name}: {e}")

        _raw_data_cache = raw_data
        _cache_key = features_path

        if raw_data:
            logger.info(
                f"{valid_files} Dateien geladen. Features: {raw_data[0].shape[1]}"
            )
        else:
            logger.warning("Keine validen Daten gefunden.")

        return raw_data

    def _create_windows(self, raw_data_list: list) -> None:
        """
        Erstellt Sliding Windows aus den Rohdaten.

        Args:
            raw_data_list (list): Liste von numpy Arrays mit Rohdaten.
        """
        for data in raw_data_list:
            n_samples = len(data)
            if n_samples < self.window_size:
                continue

            for i in range(0, n_samples - self.window_size + 1, self.stride):
                window = data[i : i + self.window_size]
                self.data_windows.append(window)

    def __len__(self) -> int:
        return len(self.data_windows)

    def __getitem__(self, idx: int) -> tuple:
        """
        Ruft ein Daten-Window ab.

        Args:
            idx (int): Index des Windows.

        Returns:
            tuple: (x, x) für VAE-Training.
        """
        window = self.data_windows[idx]
        x = torch.from_numpy(window)
        return x, x
