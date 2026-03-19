import logging
import os
from pathlib import Path

import cv2
import pandas as pd

from handpose.config_loader import get_settings

logger = logging.getLogger(__name__)


def extract_all_frames(video_path: str | Path, output_dir: str = None) -> int:
    """
    Extrahiert ALLE Frames aus einem Video und speichert sie als PNG-Bilder.

    Args:
        video_path (str | Path): Pfad zum Video (mp4).
        output_dir (str, optional): Zielverzeichnis. Falls None, wird aus run_id abgeleitet.

    Returns:
        int: Anzahl der gespeicherten Frames.
    """
    video_path_obj = Path(video_path)
    run_id = video_path_obj.stem  # z.B. "recording_20251202_213913"

    # Defaultwert für output_dir
    if output_dir is None:
        settings = get_settings()
        output_dir = (
            Path(settings.pipeline.postprocessing.extracted_frames_dir) / run_id
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Video konnte nicht geöffnet werden: {video_path}")
        return 0

    frame_index = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Speichere ALLE Frames
        filename = f"frame_{frame_index:05d}.png"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), frame)
        saved_count += 1
        frame_index += 1

    cap.release()
    logger.info(f"✅ {saved_count} Frames extrahiert nach {output_dir}")
    return saved_count


def extract_frames(
    video_path: str | Path, output_dir: str, every_nth_frame: int = 1
) -> int:
    """
    Extrahiert NUR die Best Frames aus einem Video und speichert sie als PNG-Bilder.

    Erwartete Struktur (aus Config):
        Video: .../data/02_intermediate/analysed_videos/<run_id>.mp4
        CSV:   .../data/05_results/metrics_csv/<run_id>.csv

    CSV-Spalten:
        Frame, State, Teil, Best_Frame, is_anomaly

    Args:
        video_path (str): Pfad zum Video (mp4).
        output_dir (str): Ordner, in den die Frames gespeichert werden.
        every_nth_frame (int): Wird aktuell ignoriert (Best Frames werden immer alle gespeichert).

    Returns:
        int: Anzahl der tatsächlich gespeicherten Best Frames.
    """

    video_path_obj = Path(video_path)
    run_id = video_path_obj.stem  # z.B. "recording_20251202_213913"

    # Nutze Konfiguration für CSV-Pfad
    csv_dir = Path(get_settings().pipeline.postprocessing.output_csv_dir)
    csv_path = csv_dir / f"{run_id}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Keine KPI-CSV für '{run_id}' gefunden: {csv_path}")

    # CSV laden
    df = pd.read_csv(csv_path)

    if "Frame" not in df.columns or "Best_Frame" not in df.columns:
        raise ValueError(
            f"CSV {csv_path} enthält nicht die benötigten Spalten 'Frame' und 'Best_Frame'."
        )

    # Best Frames bestimmen
    best_frames_df = df[df["Best_Frame"].astype(bool)]

    if best_frames_df.empty:
        logger.info(
            f"Keine Best Frames markiert für '{run_id}'. Extrahiere stattdessen alle State-3 Frames..."
        )
        # Fallback: Extrahiere alle GRASP_STATE Frames
        grasp_frames_df = df[df["State"] == 3]  # State 3 ist GRASP_STATE

        if grasp_frames_df.empty:
            logger.info(
                f"Keine GRASP_STATE Frames gefunden. Extrahiere jeden {every_nth_frame}. Frame..."
            )
            # Zweiter Fallback: Extrahiere jeden n-ten Frame
            best_frame_indices = list(range(0, len(df), every_nth_frame))
            best_frame_set = set(best_frame_indices)
            max_best_frame = max(best_frame_indices) if best_frame_indices else 0
        else:
            best_frame_indices = sorted(
                int(f) for f in grasp_frames_df["Frame"].unique()
            )
            best_frame_set = set(best_frame_indices)
            max_best_frame = max(best_frame_indices)
    else:
        # eindeutige, sortierte Frame-Indizes
        best_frame_indices = sorted(int(f) for f in best_frames_df["Frame"].unique())
        best_frame_set = set(best_frame_indices)
        max_best_frame = max(best_frame_indices)

    # Zielordner anlegen
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_index = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Nur speichern, wenn dieser Frame ein Best Frame ist
        if frame_index in best_frame_set:
            filename = f"frame_{frame_index:05d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1

        # Optionaler Early-Exit: Wenn wir den letzten Best Frame überschritten haben, können wir abbrechen
        if frame_index > max_best_frame:
            break

        frame_index += 1

    cap.release()
    return saved_count
