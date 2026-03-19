import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def upsample_keypoint_csv(filepath: str, target_fps: int = 30) -> None:
    """
    Upsampled eine CSV-Datei mit Hand-Keypoints auf die gewünschte Bildrate mittels Zeitinterpolation.

    Args:
        filepath (str): Pfad zur Eingabe-CSV-Datei.
        target_fps (int): Ziel-Bildrate (z.B. 30 oder 60 FPS). Standard: 30.

    Returns:
        None: Speichert Datei direkt auf Disk.
    """
    # Datei laden
    df = pd.read_csv(filepath, sep=";")

    # Timestamp konvertieren
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("datetime")

    # Frequenz setzen
    interval_us = int(1_000_000 / target_fps)
    freq = f"{interval_us}us"

    # Bestimme den neuen Index mithilfe der Timestamps und der Frequenz
    start = df.index.min()
    end = df.index.max()
    new_index = pd.date_range(start=start, end=end, freq=freq)

    # Interpoliere die Daten auf den neuen Index
    df_upsampled = df.reindex(df.index.union(new_index)).interpolate(method="time")
    df_final = df_upsampled.loc[new_index].reset_index()

    # Timestamp zurückrechnen
    df_final = df_final.rename(columns={"index": "timestamp_new"})
    df_final["timestamp"] = df_final["timestamp_new"].astype(np.int64) / 10**9
    df_final = df_final.drop(columns=["timestamp_new"])

    # Speichere die Datei ab
    filename = os.path.basename(filepath)
    outputfolder = "data/03_processed/keypoints/production"
    target_filepath = os.path.join(
        outputfolder, f"{filename.replace('.csv', f'_{target_fps}fps.csv')}"
    )
    df_final.to_csv(target_filepath, sep=";", index=False)
    logger.info(f"✅ Interpoliert und gespeichert: {filename}")


def main() -> None:
    output_dir = "data/03_processed/keypoints/production"

    try:
        files = os.listdir(output_dir)
        if not files:
            logger.warning(f"Keine Dateien in {output_dir} gefunden")
            return

        for file in files:
            upsample_keypoint_csv(os.path.join(output_dir, file), target_fps=30)
            logger.info(f"Fertig interpoliert! Datei: {file}")

        logger.info(f"✅ Alle {len(files)} Dateien verarbeitet")

    except Exception as e:
        logger.error(f"❌ Fehler während Verarbeitung: {e}")


if __name__ == "__main__":
    main()
