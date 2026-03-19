import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

################################
# Konfigurationen für Feature Engineering
################################

# Welche Knochenpaare verbunden sind (Siehe Darstellung in Confluence)
BONES = [
    # Daumen
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Zeigefinger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Mittelfinger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # Ringfinger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # Kleiner Finger
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

# Referenzknochen für die Normalisierung (Handgelenk -> Erster Knochen vom Mittelfinger)
NORMALIZATION_SCALE_BONES = (0, 9)


def get_scale_factor_for_normalization(
    vectors: dict, referenz_bone: tuple
) -> np.ndarray:
    """
    Berechnung des Skalierungsfaktors basierend auf einem Referenzknochen, sodass die Features nicht von der Handgröße abhängen.

    Args:
        vectors (dict): Dictionary mit den Vektoren für jedes Knochenpaar.
        referenz_bone (tuple): Tupel, das den Referenzknochen für den Normalisierungsfaktor definiert.
    Returns:
        np.ndarray: Array mit dem Skalierungsfaktor für jedes Frame.
    """
    referenz_vector = vectors[referenz_bone]

    # Euklidische Norm (Länge) des Referenzvektors
    normalization_factor = np.linalg.norm(referenz_vector, axis=1, keepdims=True)

    # Sicherheitshalber Nullen durch kleine Zahl ersetzen (vermeidet Division durch Null)
    normalization_factor = np.where(
        normalization_factor == 0, 1e-6, normalization_factor
    )
    return normalization_factor


def calculate_vectors(keypoints: pd.DataFrame, bone_pairs: list) -> dict:
    """
    Berechnung der Vektoren zwischen den definierten Knochenpaaren.

    Args:
        keypoints (pd.DataFrame): DataFrame mit den Koordinaten der Keypoints.
        bone_pairs (list): Liste von Tupeln, welche die Knochenpaare definieren.

    Returns:
        dict: Dictionary mit den Vektoren für jedes Knochenpaar.
    """
    vectors = {}
    for point1, point2 in bone_pairs:
        # Vektor = Zielpunkt - Startpunkt
        vec = keypoints[:, point2, :] - keypoints[:, point1, :]
        vectors[(point1, point2)] = vec
    return vectors


def calculate_distance_of_bones(vectors: dict, scale_factor: np.ndarray) -> dict:
    """
    Berechnet die normalisierte Länge (Entfernung) der Knochen.

    Args:
        vectors (dict): Dictionary mit den Vektoren für jedes Knochenpaar.
        scale_factor (np.ndarray): Skalierungsfaktor für die Normalisierung.

    Returns:
        dict: Dictionary mit den normalisierten Distanzen für jedes Knochenpaar.
    """
    distances = {}
    for key, vec in vectors.items():
        # Wenn es der Referenzknochen ist, überspringen (da Länge immer 1.0 wäre)
        if key == NORMALIZATION_SCALE_BONES:
            continue

        norm = np.linalg.norm(vec, axis=1, keepdims=True)

        # Normalisieren
        distances[key] = (norm / scale_factor).flatten()
    return distances


def calculate_angles_between_vectors(
    vector1: np.ndarray, vector2: np.ndarray
) -> np.ndarray:
    """
    Berechnet den Winkel zwischen zwei Vektoren in Grad.

    Args:
        vector1 (np.ndarray): Erstes Vektor-Array (Frames, 3).
        vector2 (np.ndarray): Zweites Vektor-Array (Frames, 3).

    Returns:
        np.ndarray: Array mit Winkeln in Grad für jedes Frame.
    """
    v1_norm = np.linalg.norm(vector1, axis=1)
    v2_norm = np.linalg.norm(vector2, axis=1)

    # Division durch Null abfangen, indem Nullen durch kleine Zahl ersetzt werden
    v1_norm = np.where(v1_norm == 0, 1e-6, v1_norm)
    v2_norm = np.where(v2_norm == 0, 1e-6, v2_norm)

    dot_product = np.sum(vector1 * vector2, axis=1)
    cosine_angle = dot_product / (v1_norm * v2_norm)

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_degrees = np.degrees(np.arccos(cosine_angle))

    return angle_degrees


class FeatureExtractionController:
    """Controller für Feature-Extraktion aus Keypoint-Daten."""

    def __init__(self) -> None:
        """
        Initialisierung des Feature-Extraction-Controllers.
        """
        pass

    def process_keypoints(
        self, csv_path: str, output_folder: str | None
    ) -> tuple[str | None, pd.DataFrame | None]:
        """
        Lädt CSV-Datei mit Hand-Keypoints, berechnet die Features und speichert sie im Output Ordner.

        Args:
            csv_path (str): Pfad zur CSV-Datei mit den Hand-Keypoints.
            output_folder (str): Ordner, in dem die Feature-CSV gespeichert werden soll.

        Returns:
            Tuple of (target_path, features_df) or (None, None) on error
        """

        try:
            df_keypoints = pd.read_csv(csv_path, sep=";")
            num_frames = len(df_keypoints)
            feature_dict = {}

            # Definition der Knochenpaare für die Winkelberechnung
            fingers_map = {
                "Daumen": [(0, 1), (1, 2), (2, 3), (3, 4)],
                "Zeigefinger": [(0, 5), (5, 6), (6, 7), (7, 8)],
                "Mittelfinger": [(0, 9), (9, 10), (10, 11), (11, 12)],
                "Ringfinger": [(0, 13), (13, 14), (14, 15), (15, 16)],
                "KleinerFinger": [(0, 17), (17, 18), (18, 19), (19, 20)],
            }

            base_angles_map = [
                ("thumb_index", (0, 1), (0, 5)),
                ("index_mid", (0, 5), (0, 9)),
                ("mid_ring", (0, 9), (0, 13)),
                ("ring_pinky", (0, 13), (0, 17)),
            ]

            # Initialisiere alle Feature-Arrays mit Nullen um das Shape zu garantieren
            for hand_prefix in ["r", "l"]:
                feature_dict[f"{hand_prefix}_confidence"] = np.zeros(
                    num_frames, dtype=np.float32
                )
                for p1, p2 in BONES:
                    for coord in ["x", "y", "z"]:
                        feature_dict[f"{hand_prefix}_vec_{p1}_{p2}_{coord}"] = np.zeros(
                            num_frames, dtype=np.float32
                        )
                for fname in fingers_map:
                    for i in range(1, 4):
                        feature_dict[f"{hand_prefix}_angle_{fname}_{i}"] = np.zeros(
                            num_frames, dtype=np.float32
                        )
                for angle_name, _, _ in base_angles_map:
                    feature_dict[f"{hand_prefix}_angle_{angle_name}"] = np.zeros(
                        num_frames, dtype=np.float32
                    )

            # Iteriere über beide Hände (r=rechts, l=links)
            for hand_prefix in ["r", "l"]:
                # Vektoren für x, y, z Koordinaten generieren
                cols_x = [f"{hand_prefix}_x_{i}" for i in range(21)]
                cols_y = [f"{hand_prefix}_y_{i}" for i in range(21)]
                cols_z = [f"{hand_prefix}_z_{i}" for i in range(21)]

                # Prüfen, ob diese Hand in der CSV existiert und Daten enthält
                if not set(cols_x).issubset(df_keypoints.columns):
                    continue

                # 1. Prüfen, ob Handdaten vorhanden sind (Summe aller Koordinaten > 0)
                if (
                    df_keypoints[cols_x[0]].sum() == 0
                    and df_keypoints[cols_y[0]].sum() == 0
                ):
                    logger.debug(
                        f"Info: Keine Daten für Hand '{hand_prefix}' in {os.path.basename(csv_path)}"
                    )
                    continue

                # Erstellen eines 3D-Arrays der Keypoints -> X, Y & Z Werte hintereinander (Frames, 21, 3)
                landmarks = np.stack(
                    [
                        df_keypoints[cols_x].values,
                        df_keypoints[cols_y].values,
                        df_keypoints[cols_z].values,
                    ],
                    axis=2,
                )

                # --- FEATURE ENGINEERING START ---

                # 1. Confidence hinzufügen
                conf_col = f"{hand_prefix}_handedness_confidence"
                if conf_col in df_keypoints.columns:
                    # NaN mit 0 auffüllen, falls Hand kurz weg ist
                    feature_dict[f"{hand_prefix}_confidence"] = (
                        df_keypoints[conf_col].fillna(0).values
                    )

                # 2. Vektoren der Knochen berechnen
                vectors = calculate_vectors(landmarks, BONES)

                # 3. Skalierungsfaktor für die Normalisierung der Handgröße berechnen
                scale_factor = get_scale_factor_for_normalization(
                    vectors, NORMALIZATION_SCALE_BONES
                )

                # 4. Vektoren normalisieren
                for key, vec in vectors.items():
                    p1, p2 = key
                    vec_norm = vec / scale_factor
                    feature_dict[f"{hand_prefix}_vec_{p1}_{p2}_x"] = vec_norm[:, 0]
                    feature_dict[f"{hand_prefix}_vec_{p1}_{p2}_y"] = vec_norm[:, 1]
                    feature_dict[f"{hand_prefix}_vec_{p1}_{p2}_z"] = vec_norm[:, 2]

                ###################################
                # Aktuell deaktiviert, da Distanzen redundant zu Vektoren sind
                ###################################
                # 5. Distanzen berechnen
                # distances = calculate_distance_of_bones(vectors, scale_factor)
                # for key, dist_val in distances.items():
                #    p1, p2 = key
                #    feature_dict[f"{hand_prefix}_dist_{p1}_{p2}"] = dist_val

                # 6. Winkel berechnen für jeden Finger
                for fname, bones in fingers_map.items():
                    # Winkel 1: Zwischen Knochen 1 und 2
                    feature_dict[f"{hand_prefix}_angle_{fname}_1"] = (
                        calculate_angles_between_vectors(
                            vectors[bones[0]], vectors[bones[1]]
                        )
                    )
                    # Winkel 2: Zwischen Knochen 2 und 3
                    feature_dict[f"{hand_prefix}_angle_{fname}_2"] = (
                        calculate_angles_between_vectors(
                            vectors[bones[1]], vectors[bones[2]]
                        )
                    )
                    # Winkel 3: Zwischen Knochen 3 und 4
                    feature_dict[f"{hand_prefix}_angle_{fname}_3"] = (
                        calculate_angles_between_vectors(
                            vectors[bones[2]], vectors[bones[3]]
                        )
                    )

                # Winkel am Handgelenk zwischen den Basis-Knochen der Finger
                for angle_name, bone1, bone2 in base_angles_map:
                    feature_dict[f"{hand_prefix}_angle_{angle_name}"] = (
                        calculate_angles_between_vectors(vectors[bone1], vectors[bone2])
                    )

            # DateFrame erstellen und speichern
            features_df = pd.DataFrame(feature_dict)

            # Wenn kein Output Ordner angegeben ist, nur DataFrame zurückgeben
            if output_folder is None:
                return None, features_df

            # Output Namen generieren
            keypointCSVName = os.path.basename(csv_path)
            featureCSVName = (
                f"{keypointCSVName.replace('recording', 'features_recording')}"
            )

            # Output Ordner erstellen und Datei speichern
            targetPath = os.path.join(output_folder, featureCSVName)
            os.makedirs(output_folder, exist_ok=True)

            features_df.to_csv(targetPath, index=False, sep=";")
            logger.info(
                f"Erfolg: {targetPath} gespeichert. ({features_df.shape[1]} Features)"
            )

            return targetPath, features_df

        except Exception as e:
            logger.error(f"Fehler bei Datei {csv_path}: {e}")
            return None, None
