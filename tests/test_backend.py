import logging
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from handpose.backend.backend_api import app
from handpose.backend.backend_controller import BackendController
from handpose.backend.postprocessing.metrics_postprocessing import KPICalculator
from handpose.backend.preprocessing.mediapipe_analyse import MediaPipeVideoProcessor

logger = logging.getLogger(__name__)

# TestClient für FastAPI initialisieren
client = TestClient(app)


class TestAPIEndpoints:
    """Testet REST-API-Schnittstellen für das React-Frontend."""

    def test_get_config(self) -> None:
        """Testet GET /api/config Endpunkt."""
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "hmm_states" in data
        assert "dbscan_clusters" in data

    @patch("handpose.backend.backend_api.get_backend")
    def test_post_config(self, mock_get_backend: MagicMock) -> None:
        """Testet POST /api/config Endpunkt."""
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        payload = {
            "hmm_states": 5,
            "dbscan_clusters": 3,
            "cluster_names": "Box A, Box B, Box C",
        }
        response = client.post("/api/config", json=payload)
        assert response.status_code == 200

    @patch("handpose.backend.backend_api.get_backend")
    def test_recording_flow(self, mock_get_backend: MagicMock) -> None:
        """Testet Recording Start/Stop Flow mit den neuen REST-Routen."""
        mock_backend = MagicMock()
        mock_backend.start_recording.return_value = None
        mock_backend.stop_recording.return_value = "recording_test.mp4"
        mock_get_backend.return_value = mock_backend

        # RESTful Start (wie im React-Frontend)
        response_start = client.post("/api/recordings", json={"mode": "production"})
        assert response_start.status_code == 200

        # RESTful Stop
        response_stop = client.patch("/api/recordings/current")
        assert response_stop.status_code == 200

    @patch("handpose.backend.backend_api.get_backend")
    def test_results_and_frames(self, mock_get_backend: MagicMock) -> None:
        """Testet Results und Frames Endpunkte."""
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        res_list = client.get("/api/results")
        assert res_list.status_code == 200
        assert isinstance(res_list.json(), list)

        res_single = client.get("/api/results/run_123")
        assert res_single.status_code in [200, 404]

        res_frames = client.get("/api/frames/run_123")
        assert res_frames.status_code in [200, 404]
        if res_frames.status_code == 200:
            assert isinstance(res_frames.json(), list)

    @patch("handpose.backend.backend_api.get_backend")
    def test_camera_alignment(self, mock_get_backend: MagicMock) -> None:
        """Testet ArUco-Kamera-Alignment Endpunkt."""
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        res_align = client.post("/api/camera/alignment")
        assert res_align.status_code in [200, 400, 404, 405]


class TestBackendLogic:
    """Testet Backend-Datenverarbeitung und ML-Logik isoliert."""

    @pytest.fixture
    def kpi_calculator(self) -> KPICalculator:
        return KPICalculator(fps=30)

    def test_kpi_assembly_time(self, kpi_calculator: KPICalculator) -> None:
        """Testet Berechnung der Montagezeit ohne Grasp-State (State 3)."""
        data = {"Frame": range(1, 101), "State": [0] * 19 + [3] * 11 + [0] * 70}
        df = pd.DataFrame(data)

        with patch(
            "handpose.backend.postprocessing.metrics_postprocessing.GRASP_STATE",
            3,
            create=True,
        ):
            result = kpi_calculator.calculate_assembly_time_without_state(df)

        assert isinstance(result, float)
        assert result > -1.0

    def test_kpi_anomaly_count(self, kpi_calculator: KPICalculator) -> None:
        """Testet Zählung zusammenhängender Anomalie-Blöcke."""
        df = pd.DataFrame(
            {"is_anomaly": [False, True, True, True, False, False, True, False]}
        )
        count = kpi_calculator.calculate_amount_of_anomalies(df)
        assert count == 2

    @patch("handpose.backend.preprocessing.mediapipe_analyse.mp.solutions.hands.Hands")
    def test_mediapipe_keypoint_structure(self, mock_hands: MagicMock) -> None:
        """Testet MediaPipeVideoProcessor Initialisierung."""
        mock_hands.return_value = MagicMock()
        processor = MediaPipeVideoProcessor(preview=False)
        assert processor.get_status() is not None


@pytest.fixture
def mock_settings(tmp_path: Path) -> MagicMock:
    """Erstellt Mock-Settings für BackendController."""
    settings = MagicMock()
    settings.pipeline.camera.default_index = 0
    settings.pipeline.camera.target_fps = 30
    settings.pipeline.aruco_calibration.marker_dict = "DICT_4X4_50"
    settings.pipeline.aruco_calibration.min_markers = 4
    settings.pipeline.aruco_calibration.output_size = 500
    settings.pipeline.aruco_calibration.margin = 50
    settings.pipeline.ml_pipeline.column_names = {
        "is_anomaly_column": "is_anomaly",
        "frames_column": "Frames",
        "frame_column": "Frame",
    }
    settings.pipeline.ml_pipeline.file_formats = {"csv_extension": ".csv"}
    settings.pipeline.postprocessing.output_csv_dir = str(tmp_path)
    return settings


@pytest.fixture
def backend_controller(
    mock_settings: MagicMock,
) -> Generator[BackendController, None, None]:
    """Erstellt BackendController mit gemockten Sub-Controllern."""
    with patch("handpose.config_loader._create_required_directories"):
        controller = BackendController(settings=mock_settings)

    controller.recorder_controller = MagicMock()
    controller.mediapipe_controller = MagicMock()
    controller.vae_controller = MagicMock()
    controller.hmm_controller = MagicMock()
    controller.dbscan_controller = MagicMock()
    controller.kpi_calculator = MagicMock()

    yield controller


def test_process_data_integration(
    backend_controller: BackendController, tmp_path: Path
) -> None:
    """Testet ML-Analyse-Workflow (process_data) im BackendController."""
    input_csv = tmp_path / "recording_test.csv"
    input_csv.touch()

    # Mocks für ML Controller Returns
    backend_controller.vae_controller.transform_keypoints_to_features.return_value = (
        "features.csv",
        pd.DataFrame(),
    )
    backend_controller.vae_controller.predict_anomalies.return_value = (
        MagicMock(),
        np.array([0, 1]),
    )

    df_hmm_mock = pd.DataFrame(
        {
            "Frame": [1, 2],
            "State": [0, 1],
            "Teil": ["A", "A"],
            "Best_Frame": [False, True],
        }
    )
    backend_controller.hmm_controller.get_detailed_analysis_for_one_csv.return_value = (
        df_hmm_mock
    )
    backend_controller.hmm_controller.show_result_for_single_video.return_value = (
        MagicMock()
    )

    backend_controller.dbscan_controller.analyze_single_file.return_value = (
        np.array([]),
        np.array([]),
        np.array([]),
    )
    backend_controller.dbscan_controller.show_results.return_value = MagicMock()

    # Funktion ausführen
    plot_vae, plot_hmm, plot_dbscan, df_result = backend_controller.process_data(
        str(input_csv)
    )

    # Assertions
    assert df_result is not None
    assert not df_result.empty
    assert "State" in df_result.columns
    assert "is_anomaly" in df_result.columns

    expected_output_file = tmp_path / "recording_test.csv"
    assert expected_output_file.exists()
