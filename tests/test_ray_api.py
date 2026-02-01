from unittest.mock import MagicMock, patch

import pytest

from mlproject.serve.ray.ray_deploy import ModelService

MOCK_CONFIG = {
    "mlflow": {
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "test_exp",
        "registry": {"model_name": "test_model"},
    },
    "training": {"artifacts_dir": "/tmp/artifacts"},
    "experiment": {"hyperparams": {"input_chunk_length": 5}, "model": "xgboost"},
    "approaches": [{"model": "xgboost", "hyperparams": {}}],
}


@pytest.fixture
def mock_config_loader():
    """
    Fixture to mock ConfigLoader and provide a dummy configuration.
    """
    with patch("mlproject.serve.ray.ray_deploy.ConfigLoader") as mock_loader:
        cfg_mock = MagicMock()
        cfg_mock.training.artifacts_dir = "/tmp/artifacts"
        cfg_mock.experiment.hyperparams.get.return_value = 5
        cfg_mock.approaches = MOCK_CONFIG["approaches"]
        mock_loader.load.return_value = cfg_mock
        yield mock_loader


def test_model_service_init_local(mock_config_loader):
    """
    Test initializing ModelService in local fallback mode when MLflow is disabled.
    """
    with patch("mlproject.serve.ray.ray_deploy.MLflowManager") as mock_mlflow, patch(
        "mlproject.serve.ray.ray_deploy.ModelFactory"
    ) as mock_factory:
        mock_mlflow.return_value.enabled = False

        service = ModelService.func_or_class()

        mock_factory.load.assert_called_once()
        assert service.is_loaded()
        assert service.get_run_id() is None
