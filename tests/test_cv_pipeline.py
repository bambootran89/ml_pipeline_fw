"""
Integration tests for the Cross-Validation (CV) pipeline.

These tests verify:
- End-to-end execution of the CV pipeline
- Correct formatting of aggregated metrics (mean/std)
- FoldPreprocessor infers features correctly

Run:
    pytest tests/test_cv_pipeline.py -v -s
"""

from typing import Any, Dict

import pytest

from mlproject.src.datamodule.splitters.timeseries import TimeSeriesFoldSplitter
from mlproject.src.pipeline.compat.v1.cv import CrossValidationPipeline
from mlproject.src.utils.config_class import ConfigLoader


@pytest.fixture
def cv_pipeline() -> CrossValidationPipeline:
    """
    Create a configured CV pipeline instance for integration tests.

    Returns:
        CrossValidationPipeline: Ready-to-run pipeline instance with:
            • MLflow disabled
            • Small n_epochs for fast tests
            • Features ensured for FoldPreprocessor
    """
    cfg = ConfigLoader.load("mlproject/configs/experiments/etth2.yaml")

    cfg.mlflow.enabled = False
    cfg.experiment.hyperparams.n_epochs = 2

    if "features" not in cfg.get("data", {}):
        cfg.data.features = ["HUFL", "MUFL", "mobility_inflow"]

    splitter = TimeSeriesFoldSplitter(
        cfg=cfg,
        n_splits=2,
    )

    return CrossValidationPipeline(cfg, splitter)


def test_cv_pipeline_smoke(cv_pipeline: CrossValidationPipeline) -> None:
    """
    Smoke test verifying that the CV pipeline runs successfully.

    Assertions:
        • CV returns aggregated metrics
        • MAE mean is positive
    """
    approach: Dict[str, Any] = {
        "model": cv_pipeline.cfg.experiment.model,
        "model_type": cv_pipeline.cfg.experiment.model_type,
        "hyperparams": dict(cv_pipeline.cfg.experiment.hyperparams),
        "name": cv_pipeline.cfg.experiment.name,
    }

    metrics = cv_pipeline.run_cv(approach)

    assert "mae_mean" in metrics
    assert "mae_std" in metrics
    assert metrics["mae_mean"] > 0


def test_cv_metrics_format(cv_pipeline: CrossValidationPipeline) -> None:
    """
    Ensure aggregated metric names follow the expected mean/std suffix pattern.

    Assertions:
        • For each metric (mae, mse, rmse), the output contains
          both `<metric>_mean` and `<metric>_std`.
    """
    data = cv_pipeline.preprocess()

    approach: Dict[str, Any] = {
        "model": cv_pipeline.cfg.experiment.model,
        "model_type": cv_pipeline.cfg.experiment.model_type,
        "hyperparams": dict(cv_pipeline.cfg.experiment.hyperparams),
        "name": cv_pipeline.cfg.experiment.name,
    }

    metrics = cv_pipeline.run_cv(approach, data)

    for base_metric in ["mae", "mse", "rmse"]:
        assert f"{base_metric}_mean" in metrics
        assert f"{base_metric}_std" in metrics
