"""
Test tuning components.

Run:
    pytest tests/test_tuning.py -v -s
"""

import pytest

from mlproject.src.datamodule.splitters.timeseries import TimeSeriesFoldSplitter
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.tuning.optuna import OptunaTuner
from mlproject.src.utils.config_class import ConfigLoader


@pytest.mark.slow
def test_optuna_tuner_smoke():
    """
    Smoke test for Optuna tuner.

    Mark as slow because it runs actual tuning.
    Run: pytest tests/test_tuning.py::test_optuna_tuner_smoke -v -s
    """
    cfg = ConfigLoader.load("mlproject/configs/experiments/etth3_tuning.yaml")

    # Disable MLflow
    cfg.mlflow.enabled = False

    # Fast test settings
    cfg.experiment.hyperparams.n_epochs = 1

    splitter = TimeSeriesFoldSplitter(
        cfg,
        n_splits=2,
    )
    mlflow_manager = MLflowManager(cfg)
    tuner = OptunaTuner(
        cfg=cfg,
        splitter=splitter,
        mlflow_manager=mlflow_manager,
        metric_name="mae_mean",
        direction="minimize",
    )

    # Run 2 trials only
    result = tuner.tune(n_trials=2, show_progress=False)

    assert "best_params" in result
    assert "best_value" in result
    assert "study" in result

    # Check best_params có hyperparameters
    assert len(result["best_params"]) > 0
