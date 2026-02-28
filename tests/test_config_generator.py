from pathlib import Path

import pytest
from omegaconf import OmegaConf

from mlproject.src.generator.orchestrator import ConfigGenerator


@pytest.fixture
def temp_train_cfg(tmp_path: Path) -> str:
    cfg = {
        "experiment": {"name": "test_exp", "type": "tabular", "model": "test_model"},
        "pipeline": {
            "name": "train_pipeline",
            "steps": [
                {"id": "load_data", "type": "data_loader"},
                {"id": "preprocess", "type": "preprocessor"},
                {"id": "train_model", "type": "trainer"},
            ],
        },
        "data": {"features": ["a"], "type": "tabular"},
        "preprocessing": {"steps": []},
    }
    p = tmp_path / "test_exp.yaml"
    with open(p, "w") as f:
        OmegaConf.save(OmegaConf.create(cfg), f)
    return str(p)


def test_config_generator(temp_train_cfg, tmp_path):
    generator = ConfigGenerator(train_config_path=temp_train_cfg)

    out_dir = str(tmp_path / "generated")
    paths = generator.generate_all(output_dir=out_dir, include_tune=True)

    assert "eval" in paths
    assert "serve" in paths
    assert "tune" in paths

    eval_cfg = OmegaConf.load(paths["eval"])
    serve_cfg = OmegaConf.load(paths["serve"])
    tune_cfg = OmegaConf.load(paths["tune"])

    assert eval_cfg.pipeline.name == "test_exp_eval"
    assert serve_cfg.pipeline.name == "test_exp_serve"
    assert tune_cfg.pipeline.name == "test_exp_tune"

    # Check that in eval/serve mode, preprocessing is_train is False
    assert eval_cfg.preprocessing.is_train is False
    assert serve_cfg.preprocessing.is_train is False

    assert (
        sum(1 for s in eval_cfg.pipeline.steps if s.get("type") == "mlflow_loader") > 0
    )
    assert (
        sum(1 for s in serve_cfg.pipeline.steps if s.get("type") == "mlflow_loader") > 0
    )
    assert sum(1 for s in tune_cfg.pipeline.steps if s.get("type") == "tuner") > 0
