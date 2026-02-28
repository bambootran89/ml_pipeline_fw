from pathlib import Path

import pytest
from omegaconf import OmegaConf

from mlproject.src.pipeline.steps.core.utils import ConfigMerger as StepsMerger
from mlproject.src.utils.config_class import ConfigMerger as CoreMerger


@pytest.fixture
def temp_experiment_cfg(tmp_path: Path) -> str:
    cfg = {
        "experiment": {"name": "test_exp", "type": "test_type", "model": "test_model"},
        "data": {"features": ["a"]},
        "preprocessing": {"steps": []},
    }
    p = tmp_path / "exp.yaml"
    with open(p, "w") as f:
        OmegaConf.save(OmegaConf.create(cfg), f)
    return str(p)


@pytest.fixture
def temp_pipeline_cfg(tmp_path: Path) -> str:
    cfg = {"pipeline": {"steps": [{"id": "step1"}]}}
    p = tmp_path / "pipe.yaml"
    with open(p, "w") as f:
        OmegaConf.save(OmegaConf.create(cfg), f)
    return str(p)


def test_core_config_merger(temp_experiment_cfg, temp_pipeline_cfg, tmp_path):
    # Test merge
    merged_cfg = CoreMerger.merge(temp_experiment_cfg, temp_pipeline_cfg, mode="train")
    assert "experiment" in merged_cfg
    assert "pipeline" in merged_cfg
    assert merged_cfg.experiment.name == "test_exp"
    assert merged_cfg.pipeline.steps[0].id == "step1"

    # Test save
    out_path = tmp_path / "merged_out.yaml"
    CoreMerger.save(merged_cfg, str(out_path))
    assert out_path.exists()

    loaded = OmegaConf.load(out_path)
    assert loaded.experiment.name == "test_exp"


def test_steps_config_merger(temp_experiment_cfg, tmp_path):
    base_cfg = OmegaConf.create({"base": "value"})

    # Test merge_external_file
    merged_file = StepsMerger.merge_external_file(base_cfg, temp_experiment_cfg)
    assert "base" in merged_file
    assert "experiment" in merged_file

    # Test merge_inline_config
    inline_cfg = {"inline": "val"}
    merged_inline = StepsMerger.merge_inline_config(base_cfg, inline_cfg)
    assert merged_inline.inline == "val"
    assert "base" in merged_inline

    # Test apply_simple_overrides
    merged_overrides = StepsMerger.apply_simple_overrides(base_cfg, new_key="new_val")
    assert merged_overrides.new_key == "new_val"

    # Test merge_model_config
    merged_model = StepsMerger.merge_model_config(
        base_cfg, model_name="new_model", hyperparams={"lr": 0.01}
    )
    assert merged_model.experiment.model == "new_model"
    assert merged_model.experiment.hyperparams.lr == 0.01
