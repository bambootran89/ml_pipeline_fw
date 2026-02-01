import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.append(os.getcwd())


@hydra.main(
    version_base=None,
    config_path="../mlproject/configs/pipeline_dag",
    config_name="train",
)
# ^ Pointing to a valid config entry point. 'dag_run' uses configs/pipelines/...
# But here we simulate what 'dag_run train' does: loads experiment + pipeline.
# To keep it simple, we just load experiment config primarily.


def main(cfg: DictConfig):
    # Logic matches MLflowManager
    mlflow_cfg = cfg.get("mlflow", {})

    # Resolve Experiment Name
    experiment_name = mlflow_cfg.get(
        "experiment_name",
        cfg.get("experiment", {}).get("name", "default_experiment"),
    )

    # Resolve Model Name
    registry_cfg = mlflow_cfg.get("registry", {})
    model_name = registry_cfg.get("model_name", "model")

    print(experiment_name)
    print(model_name)


if __name__ == "__main__":
    main()
