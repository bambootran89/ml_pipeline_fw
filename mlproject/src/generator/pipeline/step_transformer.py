"""Step transformation utilities for pipeline modes."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


class StepTransformer:
    """Transforms pipeline steps for different execution modes."""

    @staticmethod
    def setup_for_load_mode(step: Any, alias: str) -> None:
        """Configure step for eval/serve mode loading.

        Args:
            step: Step to configure.
            alias: Model alias for loading.
        """
        step.is_train = False
        step.alias = alias
        step.instance_key = f"fitted_{step.id}"

    @staticmethod
    def remove_training_configs(step: Any) -> None:
        """Remove training-only configuration attributes.

        Args:
            step: Step to clean up.
        """
        for attr in ["log_artifact", "artifact_type", "hyperparams"]:
            if hasattr(step, attr):
                delattr(step, attr)

        if step.type != "inference":
            return

        wiring = getattr(step, "wiring", None)
        if wiring is None:
            return

        inputs = getattr(wiring, "inputs", None)
        if inputs is None:
            return

        if hasattr(inputs, "datamodule"):
            delattr(inputs, "datamodule")

    @staticmethod
    def setup_model_wiring(step: Any) -> None:
        """Setup wiring for model/clustering step in eval/serve.

        Args:
            step: Step to setup wiring for.
        """
        if not hasattr(step, "wiring"):
            step.wiring = OmegaConf.create({})

        if not hasattr(step.wiring, "inputs"):
            step.wiring.inputs = OmegaConf.create({})

        if not hasattr(step.wiring, "outputs"):
            step.wiring.outputs = OmegaConf.create({})

        step.wiring.inputs.model = f"fitted_{step.id}"
        step.wiring.inputs.features = "preprocessed_data"

        step.wiring.outputs.features = (
            step.wiring.outputs.features
            if hasattr(step.wiring.outputs, "features")
            else "cluster_features"
        )
        step.wiring.outputs.model = f"fitted_{step.id}"

    @classmethod
    def transform_preprocessor(cls, step: Any, alias: str) -> None:
        """Transform preprocessor for eval/serve mode.

        Args:
            step: Preprocessor step to transform.
            alias: Model alias for loading.
        """
        cls.setup_for_load_mode(step, alias)
        cls.remove_training_configs(step)

        if hasattr(step, "wiring") and "inputs" in step.wiring:
            delattr(step.wiring, "inputs")

    @classmethod
    def transform_model_step(cls, step: Any) -> None:
        """Transform model/clustering step for eval/serve.

        Args:
            step: Model step to transform.
        """
        cls.setup_model_wiring(step)
        cls.remove_training_configs(step)

    @staticmethod
    def extract_base_name(step_id: str) -> str:
        """Extract base name from model producer step ID.

        Args:
            step_id: Full step ID.

        Returns:
            Base name without suffixes.
        """
        return step_id.replace("_features", "").replace("_model", "")
