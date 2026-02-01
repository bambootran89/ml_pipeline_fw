"""Factory for creating pipeline steps from configuration.

This module provides a factory pattern for creating pipeline steps.
Step classes are registered via register() method to avoid circular imports.

Import Order Solution:
1. factory.py defines StepFactory với empty registry
2. Các step files import factory và tự register
3. __init__.py import tất cả steps → triggers registration
"""

from __future__ import annotations

from typing import Any, Dict, Type

from omegaconf import DictConfig

from mlproject.src.pipeline.steps.core.base import BasePipelineStep


class StepFactory:
    """Factory for creating pipeline step instances.

    Step classes register themselves via register() method.
    This avoids circular imports by not importing step classes directly.

    Basic Types
    -----------
    - data_loader: Load data from source
    - preprocessor: Transform features
    - trainer: Train model (legacy)
    - evaluator: Evaluate model
    - inference: Generate predictions
    - logger: Log to MLflow
    - tuner: Hyperparameter optimization

    Advanced Types
    --------------
    - parallel: Execute branches concurrently
    - branch: Conditional execution
    - sub_pipeline: Nested pipeline
    - clustering: Clustering với auto output_as_feature
    """

    # Registry - populated by step classes calling register()
    _STEP_REGISTRY: Dict[str, Type[BasePipelineStep]] = {}

    @classmethod
    def create(cls, step_config: Dict[str, Any], cfg: DictConfig) -> BasePipelineStep:
        """Create a pipeline step from configuration.

        Parameters
        ----------
        step_config : Dict[str, Any]
            Step configuration with keys:
            - id: str (required)
            - type: str (required)
            - enabled: bool (optional, default=True)
            - depends_on: List[str] (optional)
            - wiring: Dict (optional)
            - **kwargs: Step-specific parameters
        cfg : DictConfig
            Full experiment configuration.

        Returns
        -------
        BasePipelineStep
            Instantiated step.

        Raises
        ------
        ValueError
            If step type is unknown.
        """
        step_id = step_config["id"]
        step_type = step_config["type"]
        enabled = step_config.get("enabled", True)
        depends_on = step_config.get("depends_on", [])

        if step_type not in cls._STEP_REGISTRY:
            raise ValueError(
                f"Unknown step type: '{step_type}'. "
                f"Available: {list(cls._STEP_REGISTRY.keys())}. "
                f"Make sure all step modules are imported."
            )

        step_class = cls._STEP_REGISTRY[step_type]

        # Extract step-specific kwargs
        reserved_keys = {"id", "type", "enabled", "depends_on"}
        kwargs = {k: v for k, v in step_config.items() if k not in reserved_keys}

        return step_class(
            step_id=step_id,
            cfg=cfg,
            enabled=enabled,
            depends_on=depends_on,
            **kwargs,
        )

    @classmethod
    def register(cls, type_name: str, step_class: Type[BasePipelineStep]) -> None:
        """Register a step type.

        Called by step classes to register themselves.

        Parameters
        ----------
        type_name : str
            Type string for YAML configuration.
        step_class : Type[BasePipelineStep]
            Step class to register.

        Examples
        --------
        At the end of each step file::

            # trainer_step.py
            class TrainerStep(BasePipelineStep):
                ...

            StepFactory.register("trainer", TrainerStep)
        """
        cls._STEP_REGISTRY[type_name] = step_class

    @classmethod
    def available_types(cls) -> list:
        """Get list of available step types.

        Returns
        -------
        list
            Sorted list of registered step type names.
        """
        return sorted(cls._STEP_REGISTRY.keys())
