"""Generic pipeline step with argument alignment and instance reuse.

This module provides a flexible adapter step capable of dynamically
instantiating classes, aligning input parameters to method signatures,
and managing state persistence across the pipeline.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from mlproject.src.pipeline.steps.core.base import BasePipelineStep
from mlproject.src.pipeline.steps.core.factory import StepFactory


class DynamicAdapterStep(BasePipelineStep):
    """Dynamic pipeline step with intelligent parameter mapping and state reuse.

    This step allows the pipeline to integrate external classes (like Scikit-learn
    estimators) dynamically by resolving class paths and mapping context data
    to specific method arguments.
    """

    def __init__(self, step_id: str, cfg: DictConfig, **kwargs: Any) -> None:
        """Initializes the dynamic step with execution metadata.

        Args:
            step_id: Unique identifier for the step.
            cfg: Global configuration object.
            **kwargs: Configuration parameters including:
                class_path: Dot-separated path to the class to instantiate.
                run_method: Method name to execute (e.g., 'fit', 'predict').
                hyperparams: Dictionary of parameters for instantiation.
                wiring: Input/Output mapping for context synchronization.
                instance_key: Key to retrieve/store an existing instance.
        """
        super().__init__(step_id, cfg, **kwargs)

        self.class_path: Optional[str] = kwargs.get("class_path")
        self.run_method: str = kwargs.get("run_method", "fit")
        self.hyperparams: dict[str, Any] = kwargs.get("hyperparams", {})
        self.wiring: dict[str, Any] = kwargs.get("wiring", {})
        self.log_artifact: bool = kwargs.get("log_artifact", False)
        self.artifact_type: str = kwargs.get("artifact_type", "component")
        self.instance_key: Optional[str] = kwargs.get("instance_key")
        self.method_params: dict[str, Any] = kwargs.get("method_params", {})
        self.output_as_feature = kwargs.get("output_as_feature", False)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the dynamic logic with automatic argument alignment.

        This method handles instance resolution, input collection from wiring,
        method signature alignment, and output propagation.

        Args:
            context: The current pipeline execution context.

        Returns:
            The updated context after method execution and output wiring.

        Raises:
            ValueError: If required wired inputs are missing from the context.
        """
        self.validate_dependencies(context)

        # Resolve or create instance
        instance = context.get(self.instance_key) if self.instance_key else None
        if instance is None:
            instance = self._instantiate()
        else:
            print(f"[{self.step_id}] Reusing instance from context.")

        # Collect wired inputs
        inputs_map: dict[str, str] = self.wiring.get("inputs", {})
        raw_inputs: dict[str, Any] = {k: context.get(v) for k, v in inputs_map.items()}

        # Validate collected inputs
        for key, value in raw_inputs.items():
            if value is None:
                raise ValueError(
                    f"Step '{self.step_id}': Input '{key}' is None. Check wiring."
                )

        # Align inputs to method signature
        method = getattr(instance, self.run_method)
        sig = inspect.signature(method)
        aligned_kwargs = self._align_arguments(raw_inputs, dict(sig.parameters))

        # Add static method parameters if provided
        if self.method_params:
            static_opts = OmegaConf.to_container(
                DictConfig(self.method_params), resolve=True
            )
            if isinstance(static_opts, dict):
                aligned_kwargs.update({str(k): v for k, v in static_opts.items()})

        # Call method
        print(f"[{self.step_id}] Calling method '{self.run_method}'.")
        result = method(**aligned_kwargs)

        # Wire outputs back to context
        outputs_map: dict[str, str] = self.wiring.get("outputs", {})
        self._wire_outputs(context, result, instance, outputs_map)

        # Register for discovery if enabled
        if self.log_artifact:
            self.register_for_discovery(context, instance)

        return context

    @staticmethod
    def _align_arguments(
        inputs: dict[str, Any],
        parameters: dict[str, inspect.Parameter],
    ) -> dict[str, Any]:
        """Aligns ML argument variants to match the target method signature.

        Resolves common variations in naming (e.g., 'X' vs 'data', 'y' vs 'labels')
        to ensure compatibility with different library conventions.

        Args:
            inputs: Raw input data from the pipeline context.
            parameters: The target method's parameters and their signatures.

        Returns:
            A dictionary of arguments aligned with the method's requirements.
        """
        aligned: dict[str, Any] = {}

        param_variants = {
            "X": ["X", "x", "data", "inputs"],
            "y": ["y", "Y", "target", "labels"],
            "sample_weight": ["sample_weight", "weights", "w"],
        }

        for param_name, param in parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                aligned.update(inputs)
                continue

            matched = False
            for variants in param_variants.values():
                if param_name in variants:
                    for v in variants:
                        if v in inputs:
                            aligned[param_name] = inputs[v]
                            matched = True
                            break
                    break

            if not matched and param_name in inputs:
                aligned[param_name] = inputs[param_name]

        return aligned

    def _instantiate(self) -> Any:
        """Dynamically imports and instantiates a class.

        Uses the `class_path` and `hyperparams` to create a new object.

        Returns:
            An instance of the specified class.

        Raises:
            AssertionError: If `class_path` is not provided.
        """
        assert self.class_path is not None

        module_path, class_name = self.class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        init_params = OmegaConf.to_container(DictConfig(self.hyperparams), resolve=True)
        if not isinstance(init_params, dict):
            init_params = {}

        return cls(**{str(k): v for k, v in init_params.items()})

    def _wire_outputs(
        self,
        context: Dict[str, Any],
        result: Any,
        instance: Any,
        mapping: dict[str, str],
    ) -> None:
        """Maps execution results or instances back into the context.

        Args:
            context: The context to update.
            result: The value returned by the method execution.
            instance: The class instance (useful for retrieving fitted models).
            mapping: A dictionary defining the target keys in the context.
        """
        if not mapping:
            context[self.step_id] = result
            return

        for output_name, context_key in mapping.items():
            if output_name == "model" and self.run_method == "fit":
                context[context_key] = instance
            else:
                context[context_key] = result


StepFactory.register("dynamic_adapter", DynamicAdapterStep)
