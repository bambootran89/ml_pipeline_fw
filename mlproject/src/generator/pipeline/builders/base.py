"""Base pipeline builder with shared functionality.

This module provides common functionality for building evaluation and serving
pipelines from training configurations.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ...config import ConfigurablePatternMatcher, GeneratorConfig
from ...constants import CONTEXT_KEYS, STEP_CONSTANTS
from ..dependency import DependencyBuilder
from ..step_analyzer import StepAnalyzer, StepExtractor
from ..step_transformer import StepTransformer


class BasePipelineBuilder:
    """Base class for pipeline builders with common functionality.

    This class provides shared methods used by both EvalBuilder and ServeBuilder
    to avoid code duplication and ensure consistent behavior.
    """

    def __init__(
        self,
        train_steps: List[Any],
        experiment_type: str = "tabular",
        config: Optional[GeneratorConfig] = None,
    ) -> None:
        """Initialize base pipeline builder.

        Args:
            train_steps: Original training pipeline steps.
            experiment_type: Type of experiment (timeseries, tabular).
            config: Generator configuration for customization.
        """
        self.train_steps = train_steps
        self.experiment_type = experiment_type
        self.config = config or GeneratorConfig()
        self.matcher = ConfigurablePatternMatcher(self.config)
        self.analyzer = StepAnalyzer()
        self.extractor = StepExtractor()
        self.dependency_builder = DependencyBuilder(train_steps)
        self.transformer = StepTransformer()

    def find_datamodule_for_step(self, step: Any) -> Optional[Any]:
        """Find datamodule step used by a given step.

        This method searches for the datamodule that provides data to the given step
        by examining the step's wiring configuration.

        Args:
            step: Pipeline step to find datamodule for.

        Returns:
            Datamodule step if found, None otherwise.
        """
        # Check if step has datamodule input in wiring
        if not hasattr(step, "wiring") or not hasattr(step.wiring, "inputs"):
            return None

        datamodule_key = step.wiring.inputs.get(STEP_CONSTANTS.DATAMODULE)
        if not datamodule_key:
            return None

        # Find step that outputs this datamodule key
        return self._find_step_by_output_key(STEP_CONSTANTS.DATAMODULE, datamodule_key)

    def extract_additional_feature_keys(self, step: Any) -> Optional[List[str]]:
        """Extract additional_feature_keys from datamodule used by step.

        Args:
            step: Pipeline step to extract feature keys for.

        Returns:
            List of additional feature keys if found, None otherwise.
        """
        datamodule_step = self.find_datamodule_for_step(step)
        if datamodule_step is None:
            return None

        if hasattr(datamodule_step, "additional_feature_keys"):
            keys = datamodule_step.additional_feature_keys
            if keys:
                return list(keys)

        return None

    def should_apply_windowing(self, step_id: str) -> bool:
        """Determine if windowing should be applied for timeseries step.

        Args:
            step_id: ID of the step to check.

        Returns:
            True if windowing should be applied, False otherwise.
        """
        if self.experiment_type != "timeseries":
            return False

        # Find the original training step to check its configuration
        train_step = self.extractor.find_step_by_id(self.train_steps, step_id)
        if not train_step:
            return True

        step_type = train_step.get("type", "")

        # Use configurable pattern matcher for windowing decision
        return self.matcher.should_apply_windowing(step_type, step_id)

    def _find_step_by_output_key(
        self, step_type: str, output_key: str
    ) -> Optional[Any]:
        """Find step by its output key.

        Args:
            step_type: Type of step to search for.
            output_key: Output key to match.

        Returns:
            Step if found, None otherwise.
        """
        for step in self.train_steps:
            if step.type != step_type:
                continue

            if hasattr(step, "wiring") and hasattr(step.wiring, "outputs"):
                outputs = step.wiring.outputs
                if outputs.get(step_type) == output_key:
                    return step

        return None

    def is_clustering_adapter(self, step: Any) -> bool:
        """Check if step is a clustering dynamic adapter.

        Args:
            step: Step to check.

        Returns:
            True if step is a clustering adapter.
        """
        if step.type != STEP_CONSTANTS.DYNAMIC_ADAPTER:
            return False

        if not hasattr(step, "class_path"):
            return False

        path_lower = step.class_path.lower()
        return "cluster" in path_lower or "kmeans" in path_lower

    def resolve_base_features_key(self, step: Any) -> str:
        """Resolve base features key from step wiring.

        Args:
            step: Step to extract features key from.

        Returns:
            Base features key string.
        """

        if hasattr(step, "wiring") and hasattr(step.wiring, "inputs"):
            inputs = step.wiring.inputs
            return self.matcher.resolve_feature_key(inputs)
        return CONTEXT_KEYS.PREPROCESSED_DATA
