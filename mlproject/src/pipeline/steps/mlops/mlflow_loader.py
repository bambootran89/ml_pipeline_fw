"""
Unified MLflow artifact loader step for multi-component pipelines.

Responsibilities:
- Restore multiple fitted pipeline components from MLflow Model Registry.
- Inject restored instances into pipeline execution context using wiring keys.
- Skip execution gracefully if MLflow tracking is disabled.
- Maintain strict typing and formatting for static analysis quality gates.

Quality gates:
- Compatible with mypy static type checking.
- Maximum line length limited to 88 characters.
- Pylint-compatible code style and structure.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mlproject.src.pipeline.steps.core.base import BasePipelineStep
from mlproject.src.pipeline.steps.core.constants import DefaultValues
from mlproject.src.pipeline.steps.core.factory import StepFactory
from mlproject.src.tracking.mlflow_manager import MLflowManager

logger = logging.getLogger(__name__)


class MLflowLoaderStep(BasePipelineStep):
    """Restore multiple fitted components from MLflow into pipeline context."""

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        alias: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MLflow loader with registry alias and component map."""
        super().__init__(step_id, cfg, enabled, depends_on)
        self.mlflow_manager: MLflowManager = MLflowManager(self.cfg)
        self.alias: str = alias if alias is not None else DefaultValues.MLFLOW_ALIAS
        self.load_map: List[Dict[str, str]] = kwargs.get("load_map", [])

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Load registered MLflow artifacts into the pipeline execution context."""
        if not self.mlflow_manager.enabled:
            logger.info(f"[{self.step_id}] MLflow disabled. Skipping artifact loader.")
            return context

        exp_name: str = self.cfg.experiment.name
        logger.info(
            f"[{self.step_id}] Loading MLflow artifacts for experiment: {exp_name}"
        )

        for entry in self.load_map:
            logger.info("*" * 100, entry)
            source_id: Optional[str] = entry.get("step_id")
            target_key: Optional[str] = entry.get("context_key")

            if not source_id or not target_key:
                continue

            registry_name: str = f"{exp_name}_{source_id}"
            logger.info(f"[{self.step_id}] Retrieving registry entry: {registry_name}")

            component: Any = self.mlflow_manager.load_component(
                name=registry_name,
                alias=self.alias,
            )

            if component is not None:
                context[target_key] = component
                logger.info(
                    f"[{self.step_id}] Injected artifact into context as: {target_key}"
                )
            else:
                logger.info(
                    f"[{self.step_id}] Warning: Artifact '{registry_name}' "
                    "could not be restored from MLflow."
                )

        return context


StepFactory.register("mlflow_loader", MLflowLoaderStep)
