"""Pipeline executor with dependency resolution.

Enhanced to support runtime context pre-initialization for serving mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from mlproject.src.pipeline.steps.core.base import BasePipelineStep
from mlproject.src.pipeline.steps.core.factory import StepFactory


class PipelineExecutor:
    """Execute pipeline steps in dependency order.

    This executor:
    - Resolves step dependencies using topological sort
    - Manages shared context across steps
    - Handles step enablement and skipping
    - Supports runtime context pre-initialization (serving mode)
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize pipeline executor.

        Parameters
        ----------
        cfg : DictConfig
            Full configuration including pipeline definition.
        """
        self.cfg = cfg
        self.steps: List[BasePipelineStep] = []
        self._build_steps()

    def _build_steps(self) -> None:
        """Build step instances from configuration."""
        pipeline_cfg = self.cfg.get("pipeline", {})
        step_configs = pipeline_cfg.get("steps", [])

        for step_config in step_configs:
            step = StepFactory.create(step_config, self.cfg)
            self.steps.append(step)

        print(f"[Pipeline] Built {len(self.steps)} steps")

    def _topological_sort(self) -> List[BasePipelineStep]:
        """Sort steps by dependencies using Kahn's algorithm.

        Returns
        -------
        List[BasePipelineStep]
            Steps in execution order.

        Raises
        ------
        RuntimeError
            If circular dependencies detected.
        """
        step_map = {s.step_id: s for s in self.steps}
        in_degree = {s.step_id: 0 for s in self.steps}
        adjacency: Dict[str, List[str]] = {s.step_id: [] for s in self.steps}

        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_map:
                    raise RuntimeError(
                        f"Step '{step.step_id}' depends on unknown step '{dep}'"
                    )
                adjacency[dep].append(step.step_id)
                in_degree[step.step_id] += 1

        # Kahn's algorithm
        queue = [s.step_id for s in self.steps if in_degree[s.step_id] == 0]
        sorted_ids: List[str] = []

        while queue:
            current_id = queue.pop(0)
            sorted_ids.append(current_id)

            for neighbor_id in adjacency[current_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        if len(sorted_ids) != len(self.steps):
            raise RuntimeError("Circular dependency detected in pipeline")

        return [step_map[step_id] for step_id in sorted_ids]

    def execute(
        self, initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute all enabled steps in dependency order.

        Parameters
        ----------
        initial_context : Dict[str, Any], optional
            Pre-initialized context for serving mode.

        Returns
        -------
        Dict[str, Any]
            Final pipeline context with all outputs.
        """
        sorted_steps = self._topological_sort()

        context: Dict[str, Any] = initial_context.copy() if initial_context else {}

        print("[Pipeline] Execution order:")
        for i, step in enumerate(sorted_steps, 1):
            status = "enabled" if step.enabled else "disabled"
            print(f"  {i}. {step.step_id} ({status})")

        print("\n[Pipeline] Executing steps...")

        for step in sorted_steps:
            if not step.enabled:
                print(f"[Pipeline] Skipping disabled step: {step.step_id}")
                continue

            print(f"\n[Pipeline] Executing: {step.step_id}")
            context = step.execute(context)

        print("\n[Pipeline] Execution complete")
        return context
