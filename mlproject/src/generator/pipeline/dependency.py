"""Dependency resolution utilities for pipeline transformation."""

from __future__ import annotations

from typing import Any, List, Optional, Set

from .step_analyzer import StepExtractor


class DependencyBuilder:
    """Builds and resolves step dependencies."""

    def __init__(self, train_steps: List[Any]) -> None:
        """Initialize dependency builder.

        Args:
            train_steps: Original training pipeline steps.
        """
        self.train_steps = train_steps
        self.extractor = StepExtractor()

    def build_eval_dependencies(
        self,
        step: Any,
        init_id: str,
        preprocessor_id: Optional[str],
        producer_ids: Set[str],
    ) -> List[str]:
        """Build dependencies for evaluator step.

        Args:
            step: Original model producer step.
            init_id: Initialization step ID.
            preprocessor_id: Preprocessor step ID if exists.
            producer_ids: Set of all model producer IDs.

        Returns:
            List of dependency step IDs.
        """
        deps = [init_id]

        if hasattr(step, "depends_on") and step.depends_on:
            for dep in step.depends_on:
                if not self._is_valid_dependency(dep, producer_ids):
                    continue

                parent = self.extractor.find_parent_sub_pipeline(self.train_steps, dep)

                if parent and parent not in deps:
                    deps.append(parent)
                elif not parent and dep not in deps:
                    deps.append(dep)

        if preprocessor_id and preprocessor_id not in deps:
            deps.append(preprocessor_id)

        return deps

    def resolve_dependencies(
        self,
        step: Any,
        producer_ids: Set[str],
    ) -> List[str]:
        """Recursively resolve and filter dependencies.

        Args:
            step: Step to resolve dependencies for.
            producer_ids: Set of model producer IDs to filter out.

        Returns:
            List of resolved dependency IDs.
        """
        resolved: List[str] = []
        visited: Set[str] = set()

        def resolve_recursive(dep: str) -> None:
            if dep in visited:
                return

            visited.add(dep)

            if self._is_valid_dependency(dep, producer_ids):
                resolved.append(dep)
                return

            for train_step in self.train_steps:
                if train_step["id"] != dep:
                    continue

                if not train_step.get("depends_on"):
                    continue

                for child_dep in train_step["depends_on"]:
                    resolve_recursive(child_dep)

        if not getattr(step, "depends_on", None):
            return []

        for dep in step.depends_on:
            resolve_recursive(dep)

        return resolved

    def _is_valid_dependency(self, dep: str, producer_ids: Set[str]) -> bool:
        """Check if dependency should be kept.

        Args:
            dep: Dependency step ID.
            producer_ids: Set of model producer IDs.

        Returns:
            True if dependency is valid for eval/serve mode.
        """
        if dep.startswith("tune_"):
            return False

        if dep in producer_ids:
            return False

        if not self.train_steps:
            return True

        step = self.extractor.find_step_by_id(self.train_steps, dep)
        if not step:
            return True

        invalid_types = ["datamodule", "trainer", "clustering", "framework_model"]
        return step.type not in invalid_types
