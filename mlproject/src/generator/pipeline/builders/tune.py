"""Tuning pipeline builder."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf

from ..step_analyzer import StepAnalyzer


class TuneBuilder:
    """Builds tuning pipeline from training pipeline."""

    def __init__(self, train_steps: List[Any]) -> None:
        """Initialize tune pipeline builder.

        Args:
            train_steps: Original training pipeline steps.
        """
        self.train_steps = train_steps
        self.analyzer = StepAnalyzer()

    def build(self) -> List[Any]:
        """Build complete tuning pipeline.

        Returns:
            List of tune pipeline steps.
        """
        steps: List[Any] = []
        tuner_map: Dict[str, str] = {}

        for step in self.train_steps:
            self._process_step(step, steps, tuner_map)

        return steps

    def _process_step(
        self, step: Any, steps: List[Any], tuner_map: Dict[str, str]
    ) -> None:
        """Process single step for tune pipeline.

        Args:
            step: Step to process.
            steps: List to append processed steps.
            tuner_map: Mapping of model_id to tuner_id.
        """
        if self.analyzer.is_model_producer(step):
            self._process_producer(step, steps, tuner_map)
            return

        if step.type == "sub_pipeline":
            self._process_sub_pipeline(step, steps, tuner_map)
            return

        if step.type == "parallel":
            self._process_parallel(step, steps, tuner_map)
            return

        if step.type == "branch":
            self._process_branch(step, steps, tuner_map)
            return

        steps.append(copy.deepcopy(step))

    def _process_producer(
        self, step: Any, steps: List[Any], tuner_map: Dict[str, str]
    ) -> None:
        """Add tuner and modified producer."""
        tuner = self._create_tuner(step)
        steps.append(tuner)

        tuner_map[step.id] = tuner.id

        modified = self._modify_for_tuning(step, tuner.id)
        steps.append(modified)

    def _create_tuner(self, producer: Any) -> DictConfig:
        """Create tuner step for model producer.

        Args:
            producer: Model producer step.

        Returns:
            Tuner step config.
        """
        tuner_id = f"tune_{producer.id}"
        depends_on = producer.depends_on if hasattr(producer, "depends_on") else []

        config = {
            "id": tuner_id,
            "type": "tuner",
            "enabled": True,
            "depends_on": depends_on,
            "target_model_id": producer.id,
        }

        return OmegaConf.create(config)

    def _modify_for_tuning(self, step: Any, tuner_id: str) -> Any:
        """Modify producer to use tuned params.

        Args:
            step: Producer step to modify.
            tuner_id: ID of tuner step.

        Returns:
            Modified step.
        """
        modified = copy.deepcopy(step)
        modified.use_tuned_params = True
        modified.tune_step_id = tuner_id

        if not hasattr(modified, "depends_on"):
            modified.depends_on = []

        if tuner_id not in modified.depends_on:
            modified.depends_on.insert(0, tuner_id)

        return modified

    def _process_sub_pipeline(
        self, step: Any, steps: List[Any], tuner_map: Dict[str, str]
    ) -> None:
        """Process sub-pipeline recursively."""
        modified = copy.deepcopy(step)

        if not hasattr(modified, "pipeline"):
            steps.append(modified)
            return

        if not hasattr(modified.pipeline, "steps"):
            steps.append(modified)
            return

        nested: List[Any] = []
        for nested_step in modified.pipeline.steps:
            self._process_step(nested_step, nested, tuner_map)

        modified.pipeline.steps = nested
        steps.append(modified)

    def _process_parallel(
        self, step: Any, steps: List[Any], tuner_map: Dict[str, str]
    ) -> None:
        """Process parallel branches."""
        modified = copy.deepcopy(step)

        if not hasattr(modified, "branches"):
            steps.append(modified)
            return

        new_branches: List[Any] = []

        for branch in modified.branches:
            if self.analyzer.is_model_producer(branch):
                tuner = self._create_tuner(branch)
                steps.append(tuner)

                tuner_map[branch.id] = tuner.id

                modified_branch = self._modify_for_tuning(branch, tuner.id)
                new_branches.append(modified_branch)
            else:
                new_branches.append(branch)

        modified.branches = new_branches
        steps.append(modified)

    def _process_branch(
        self, step: Any, steps: List[Any], tuner_map: Dict[str, str]
    ) -> None:
        """Process conditional branch."""
        modified = copy.deepcopy(step)

        for branch_name in ("if_true", "if_false"):
            if not hasattr(modified, branch_name):
                continue

            branch = getattr(modified, branch_name)

            if not self.analyzer.is_model_producer(branch):
                continue

            tuner = self._create_tuner(branch)
            steps.append(tuner)

            tuner_map[branch.id] = tuner.id

            modified_branch = self._modify_for_tuning(branch, tuner.id)
            setattr(modified, branch_name, modified_branch)

        steps.append(modified)
