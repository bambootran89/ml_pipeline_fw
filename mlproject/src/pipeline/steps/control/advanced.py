"""Advanced pipeline steps for complex workflows.

This module provides steps for:
- Parallel execution of multiple branches
- Conditional branching based on context values
- Sub-pipeline execution as single step
"""

from __future__ import annotations

import operator
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from mlproject.src.pipeline.steps.core.base import PipelineStep
from mlproject.src.pipeline.steps.core.constants import DefaultValues
from mlproject.src.pipeline.steps.core.factory import StepFactory


class ParallelStep(PipelineStep):
    """Execute multiple step branches in parallel.

    Supports additional_feature_keys for composing features before
    passing to branches.
    """

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        branches: Optional[List[Dict[str, Any]]] = None,
        max_workers: Optional[int] = None,
        additional_feature_keys: Optional[List[str]] = None,
        feature_align_method: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Initializes the parallel execution step.

        Args:
            step_id: Unique identifier for the step.
            cfg: Global configuration object.
            enabled: Whether the step is active. Defaults to True.
            depends_on: List of step IDs this step relies on.
            branches: List of step configurations to run in parallel.
            max_workers: Maximum number of threads. Defaults to global setting.
            additional_feature_keys: Keys to compose before execution.
            feature_align_method: Strategy for aligning features.
            **kwargs: Additional metadata for the base class.
        """
        super().__init__(
            step_id,
            cfg,
            enabled,
            depends_on,
            additional_feature_keys=additional_feature_keys,
            feature_align_method=feature_align_method,
            **kwargs,
        )
        self.branches = branches or []
        self.max_workers = (
            max_workers if max_workers is not None else DefaultValues.MAX_WORKERS
        )

    def _execute_branch(
        self,
        branch_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Creates and executes a single branch step.

        Args:
            branch_config: Configuration dictionary for the branch step.
            context: The current execution context.

        Returns:
            A dictionary containing the branch ID and its execution result.
        """
        branch_ctx = deepcopy(context)
        step = StepFactory.create(branch_config, self.cfg)
        result = step.execute(branch_ctx)
        return {"branch_id": branch_config.get("id", "unknown"), "result": result}

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Runs configured branches in parallel and merges their results.

        Args:
            context: The input pipeline context.

        Returns:
            The context updated with merged results from all parallel branches.

        Raises:
            Exception: If any branch execution fails.
        """
        self.validate_dependencies(context)

        if not self.branches:
            print(f"[{self.step_id}] No branches configured, skipping")
            return context

        print(
            f"[{self.step_id}] Executing {len(self.branches)} branches "
            f"(max_workers={self.max_workers})"
        )

        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._execute_branch, branch, context): branch
                for branch in self.branches
            }
            for future in as_completed(futures):
                branch = futures[future]
                branch_id = branch.get("id", "unknown")
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[{self.step_id}] Branch '{branch_id}' completed")
                except Exception as e:
                    print(f"[{self.step_id}] Branch '{branch_id}' failed: {e}")
                    raise
        merged = self._merge_results(context, results)
        print(f"[{self.step_id}] All branches completed")
        return merged

    def _merge_results(
        self, context: Dict[str, Any], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merges results from multiple branches into a single context.

        Args:
            context: The original context to merge into.
            results: List of result dictionaries from parallel branches.

        Returns:
            A new context dictionary containing merged values.
        """
        merged = context.copy()
        for result in results:
            for key, value in result["result"].items():
                if key not in context:
                    merged[key] = value
                else:
                    if isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key].update(value)
                    else:
                        merged[key] = value

        return merged


class BranchStep(PipelineStep):
    """Conditional execution based on context values.

    Supports additional_feature_keys for composing features before
    condition evaluation.
    """

    OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
        ">": operator.gt,
        "<": operator.lt,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        "<=": operator.le,
        "in": lambda a, b: a in b,
        "not_in": lambda a, b: a not in b,
    }

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        condition: Optional[Dict[str, Any]] = None,
        if_true: Optional[Dict[str, Any]] = None,
        if_false: Optional[Dict[str, Any]] = None,
        additional_feature_keys: Optional[List[str]] = None,
        feature_align_method: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Initializes the conditional branching step.

        Args:
            step_id: Unique identifier for the step.
            cfg: Global configuration object.
            enabled: Whether the step is active.
            depends_on: List of step IDs this step relies on.
            condition: Dict containing 'key', 'operator', and 'value'.
            if_true: Step configuration to run if condition is met.
            if_false: Step configuration to run if condition fails.
            additional_feature_keys: Keys to compose before evaluation.
            feature_align_method: Strategy for aligning features.
            **kwargs: Additional metadata.
        """
        super().__init__(
            step_id,
            cfg,
            enabled,
            depends_on,
            additional_feature_keys=additional_feature_keys,
            feature_align_method=feature_align_method,
            **kwargs,
        )
        self.condition = condition or {}
        self.if_true = if_true
        self.if_false = if_false

    def _evaluate_condition(self, context: Dict[str, Any]) -> bool:
        """Checks if the defined condition evaluates to True.

        Args:
            context: The current pipeline context.

        Returns:
            True if the condition is met, False otherwise.

        Raises:
            ValueError: If an unsupported operator is provided.
        """
        if not self.condition:
            return True

        key = self.condition.get("key", "")
        op_name = self.condition.get("operator", "==")
        expected = self.condition.get("value")
        actual = context.get(key)

        if op_name not in self.OPERATORS:
            raise ValueError(f"Unknown operator: {op_name}")

        try:
            result = self.OPERATORS[op_name](actual, expected)
        except TypeError:
            result = False

        print(
            f"[{self.step_id}] Condition: {key} {op_name} {expected} "
            f"(actual={actual}) -> {result}"
        )
        return result

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates the condition and executes the corresponding branch.

        Args:
            context: The pipeline context.

        Returns:
            The resulting context from the executed branch or the original
            context if no branch matches.
        """
        self.validate_dependencies(context)
        condition_result = self._evaluate_condition(context)

        if condition_result and self.if_true:
            print(f"[{self.step_id}] Executing TRUE branch")
            step = StepFactory.create(self.if_true, self.cfg)
            return step.execute(context)
        elif not condition_result and self.if_false:
            print(f"[{self.step_id}] Executing FALSE branch")
            step = StepFactory.create(self.if_false, self.cfg)
            return step.execute(context)
        else:
            print(f"[{self.step_id}] No branch to execute")
            return context


class SubPipelineStep(PipelineStep):
    """Execute a nested pipeline as a single step.

    Supports additional_feature_keys for composing features before
    passing to sub-pipeline.
    """

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        pipeline: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        isolated: bool = False,
        additional_feature_keys: Optional[List[str]] = None,
        feature_align_method: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Initializes the sub-pipeline execution step.

        Args:
            step_id: Unique identifier for the step.
            cfg: Parent pipeline configuration.
            enabled: Whether the step is active.
            depends_on: List of dependency IDs.
            pipeline: In-line sub-pipeline configuration.
            config_path: External path to a sub-pipeline YAML.
            isolated: If True, only explicit input_keys are passed to sub-pipeline.
            additional_feature_keys: Keys to compose before execution.
            feature_align_method: Strategy for aligning features.
            **kwargs: Additional metadata.
        """
        super().__init__(
            step_id,
            cfg,
            enabled,
            depends_on,
            additional_feature_keys=additional_feature_keys,
            feature_align_method=feature_align_method,
            **kwargs,
        )
        self.pipeline_config = pipeline
        self.config_path = config_path
        self.isolated = isolated

    def _build_sub_executor(self):
        """Builds a PipelineExecutor for the nested pipeline.

        Returns:
            A PipelineExecutor instance.

        Raises:
            ValueError: If neither 'pipeline' nor 'config_path' is provided.
        """
        from mlproject.src.pipeline.executor import (  # pylint: disable=C0415
            PipelineExecutor,
        )

        if self.config_path:
            sub_cfg = OmegaConf.load(self.config_path)
            merged = OmegaConf.merge(self.cfg, sub_cfg)
        elif self.pipeline_config:
            sub_cfg = OmegaConf.create({"pipeline": self.pipeline_config})
            merged = OmegaConf.merge(self.cfg, sub_cfg)
        else:
            raise ValueError(
                f"Step '{self.step_id}' requires 'pipeline' or 'config_path'."
            )

        if not isinstance(merged, DictConfig):
            container = OmegaConf.to_container(merged, resolve=True)
            merged = OmegaConf.create(container)

        return PipelineExecutor(merged)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the sub-pipeline and merges results.

        Args:
            context: The current pipeline context.

        Returns:
            The context updated with results from the sub-pipeline execution.
        """
        self.validate_dependencies(context)
        print(f"[{self.step_id}] Executing sub-pipeline")

        sub_ctx = {} if self.isolated else context.copy()

        for local_name, source_key in self.input_keys.items():
            if source_key in context:
                sub_ctx[local_name] = context[source_key]

        executor = self._build_sub_executor()
        sub_result = executor.execute(initial_context=sub_ctx)

        merged = context.copy()
        for local_name, target_key in self.output_keys.items():
            if local_name in sub_result:
                merged[target_key] = sub_result[local_name]

        for key, value in sub_result.items():
            if key not in context and key not in self.output_keys:
                merged[key] = value
        print(f"[{self.step_id}] Sub-pipeline completed")
        return merged


# Register step types
StepFactory.register("parallel", ParallelStep)
StepFactory.register("branch", BranchStep)
StepFactory.register("sub_pipeline", SubPipelineStep)
