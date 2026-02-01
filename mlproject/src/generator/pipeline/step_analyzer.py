"""Step analysis utilities for pipeline transformation."""

from __future__ import annotations

from typing import Any, List, Optional, Set


class StepAnalyzer:
    """Analyzes and categorizes pipeline steps."""

    MODEL_PRODUCER_TYPES = {"trainer", "clustering", "framework_model"}
    SPECIAL_TYPES = {"sub_pipeline", "branch", "parallel"}

    @classmethod
    def is_model_producer(cls, step: Any) -> bool:
        """Check if step produces a model artifact.

        Args:
            step: Pipeline step to check.

        Returns:
            True if step is a model producer.
        """
        if not hasattr(step, "type"):
            return False

        if step.type in cls.MODEL_PRODUCER_TYPES:
            return True

        if step.type == "dynamic_adapter":
            return hasattr(step, "artifact_type") and step.artifact_type == "model"

        return False

    @classmethod
    def is_preprocessor(cls, step: Any) -> bool:
        """Check if step is a preprocessor.

        Args:
            step: Pipeline step to check.

        Returns:
            True if step is a preprocessor.
        """
        if step.type == "preprocessor":
            return True

        if step.type == "dynamic_adapter":
            return hasattr(step, "artifact_type") and step.artifact_type == "preprocess"

        return False

    @classmethod
    def is_clustering(cls, step: Any) -> bool:
        """Check if step is a clustering model.

        Args:
            step: Pipeline step to check.

        Returns:
            True if step is clustering model.
        """
        if step.type == "clustering":
            return True

        if step.type == "dynamic_adapter" and hasattr(step, "class_path"):
            path_lower = step.class_path.lower()
            return "cluster" in path_lower or "kmeans" in path_lower

        return False

    @classmethod
    def infer_experiment_type(cls, steps: List[Any]) -> str:
        """Infer experiment type (timeseries/tabular) from steps.

        Args:
            steps: Pipeline steps.

        Returns:
            Inferred experiment type.
        """
        for step in steps:
            # Check datamodule for timeseries markers
            if step.get("type") == "datamodule":
                if "input_chunk_length" in step or "output_chunk_length" in step:
                    return "timeseries"
                if "wiring" in step and "inputs" in step.wiring:
                    # Some datamodules might have these in wiring or nested in cfg
                    pass

            # Check trainer for timeseries markers in hyperparams
            if step.get("type") == "trainer" and "hyperparams" in step:
                hp = step.hyperparams
                if "input_chunk_length" in hp or "output_chunk_length" in hp:
                    return "timeseries"

            # Check nested sub-pipelines
            if step.get("type") == "sub_pipeline" and hasattr(step, "pipeline"):
                if hasattr(step.pipeline, "steps"):
                    nested_type = cls.infer_experiment_type(step.pipeline.steps)
                    if nested_type == "timeseries":
                        return "timeseries"

        return "tabular"

    @classmethod
    def extract_features_input(cls, step: Any) -> str:
        """Extract features input key from step wiring.

        Args:
            step: Pipeline step with potential wiring config.

        Returns:
            Input key for features data.
        """
        if not hasattr(step, "wiring") or "inputs" not in step.wiring:
            return "preprocessed_data"

        inputs = step.wiring.inputs

        if "X" in inputs:
            return inputs["X"]

        if "features" in inputs:
            return inputs["features"]

        return "preprocessed_data"


class StepExtractor:
    """Extracts specific step types from pipeline."""

    def __init__(self) -> None:
        """Initialize step extractor."""
        self.analyzer = StepAnalyzer()

    def extract_model_producers(self, steps: List[Any]) -> List[Any]:
        """Recursively extract all model producer steps.

        Args:
            steps: List of pipeline steps to search.

        Returns:
            List of model producer steps.
        """
        producers: List[Any] = []

        for step in steps:
            if self.analyzer.is_model_producer(step):
                producers.append(step)
            elif step.type == "sub_pipeline":
                producers.extend(self._from_sub_pipeline(step))
            elif step.type == "branch":
                producers.extend(self._from_branch(step))
            elif step.type == "parallel":
                producers.extend(self._from_parallel(step))

        return producers

    def extract_preprocessors(self, steps: List[Any]) -> List[Any]:
        """Recursively extract all preprocessor steps.

        Args:
            steps: List of pipeline steps to search.

        Returns:
            List of preprocessor steps.
        """
        preprocessors: List[Any] = []

        for step in steps:
            if self.analyzer.is_preprocessor(step):
                preprocessors.append(step)
            elif step.type == "sub_pipeline":
                nested = self._from_sub_pipeline_preprocess(step)
                preprocessors.extend(nested)

        return preprocessors

    def find_step_by_id(self, steps: List[Any], step_id: str) -> Optional[Any]:
        """Find step by ID recursively.

        Args:
            steps: List of steps to search.
            step_id: Step ID to find.

        Returns:
            Found step or None.
        """
        for step in steps:
            if hasattr(step, "id") and step.id == step_id:
                return step

            if step.type == "sub_pipeline":
                found = self._search_sub_pipeline(step, step_id)
                if found:
                    return found
            elif step.type == "branch":
                found = self._search_branch(step, step_id)
                if found:
                    return found
            elif step.type == "parallel":
                found = self._search_parallel(step, step_id)
                if found:
                    return found

        return None

    def find_parent_sub_pipeline(self, steps: List[Any], step_id: str) -> Optional[str]:
        """Find parent sub-pipeline ID containing step.

        Args:
            steps: Top-level pipeline steps.
            step_id: ID of step to find parent for.

        Returns:
            Parent sub-pipeline ID or None.
        """
        for step in steps:
            if step.type != "sub_pipeline":
                continue

            if not hasattr(step, "pipeline"):
                continue

            if not hasattr(step.pipeline, "steps"):
                continue

            for nested in step.pipeline.steps:
                if hasattr(nested, "id") and nested.id == step_id:
                    return step.id

        return None

    def collect_producer_ids(self, producers: Optional[List[Any]]) -> Set[str]:
        """Collect model producer step IDs.

        Args:
            producers: List of model producer steps.

        Returns:
            Set of producer IDs.
        """
        if not producers:
            return set()

        return {p.id for p in producers}

    def _from_sub_pipeline(self, step: Any) -> List[Any]:
        """Extract producers from sub-pipeline step."""
        if not hasattr(step, "pipeline"):
            return []

        if not hasattr(step.pipeline, "steps"):
            return []

        return self.extract_model_producers(step.pipeline.steps)

    def _from_branch(self, step: Any) -> List[Any]:
        """Extract producers from branch step."""
        producers = []

        for branch_name in ["if_true", "if_false"]:
            if not hasattr(step, branch_name):
                continue

            branch = getattr(step, branch_name)
            if self.analyzer.is_model_producer(branch):
                producers.append(branch)

        return producers

    def _from_parallel(self, step: Any) -> List[Any]:
        """Extract producers from parallel step."""
        producers: List[Any] = []

        if not hasattr(step, "branches"):
            return producers

        for branch in step.branches:
            if self.analyzer.is_model_producer(branch):
                producers.append(branch)

        return producers

    def _from_sub_pipeline_preprocess(self, step: Any) -> List[Any]:
        """Extract preprocessors from sub-pipeline."""
        if not hasattr(step, "pipeline"):
            return []

        if not hasattr(step.pipeline, "steps"):
            return []

        return self.extract_preprocessors(step.pipeline.steps)

    def _search_sub_pipeline(self, step: Any, step_id: str) -> Optional[Any]:
        """Search for step in sub-pipeline."""
        if not hasattr(step, "pipeline"):
            return None

        if not hasattr(step.pipeline, "steps"):
            return None

        return self.find_step_by_id(step.pipeline.steps, step_id)

    def _search_branch(self, step: Any, step_id: str) -> Optional[Any]:
        """Search for step in branch."""
        for branch_name in ["if_true", "if_false"]:
            if not hasattr(step, branch_name):
                continue

            branch = getattr(step, branch_name)
            if hasattr(branch, "id") and branch.id == step_id:
                return branch

        return None

    def _search_parallel(self, step: Any, step_id: str) -> Optional[Any]:
        """Search for step in parallel branches."""
        if not hasattr(step, "branches"):
            return None

        for branch in step.branches:
            if hasattr(branch, "id") and branch.id == step_id:
                return branch

        return None
