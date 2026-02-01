"""Serving pipeline builder."""

# pylint: disable=R0911

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from omegaconf import DictConfig, OmegaConf

from ...config import GeneratorConfig
from ...constants import CONTEXT_KEYS, STEP_CONSTANTS
from .base import BasePipelineBuilder
from .loader import LoaderBuilder


class ServeBuilder(BasePipelineBuilder):
    """Builds serving pipeline from training pipeline.

    Inherits common functionality from BasePipelineBuilder and adds
    serving-specific pipeline generation logic.
    """

    def __init__(
        self,
        train_steps: List[Any],
        experiment_type: str = "tabular",
        config: Optional[GeneratorConfig] = None,
    ) -> None:
        """Initialize serve pipeline builder.

        Args:
            train_steps: Original training pipeline steps.
            experiment_type: Type of experiment (timeseries, tabular).
            config: GeneratorConfig instance for customization.
        """
        super().__init__(train_steps, experiment_type, config)
        self.loader_builder = LoaderBuilder(self.config)

    def build(
        self,
        alias: str,
        init_id: str,
        preprocessor_step: Optional[Any],
        feature_pipeline: Optional[Any] = None,
    ) -> List[Any]:
        """Build complete serving pipeline.

        Args:
            alias: Model alias for loading.
            init_id: Initialization step ID.
            preprocessor_step: Legacy preprocessor step if exists.
            feature_pipeline: Feature pipeline config.

        Returns:
            List of serve pipeline steps.
        """
        steps: List[Any] = []

        preprocessors = self.extractor.extract_preprocessors(self.train_steps)
        producers = self.extractor.extract_model_producers(self.train_steps)

        self.loader_builder.add_mlflow_loader(
            steps,
            init_id,
            alias,
            preprocessors,
            producers,
            preprocessor_step,
            feature_pipeline,
        )

        last_id = self._add_preprocessors(steps, alias, init_id)

        last_id = self._add_legacy_preprocessor(
            steps, preprocessor_step, alias, init_id, preprocessors, last_id
        )

        pipeline_flags = {
            "has_sub": any(
                s.type == STEP_CONSTANTS.SUB_PIPELINE for s in self.train_steps
            ),
            "has_branch": any(
                s.type == STEP_CONSTANTS.BRANCH for s in self.train_steps
            ),
            "has_parallel": any(
                s.type == STEP_CONSTANTS.PARALLEL for s in self.train_steps
            ),
        }

        special_last_id, branch_ids = self._handle_special_steps(
            steps, alias, pipeline_flags["has_sub"], pipeline_flags["has_branch"]
        )

        if pipeline_flags["has_sub"] or pipeline_flags["has_branch"]:
            last_id = special_last_id

        inference_producers = self._get_inference_producers(
            producers, branch_ids, pipeline_flags
        )

        last_id = self._add_inference_steps(
            steps, inference_producers, init_id, last_id
        )

        self._add_final_profiling(steps, last_id)

        return steps

    def _add_preprocessors(self, steps: List[Any], alias: str, init_id: str) -> str:
        """Add preprocessors and adapters to pipeline."""
        last_id = init_id

        for step in self.train_steps:
            if step.type == STEP_CONSTANTS.PREPROCESSOR:
                prep = copy.deepcopy(step)
                prep.is_train = False
                prep.alias = alias
                prep.instance_key = f"{CONTEXT_KEYS.FITTED_PREFIX}{step.id}"
                prep.depends_on = [init_id]
                steps.append(prep)
                last_id = prep.id

            elif step.type == STEP_CONSTANTS.DYNAMIC_ADAPTER:
                if not self._should_add_adapter(step):
                    continue

                adapter = copy.deepcopy(step)
                adapter.is_train = False
                adapter.instance_key = f"{CONTEXT_KEYS.FITTED_PREFIX}{step.id}"
                adapter.depends_on = [init_id]

                if hasattr(step, "depends_on"):
                    adapter.depends_on.extend(step.depends_on)

                self.transformer.remove_training_configs(adapter)
                steps.append(adapter)
                last_id = adapter.id

        return last_id

    def _should_add_adapter(self, step: Any) -> bool:
        """Check if dynamic adapter should be added."""
        if not hasattr(step, "log_artifact") or not step.log_artifact:
            return False

        if not hasattr(step, "artifact_type"):
            return False

        return step.artifact_type == "preprocess"

    def _add_legacy_preprocessor(
        self,
        steps: List[Any],
        preprocessor_step: Optional[Any],
        alias: str,
        init_id: str,
        all_preprocessors: List[Any],
        current_last_id: str,
    ) -> str:
        """Add legacy preprocessor if needed."""
        if not preprocessor_step:
            return current_last_id

        if any(p.id == preprocessor_step.id for p in all_preprocessors):
            return current_last_id

        prep = copy.deepcopy(preprocessor_step)
        prep.is_train = False
        prep.alias = alias
        prep.instance_key = CONTEXT_KEYS.TRANSFORM_MANAGER
        prep.depends_on = [init_id]
        steps.append(prep)

        return prep.id

    def _handle_special_steps(
        self,
        steps: List[Any],
        alias: str,
        has_sub: bool,
        has_branch: bool,
    ) -> Tuple[str, Set[str]]:
        """Handle sub-pipelines and branches."""
        last_id = STEP_CONSTANTS.INIT_ARTIFACTS_ID
        branch_ids: Set[str] = set()

        if has_sub:
            for step in self.train_steps:
                if step.type == STEP_CONSTANTS.SUB_PIPELINE:
                    transformed = self._transform_sub_pipeline(step, alias)
                    steps.append(transformed)
                    last_id = step.id

        if has_branch:
            for step in self.train_steps:
                if step.type == STEP_CONSTANTS.BRANCH:
                    transformed = self._transform_branch(step)
                    steps.append(transformed)
                    last_id = step.id
                    branch_ids.update(self._extract_branch_ids(step))

        return last_id, branch_ids

    def _transform_sub_pipeline(self, step: Any, alias: str) -> Any:
        """Transform sub-pipeline for serve mode."""
        producer_ids = self.extractor.collect_producer_ids(
            self.extractor.extract_model_producers(self.train_steps)
        )

        resolved_deps = self.dependency_builder.resolve_dependencies(step, producer_ids)

        if hasattr(step, "depends_on"):
            step.depends_on = resolved_deps

        transformed = copy.deepcopy(step)

        if not hasattr(transformed, "pipeline"):
            return transformed

        if not hasattr(transformed.pipeline, "steps"):
            return transformed

        for sub_step in transformed.pipeline.steps:
            if sub_step.type == STEP_CONSTANTS.PREPROCESSOR:
                self.transformer.transform_preprocessor(sub_step, alias)

            elif sub_step.type in [STEP_CONSTANTS.CLUSTERING, STEP_CONSTANTS.MODEL]:
                sub_step.type = STEP_CONSTANTS.INFERENCE
                self.transformer.transform_model_step(sub_step)

                sub_step.wiring.outputs.predictions = (
                    sub_step.wiring.outputs.predictions
                    if hasattr(sub_step.wiring.outputs, "predictions")
                    else f"{sub_step['id']}{CONTEXT_KEYS.PREDICTIONS_SUFFIX}"
                )

        return transformed

    def _transform_branch(self, step: Any) -> Any:
        """Transform branch step for serve mode."""
        transformed = copy.deepcopy(step)

        if hasattr(transformed, "if_true"):
            inference = self._branch_to_inference(transformed.if_true)
            if inference:
                transformed.if_true = inference

        if hasattr(transformed, "if_false"):
            inference = self._branch_to_inference(transformed.if_false)
            if inference:
                transformed.if_false = inference

        return transformed

    def _branch_to_inference(self, branch: Any) -> Optional[DictConfig]:
        """Convert branch to inference step."""
        if not self.analyzer.is_model_producer(branch):
            return None

        inf_id = f"{branch.id}{CONTEXT_KEYS.INFERENCE_SUFFIX}"

        config = {
            "id": inf_id,
            "type": STEP_CONSTANTS.FEATURE_INFERENCE,
            "enabled": True,
            "depends_on": [STEP_CONSTANTS.PREPROCESS_ID],
            "source_model_key": f"{CONTEXT_KEYS.FITTED_PREFIX}{branch.id}",
            "base_features_key": self.resolve_base_features_key(branch),
            "output_key": f"{branch.id}{CONTEXT_KEYS.PREDICTIONS_SUFFIX}",
            "apply_windowing": self.should_apply_windowing(branch.id),
        }

        return OmegaConf.create(config)

    def _create_inference_wiring(self, model_id: str) -> Dict[str, Any]:
        """Create wiring for inference step."""
        return {
            "inputs": {
                "model": f"{CONTEXT_KEYS.FITTED_PREFIX}{model_id}",
                "features": CONTEXT_KEYS.PREPROCESSED_DATA,
            },
            "outputs": {
                "predictions": f"{model_id}{CONTEXT_KEYS.PREDICTIONS_SUFFIX}",
            },
        }

    def _extract_branch_ids(self, step: Any) -> Set[str]:
        """Extract producer IDs from branch step."""
        ids: Set[str] = set()

        for branch_name in ["if_true", "if_false"]:
            if not hasattr(step, branch_name):
                continue

            branch = getattr(step, branch_name)
            if hasattr(branch, "id"):
                ids.add(branch.id)

        return ids

    def _get_inference_producers(
        self,
        all_producers: List[Any],
        branch_ids: Set[str],
        pipeline_flags: Dict[str, bool],
    ) -> List[Any]:
        """Determine which producers need inference steps."""
        has_sub = pipeline_flags.get("has_sub", False)
        has_branch = pipeline_flags.get("has_branch", False)
        has_parallel = pipeline_flags.get("has_parallel", False)

        if not (has_sub or has_branch or has_parallel):
            return all_producers

        if has_parallel:
            return [p for p in all_producers if p.id not in branch_ids]

        return [
            p
            for p in all_producers
            if p.id not in branch_ids and any(s.id == p.id for s in self.train_steps)
        ]

    def _add_inference_steps(
        self,
        steps: List[Any],
        producers: List[Any],
        init_id: str,
        last_id: str,
    ) -> str:
        """Add inference steps to pipeline."""
        for producer in producers:
            inf_id = f"{producer.id}{CONTEXT_KEYS.INFERENCE_SUFFIX}"

            config: Dict[str, Any] = {
                "id": inf_id,
                "type": STEP_CONSTANTS.FEATURE_INFERENCE,
                "enabled": True,
                "depends_on": [init_id, last_id],
                "source_model_key": f"{CONTEXT_KEYS.FITTED_PREFIX}{producer.id}",
                "base_features_key": self.resolve_base_features_key(producer),
                "output_key": f"{producer.id}{CONTEXT_KEYS.PREDICTIONS_SUFFIX}",
                "apply_windowing": self.should_apply_windowing(producer.id),
            }

            # Add additional_feature_keys if producer uses composed features
            additional_keys = self.extract_additional_feature_keys(producer)
            if additional_keys:
                config["additional_feature_keys"] = additional_keys
                print(
                    f"[ServeBuilder] Propagating additional_feature_keys to {inf_id}: "
                    f"{additional_keys}"
                )

            inf_step = OmegaConf.create(config)

            steps.append(inf_step)
            last_id = inf_id

        return last_id

    def _add_final_profiling(self, steps: List[Any], last_id: str) -> None:
        """Add final profiling step."""
        steps.append(
            OmegaConf.create(
                {
                    "id": STEP_CONSTANTS.FINAL_PROFILING_ID,
                    "type": STEP_CONSTANTS.PROFILING,
                    "enabled": True,
                    "depends_on": [last_id],
                    "exclude_keys": CONTEXT_KEYS.PROFILING_EXCLUDE_KEYS,
                }
            )
        )
