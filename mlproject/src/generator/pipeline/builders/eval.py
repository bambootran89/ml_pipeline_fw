"""Eval builder with feature pipeline support."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from omegaconf import DictConfig, OmegaConf

from ...config import GeneratorConfig
from ...constants import CONTEXT_KEYS, STEP_CONSTANTS
from ..feature_parser import EngineeredFeature, FeaturePipeline
from .base import BasePipelineBuilder
from .loader import LoaderBuilder


class EvalBuilder(BasePipelineBuilder):
    """Evaluation pipeline builder.

    Inherits common functionality from BasePipelineBuilder and adds
    evaluation-specific pipeline generation logic.
    """

    # pylint: disable=R0911
    def __init__(
        self,
        train_steps: List[Any],
        experiment_type: str = "tabular",
        config: Optional[GeneratorConfig] = None,
    ) -> None:
        """Initialize eval builder.

        Parameters
        ----------
        train_steps : List[Any]
            Original training steps.
        experiment_type : str
            Type of experiment (timeseries, tabular).
        config : Optional[GeneratorConfig]
            Generator configuration for customization.
        """
        super().__init__(train_steps, experiment_type, config)
        self.loader_builder = LoaderBuilder(self.config)

    def build(
        self,
        alias: str,
        init_id: str,
        preprocessor_step: Optional[Any],
        feature_pipeline: Optional[FeaturePipeline] = None,
    ) -> List[Any]:
        """Build evaluation pipeline.

        Parameters
        ----------
        alias : str
            Model alias for loading.
        init_id : str
            Init step ID.
        preprocessor_step : Optional[Any]
            Legacy preprocessor.
        feature_pipeline : Optional[FeaturePipeline]
            Feature pipeline config.

        Returns
        -------
        List[Any]
            Eval pipeline steps.
        """
        steps: List[Any] = []

        # Data loader
        data_loader = next(
            (s for s in self.train_steps if s.type == STEP_CONSTANTS.DATA_LOADER), None
        )
        if data_loader:
            steps.append(data_loader)

        # Extract components
        preprocessors = self.extractor.extract_preprocessors(self.train_steps)
        producers = self.extractor.extract_model_producers(self.train_steps)

        # MLflow loader
        self.loader_builder.add_mlflow_loader(
            steps,
            init_id,
            alias,
            preprocessors,
            producers,
            preprocessor_step,
            feature_pipeline,
        )

        # Legacy preprocessor
        self._add_legacy_preprocessor(
            steps, preprocessor_step, init_id, preprocessors, data_loader
        )

        # Regular preprocessors
        self._add_preprocessors(steps, alias, init_id, data_loader)

        # Feature inference steps (NEW)
        if feature_pipeline:
            self._add_feature_inference_steps(steps, feature_pipeline, init_id)

        # Sub-pipelines and branches
        special, special_ids, branch_ids = self._process_special_steps(
            alias, feature_pipeline
        )
        steps.extend(special)

        # Evaluators
        evaluator_ids = self._add_evaluators(
            steps,
            producers,
            branch_ids,
            init_id,
            next(
                (
                    s.id
                    for s in self.train_steps
                    if s.type == STEP_CONSTANTS.PREPROCESSOR
                ),
                None,
            ),
            [
                sid
                for sid, step in zip(special_ids, special)
                if hasattr(step, "pipeline")
            ],
        )

        evaluator_ids.extend(
            [
                sid
                for sid, step in zip(special_ids, special)
                if hasattr(step, "condition")
            ]
        )

        # Auxiliary steps
        self._add_auxiliary_steps(steps, evaluator_ids)

        return steps

    def _add_feature_inference_steps(
        self,
        steps: List[Any],
        feature_pipeline: FeaturePipeline,
        init_id: str,
    ) -> None:
        """Add feature inference steps for engineered features.

        Parameters
        ----------
        steps : List[Any]
            Pipeline steps to append to.
        feature_pipeline : FeaturePipeline
            Feature pipeline config.
        init_id : str
            Init artifacts step ID.
        """
        for feat in feature_pipeline.engineered:
            # Skip if inside sub-pipeline (handled separately)
            if feat.parent_pipeline:
                continue

            step = self._create_feature_inference_step(feat, init_id)
            steps.append(step)

            print(
                f"[EvalBuilder] Added feature inference: "
                f"{step['id']} -> {feat.output_key}"
            )

    def _create_feature_inference_step(
        self, feat: EngineeredFeature, init_id: str
    ) -> DictConfig:
        """Create feature inference step config.

        Parameters
        ----------
        feat : EngineeredFeature
            Feature to generate.
        init_id : str
            Init artifacts step ID.

        Returns
        -------
        DictConfig
            Feature inference step.
        """
        step_id = f"{feat.source_step_id}{CONTEXT_KEYS.INFERENCE_SUFFIX}"
        model_key = f"{CONTEXT_KEYS.FITTED_PREFIX}{feat.source_step_id}"

        return OmegaConf.create(
            {
                "id": step_id,
                "type": STEP_CONSTANTS.FEATURE_INFERENCE,
                "enabled": True,
                "depends_on": [init_id],
                "source_model_key": model_key,
                "base_features_key": CONTEXT_KEYS.PREPROCESSED_DATA,
                "output_key": feat.output_key,
                "apply_windowing": self.should_apply_windowing(feat.source_step_id),
            }
        )

    def _transform_sub_pipeline(
        self, step: Any, alias: str, feature_pipeline: Optional[FeaturePipeline]
    ) -> Any:
        """Transform sub-pipeline for eval mode.

        Parameters
        ----------
        step : Any
            Sub-pipeline step.
        alias : str
            Model alias.
        feature_pipeline : Optional[FeaturePipeline]
            Feature pipeline config.

        Returns
        -------
        Any
            Transformed step.
        """
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

        # Transform nested steps
        for sub_step in transformed.pipeline.steps:
            if sub_step.type == STEP_CONSTANTS.PREPROCESSOR:
                self.transformer.transform_preprocessor(sub_step, alias)

            elif sub_step.type == STEP_CONSTANTS.DYNAMIC_ADAPTER:
                self._transform_dynamic_adapter_in_sub(
                    sub_step, alias, feature_pipeline
                )

            elif sub_step.type in [STEP_CONSTANTS.CLUSTERING, STEP_CONSTANTS.MODEL]:
                self._transform_model_in_sub(sub_step, alias, feature_pipeline)

        return transformed

    def _transform_dynamic_adapter_in_sub(
        self,
        step: Any,
        alias: str,
        feature_pipeline: Optional[FeaturePipeline],
    ) -> None:
        """Transform dynamic adapter in sub-pipeline."""
        if not step.get("log_artifact", False):
            return
        _ = alias
        # Check if it's a feature producer
        is_feature = False
        if feature_pipeline:
            is_feature = any(
                f.source_step_id == step["id"] for f in feature_pipeline.engineered
            )

        if is_feature:
            # Convert to feature inference mode
            step.is_train = False
            step.instance_key = f"{CONTEXT_KEYS.FITTED_PREFIX}{step['id']}"
            self.transformer.remove_training_configs(step)

            # Change run_method to transform
            if step.get("run_method") == "fit_transform":
                step.run_method = "transform"

    def _transform_model_in_sub(
        self,
        step: Any,
        alias: str,
        feature_pipeline: Optional[FeaturePipeline],
    ) -> None:
        """Transform model/clustering in sub-pipeline."""
        # Check if it's a feature producer
        is_feature = False
        _ = alias
        if feature_pipeline:
            is_feature = any(
                f.source_step_id == step["id"] for f in feature_pipeline.engineered
            )

        if is_feature:
            # Convert to inference mode
            step.type = STEP_CONSTANTS.FEATURE_INFERENCE
            step.source_model_key = f"{CONTEXT_KEYS.FITTED_PREFIX}{step['id']}"
            step.base_features_key = CONTEXT_KEYS.PREPROCESSED_DATA

            # Get output key from feature pipeline
            if feature_pipeline:
                for feat in feature_pipeline.engineered:
                    if feat.source_step_id == step["id"]:
                        step.output_key = feat.output_key
                        break

    def _process_special_steps(
        self, alias: str, feature_pipeline: Optional[FeaturePipeline]
    ) -> Tuple[List[Any], List[str], Set[str]]:
        """Process sub-pipelines and branches."""
        special = []
        special_ids = []
        branch_ids: Set[str] = set()

        for step in self.train_steps:
            if step.type == STEP_CONSTANTS.SUB_PIPELINE:
                transformed = self._transform_sub_pipeline(
                    step, alias, feature_pipeline
                )
                special.append(transformed)
                special_ids.append(step.id)

            elif step.type == STEP_CONSTANTS.BRANCH:
                transformed = self._transform_branch(step)
                special.append(transformed)
                special_ids.append(step.id)

                for branch_name in ["if_true", "if_false"]:
                    if not hasattr(step, branch_name):
                        continue

                    branch = getattr(step, branch_name)
                    if hasattr(branch, "id"):
                        branch_ids.add(branch.id)

            elif step.type == STEP_CONSTANTS.PARALLEL:
                special_ids.append(step.id)

        return special, special_ids, branch_ids

    # Keep existing methods from original EvalBuilder
    def _add_legacy_preprocessor(
        self,
        steps: List[Any],
        preprocessor_step: Optional[Any],
        init_id: str,
        all_preprocessors: List[Any],
        data_loader: Optional[Any],
    ) -> None:
        """Add legacy preprocessor if needed."""
        if not preprocessor_step:
            return

        if any(p.id == preprocessor_step.id for p in all_preprocessors):
            return

        prep = copy.deepcopy(preprocessor_step)
        prep.is_train = False
        prep.alias = init_id
        prep.instance_key = CONTEXT_KEYS.TRANSFORM_MANAGER
        prep.depends_on = [init_id]

        if data_loader:
            prep.depends_on.append(STEP_CONSTANTS.LOAD_DATA_ID)

        steps.append(prep)

    def _add_preprocessors(
        self,
        steps: List[Any],
        alias: str,
        init_id: str,
        data_loader: Optional[Any],
    ) -> None:
        """Add preprocessors to pipeline."""
        for step in self.train_steps:
            if step.type == STEP_CONSTANTS.PREPROCESSOR:
                prep = copy.deepcopy(step)
                prep.is_train = False
                prep.alias = alias
                prep.instance_key = f"{CONTEXT_KEYS.FITTED_PREFIX}{step.id}"
                prep.depends_on = [init_id]

                if data_loader:
                    prep.depends_on.append(STEP_CONSTANTS.LOAD_DATA_ID)

                steps.append(prep)

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

    def _should_add_adapter(self, step: Any) -> bool:
        """Check if adapter should be added."""
        if not hasattr(step, "log_artifact") or not step.log_artifact:
            return False

        if not hasattr(step, "artifact_type"):
            return False

        return step.artifact_type == STEP_CONSTANTS.PREPROCESS_ID

    def _transform_branch(self, step: Any) -> Any:
        """Transform branch for eval."""
        producer_ids = self.extractor.collect_producer_ids(
            self.extractor.extract_model_producers(self.train_steps)
        )

        resolved_deps = self.dependency_builder.resolve_dependencies(step, producer_ids)

        if hasattr(step, "depends_on"):
            step.depends_on = resolved_deps

        transformed = copy.deepcopy(step)

        if hasattr(transformed, "if_true"):
            evaluator = self._branch_to_evaluator(transformed.if_true)
            if evaluator:
                transformed.if_true = evaluator

        if hasattr(transformed, "if_false"):
            evaluator = self._branch_to_evaluator(transformed.if_false)
            if evaluator:
                transformed.if_false = evaluator

        return transformed

    def _branch_to_evaluator(self, branch: Any) -> Optional[DictConfig]:
        """Convert branch to evaluator."""
        if not self.analyzer.is_model_producer(branch):
            return None

        eval_id = (
            f"{branch.id}{CONTEXT_KEYS.EVALUATE_SUFFIX}"
            if not branch.id.endswith(CONTEXT_KEYS.EVALUATE_SUFFIX)
            else branch.id
        )

        outputs = (
            branch.wiring.outputs
            if hasattr(branch, "wiring") and hasattr(branch.wiring, "outputs")
            else None
        )

        wiring = self._create_evaluator_wiring(branch.id, branch.type, outputs)

        config = {
            "id": eval_id,
            "type": STEP_CONSTANTS.EVALUATOR,
            "enabled": True,
            "depends_on": [STEP_CONSTANTS.PREPROCESS_ID],
            "wiring": wiring,
        }

        if branch.type == STEP_CONSTANTS.CLUSTERING:
            config["step_eval_type"] = STEP_CONSTANTS.CLUSTERING

        return OmegaConf.create(config)

    def _create_evaluator_wiring(
        self, model_id: str, model_type: str, outputs: Optional[Any]
    ) -> Dict[str, Any]:
        """Create evaluator wiring."""
        wiring = {
            "inputs": {
                "model": f"{CONTEXT_KEYS.FITTED_PREFIX}{model_id}",
                "features": CONTEXT_KEYS.PREPROCESSED_DATA,
            },
            "outputs": {
                "metrics": (
                    outputs.get("metrics", CONTEXT_KEYS.EVALUATION_METRICS)
                    if outputs
                    else CONTEXT_KEYS.EVALUATION_METRICS
                )
            },
        }

        if model_type != STEP_CONSTANTS.CLUSTERING:
            wiring["inputs"]["targets"] = CONTEXT_KEYS.TARGET_DATA

        return wiring

    def _add_evaluators(
        self,
        steps: List[Any],
        producers: List[Any],
        branch_ids: Set[str],
        init_id: str,
        preprocess_id: Optional[str],
        sub_pipeline_ids: List[str],
    ) -> List[str]:
        """Add evaluators."""
        evaluator_ids = []

        for producer in producers:
            if producer.id in branch_ids:
                continue

            if self.is_clustering_adapter(producer):
                continue

            evaluator = self._make_evaluator(
                producer, init_id, preprocess_id, producers
            )

            for sid in sub_pipeline_ids:
                if sid not in evaluator.depends_on:
                    evaluator.depends_on.append(sid)

            evaluator_ids.append(evaluator.id)
            steps.append(evaluator)

        return evaluator_ids

    def _make_evaluator(
        self,
        producer: Any,
        init_id: str,
        preprocessor_id: Optional[str],
        all_producers: List[Any],
    ) -> DictConfig:
        """Create evaluator config."""
        base_name = self.transformer.extract_base_name(producer.id)
        eval_id = f"{base_name}{CONTEXT_KEYS.EVALUATE_SUFFIX}"

        is_clustering = self.analyzer.is_clustering(producer)
        features_input = self.analyzer.extract_features_input(producer)

        inputs = self._build_evaluator_inputs(
            producer.id, features_input, is_clustering
        )

        producer_ids = self.extractor.collect_producer_ids(all_producers)

        deps = self.dependency_builder.build_eval_dependencies(
            producer, init_id, preprocessor_id, producer_ids
        )

        config: Dict[str, Any] = {
            "id": eval_id,
            "type": STEP_CONSTANTS.EVALUATOR,
            "enabled": True,
            "depends_on": deps,
            "wiring": {
                "inputs": inputs,
                "outputs": {"metrics": f"{base_name}{CONTEXT_KEYS.METRICS_SUFFIX}"},
            },
        }

        if is_clustering:
            config["step_eval_type"] = STEP_CONSTANTS.CLUSTERING

        # Add additional_feature_keys if producer uses composed features
        additional_keys = self.extract_additional_feature_keys(producer)
        if additional_keys:
            config["additional_feature_keys"] = additional_keys
            print(
                f"[EvalBuilder] Propagating additional_feature_keys to {eval_id}: "
                f"{additional_keys}"
            )

        return OmegaConf.create(config)

    def _build_evaluator_inputs(
        self, mp_id: str, features: str, is_clustering: bool
    ) -> Dict[str, Any]:
        """Build evaluator inputs."""
        inputs: Dict[str, Any] = {
            "features": features,
            "model": f"{CONTEXT_KEYS.FITTED_PREFIX}{mp_id}",
        }

        if not is_clustering:
            inputs["targets"] = CONTEXT_KEYS.TARGET_DATA

        return inputs

    def _add_auxiliary_steps(self, steps: List[Any], evaluator_ids: List[str]) -> None:
        """Add logger and profiling."""
        for step in self.train_steps:
            if step.type not in [STEP_CONSTANTS.LOGGER, STEP_CONSTANTS.PROFILING]:
                continue

            aux = copy.deepcopy(step)
            aux.depends_on = evaluator_ids

            if aux.type == STEP_CONSTANTS.PROFILING:
                aux.exclude_keys = CONTEXT_KEYS.PROFILING_EXCLUDE_KEYS

            steps.append(aux)
