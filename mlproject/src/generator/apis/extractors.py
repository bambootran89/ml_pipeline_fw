"""Mixins for extracting API logic from pipeline configurations."""

# flake8: noqa: C901
# pylint: disable=R1702, R0912

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import ConfigurablePatternMatcher, GeneratorConfig
from ..constants import (
    CONTEXT_KEYS,
    DATA_DEFAULTS,
    INFERENCE_METHODS,
    MODEL_PATTERNS,
    STEP_CONSTANTS,
)


class ApiGeneratorExtractorsMixin:
    """Extractions logic for API generation.

    Splits large configuration pipelines into minimal inference steps
    by identifying model-loading nodes and their dependencies.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize the extractor mixin.

        Args:
            config: Optional GeneratorConfig instance. If None, uses default config.
        """
        self.config = config or GeneratorConfig()
        self.matcher = ConfigurablePatternMatcher(self.config)

    def _extract_inference_steps(self, steps: List[Any]) -> List[Dict[str, Any]]:
        """Extract all model inference steps from pipeline.

        This identifies steps that perform model prediction and collects
        their configuration for code generation.

        Args:
            steps: List of pipeline steps.

        Returns:
            List of extracted inference information.
        """
        inference_steps: List[Dict[str, Any]] = []

        # Dispatcher map for step handlers
        handlers = {
            STEP_CONSTANTS.TRAINER: self._get_training_inference,
            STEP_CONSTANTS.CLUSTERING: self._get_training_inference,
            STEP_CONSTANTS.INFERENCE: self._get_inference_info,
            STEP_CONSTANTS.FEATURE_INFERENCE: self._get_inference_info,
            STEP_CONSTANTS.BRANCH: self._get_branch_inference,
            STEP_CONSTANTS.SUB_PIPELINE: self._get_nested_inference,
            STEP_CONSTANTS.PARALLEL: self._get_nested_inference,
        }

        for step in steps:
            step_type = getattr(step, "type", None) or step.get("type")
            handler = handlers.get(step_type)
            if handler:
                inference_steps.extend(handler(step))
        return inference_steps

    def _sort_by_dependencies(
        self, steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort inference steps by additional_feature_keys dependencies.

        Uses topological sort to ensure steps are executed in dependency order.
        """
        if not steps:
            return steps

        # Build maps for efficient lookup
        output_to_step_map = {
            step.get("output_key"): step for step in steps if step.get("output_key")
        }
        step_id_to_step_map = {}
        dependency_graph: Dict[str, List[str]] = {}

        # Build dependency graph
        for step in steps:
            step_id = step.get("id", "")
            step_id_to_step_map[step_id] = step
            dependency_graph[step_id] = []

            additional_keys = step.get("additional_feature_keys") or []
            for feature_key in additional_keys:
                if feature_key in output_to_step_map:
                    dependency_step_id = output_to_step_map[feature_key].get("id", "")
                    if dependency_step_id and dependency_step_id != step_id:
                        dependency_graph[step_id].append(dependency_step_id)

        # Perform topological sort using DFS
        sorted_steps: List[Dict[str, Any]] = []
        visited_steps: set = set()
        currently_visiting: set = set()

        def visit_step(step_id: str) -> None:
            """Visit step and its dependencies recursively."""
            if step_id in visited_steps:
                return
            if step_id in currently_visiting:
                # Circular dependency detected, skip
                return

            currently_visiting.add(step_id)

            # Visit all dependencies first
            for dependency_id in dependency_graph.get(step_id, []):
                if dependency_id in step_id_to_step_map:
                    visit_step(dependency_id)

            currently_visiting.remove(step_id)
            visited_steps.add(step_id)

            # Add step to result after dependencies
            if step_id in step_id_to_step_map:
                sorted_steps.append(step_id_to_step_map[step_id])

        # Visit all steps
        for step_id in dependency_graph:
            if step_id not in visited_steps:
                visit_step(step_id)

        # Add any remaining steps not in dependency graph
        for step in steps:
            if step not in sorted_steps:
                sorted_steps.append(step)

        return sorted_steps

    def _infer_model_type(self, model_key: str) -> str:
        """
        Infer model type from model key based on predefined patterns.

        Args:
            model_key: The string key of the model.

        Returns:
            String representing the model category (default is "ml").
        """
        return self.matcher.infer_model_type(model_key)

    def _get_training_inference(self, step: Any) -> List[Dict[str, Any]]:
        """Extract inference info from a training step."""
        step_id = getattr(step, "id", None) or step.get("id")
        inputs = getattr(step, "inputs", []) or step.get("inputs", [])

        model_key = step_id
        model_type = self._infer_model_type(model_key)

        return [
            {
                "id": f"{step_id}{CONTEXT_KEYS.INFERENCE_SUFFIX}",
                "model_key": model_key,
                "model_type": model_type,
                "inputs": list(inputs),
                "features_key": CONTEXT_KEYS.PREPROCESSED_DATA,
                "output_key": f"{step_id}{CONTEXT_KEYS.PREDICTIONS_SUFFIX}",
            }
        ]

    def _get_inference_info(self, step: Any) -> List[Dict[str, Any]]:
        """Extract info from an explicit inference step."""
        step_id = getattr(step, "id", None) or step.get("id")
        inputs = getattr(step, "inputs", []) or step.get("inputs", [])

        # Use get for args if it exists, otherwise check for
        # model_key/source_model_key in step directly
        model_key = None
        args = getattr(step, "args", None) or step.get("args")
        if args:
            model_key = args.get("model_key")

        if model_key is None:
            # Check wiring for model input if not explicitly set
            wiring = step.get("wiring")
            if wiring:
                inputs = wiring.get("inputs")
                if inputs:
                    model_key = inputs.get("model")

        if model_key is None:
            model_key = (
                step.get("source_model_key")
                or step.get("model_key")
                or step_id.replace(CONTEXT_KEYS.INFERENCE_SUFFIX, "")
            )

        model_type = self._infer_model_type(model_key)

        # Handle different field names for features/output
        features_key = step.get("features_key") or step.get("base_features_key")

        if not features_key:
            wiring = step.get("wiring")
            if wiring and wiring.get("inputs"):
                inputs = wiring.get("inputs")
                # heuristic to find input data key
                features_key = self.matcher.resolve_feature_key(inputs)

        if not features_key:
            features_key = CONTEXT_KEYS.PREPROCESSED_DATA

        output_key = (
            step.get("output_key") or f"{model_key}{CONTEXT_KEYS.PREDICTIONS_SUFFIX}"
        )

        # Extract additional_feature_keys if present
        additional_feature_keys = step.get("additional_feature_keys")
        if additional_feature_keys:
            additional_feature_keys = list(additional_feature_keys)

        return [
            {
                "id": step_id,
                "model_key": model_key,
                "model_type": model_type,
                "inputs": list(inputs),
                "features_key": features_key,
                "output_key": output_key,
                "additional_feature_keys": additional_feature_keys,
            }
        ]

    def _get_branch_inference(self, step: Any) -> List[Dict[str, Any]]:
        """Recursively extract inference info from branch steps."""
        results = []
        if_true = step.get("if_true")
        if_false = step.get("if_false")

        if if_true:
            results.extend(self._extract_inference_steps([if_true]))
        if if_false:
            results.extend(self._extract_inference_steps([if_false]))
        return results

    def _get_nested_inference(self, step: Any) -> List[Dict[str, Any]]:
        """Recursively extract inference info from sub_pipeline or parallel steps."""
        pipeline = step.get("pipeline")
        if pipeline and "steps" in pipeline:
            return self._extract_inference_steps(pipeline.steps)

        # Handle parallel directly if it's a list of steps
        steps = step.get("steps")
        if steps:
            return self._extract_inference_steps(steps)

        return []

    def _extract_load_map(self, steps: List[Any]) -> Dict[str, str]:
        """Extract mapping of model keys to MLflow artifact names."""
        load_map = {}
        for step in steps:
            step_type = getattr(step, "type", None) or step.get("type")
            if step_type in [
                STEP_CONSTANTS.INIT_ARTIFACTS_ID,
                STEP_CONSTANTS.MLFLOW_LOADER,
            ]:
                # Handle artifacts from init_artifacts or mlflow_loader step
                items = (
                    step.get("load_map")
                    or step.get("artifacts")
                    or step.get("args", {}).get("artifacts", {})
                )

                if items:
                    for item in items:
                        if hasattr(item, "get"):
                            key = item.get("context_key") or item.get("step_id")
                            val = item.get("step_id")
                            if key and val:
                                load_map[key] = val
                        elif isinstance(items, dict):
                            for k, v in items.items():
                                k_lower = k.lower()
                                if (
                                    STEP_CONSTANTS.MODEL.lower() in k_lower
                                    or STEP_CONSTANTS.PREPROCESS_ID in k_lower
                                    or "scaler" in k_lower
                                ):
                                    load_map[k] = v
                            break

            # Recursively explore nested steps for load_map (unlikely but possible)
            elif step_type == STEP_CONSTANTS.BRANCH:
                load_map.update(
                    self._extract_load_map([step.get("if_true"), step.get("if_false")])
                )
            elif step_type in [STEP_CONSTANTS.SUB_PIPELINE, STEP_CONSTANTS.PARALLEL]:
                nested_steps = step.get("steps") or (
                    step.get("pipeline", {}).get("steps")
                )
                if nested_steps:
                    load_map.update(self._extract_load_map(nested_steps))

        return load_map

    def _extract_preprocessor_info(self, steps: List[Any]) -> Optional[Dict[str, Any]]:
        """Extract preprocessor configuration if present."""
        for step in steps:
            if not step:
                continue
            step_type = getattr(step, "type", None) or step.get("type")

            if step_type == STEP_CONSTANTS.PREPROCESSOR:
                return {
                    "id": getattr(step, "id", None) or step.get("id"),
                    "instance_key": step.get("instance_key"),
                    "args": getattr(step, "args", None) or step,
                }

            if step_type in [
                STEP_CONSTANTS.INIT_ARTIFACTS_ID,
                STEP_CONSTANTS.MLFLOW_LOADER,
            ]:
                items = (
                    step.get("load_map")
                    or step.get("artifacts")
                    or step.get("args", {}).get("artifacts", {})
                )
                if items:
                    for item in items:
                        if hasattr(item, "get"):
                            key = item.get("context_key") or item.get("step_id")
                            val = item.get("step_id")
                            if key:
                                key_lower = key.lower()
                                if (
                                    STEP_CONSTANTS.PREPROCESS_ID in key_lower
                                    or "scaler" in key_lower
                                ):
                                    return {"id": key, "artifact_name": val}
                        elif isinstance(items, dict):
                            for k, v in items.items():
                                k_lower = k.lower()
                                if (
                                    STEP_CONSTANTS.PREPROCESS_ID in k_lower
                                    or "scaler" in k_lower
                                ):
                                    return {"id": k, "artifact_name": v}
                            break

            # Recursive search
            if step_type == STEP_CONSTANTS.BRANCH:
                info = self._extract_preprocessor_info(
                    [step.get("if_true"), step.get("if_false")]
                )
                if info:
                    return info
            elif step_type in [STEP_CONSTANTS.SUB_PIPELINE, STEP_CONSTANTS.PARALLEL]:
                nested_steps = step.get("steps") or (
                    step.get("pipeline", {}).get("steps")
                )
                if nested_steps:
                    info = self._extract_preprocessor_info(nested_steps)
                    if info:
                        return info

        return None

    def _get_preprocessor_artifact_name(
        self, preprocessor: Optional[Dict[str, Any]], load_map: Dict[str, str]
    ) -> Optional[str]:
        """Extract preprocessor artifact name from configuration or load map."""
        if not preprocessor:
            return None

        # Try artifact_name directly
        if "artifact_name" in preprocessor:
            return preprocessor["artifact_name"]

        # Try instance_key first (as it's often the key in load_map)
        prep_key = preprocessor.get("instance_key")
        if prep_key and prep_key in load_map:
            return load_map[prep_key]

        # Try to find in load_map using id
        prep_id = preprocessor.get("id")
        if prep_id and prep_id in load_map:
            return load_map[prep_id]

        # Fallback to id if it looks like a preprocessor
        if prep_id:
            prep_id_lower = prep_id.lower()
            if (
                STEP_CONSTANTS.PREPROCESS_ID in prep_id_lower
                or "scaler" in prep_id_lower
            ):
                return prep_id

        return None

    def _extract_data_config(self, cfg: Any) -> Dict[str, Any]:
        """Extract data configuration (type, features, windowing)."""
        data_cfg: Dict[str, Any] = {}

        # 1. Try to get from data section
        data = getattr(cfg, "data", None) or cfg.get("data")
        if data:
            data_cfg["data_type"] = data.get("type", DATA_DEFAULTS.TIMESERIES)
            data_cfg["features"] = list(data.get("features", []))
            data_cfg["path"] = data.get("path", "")
            data_cfg["entity_key"] = data.get("entity_key", "")

        # 2. Try to get from experiment section (overrides data)
        experiment = getattr(cfg, "experiment", None) or cfg.get("experiment")
        if experiment:
            exp_type = experiment.get("type")
            if exp_type in [DATA_DEFAULTS.CLASSIFICATION, DATA_DEFAULTS.REGRESSION]:
                data_cfg["data_type"] = DATA_DEFAULTS.TABULAR
            elif exp_type in [DATA_DEFAULTS.MULTIVARIATE, DATA_DEFAULTS.UNIVARIATE]:
                data_cfg["data_type"] = DATA_DEFAULTS.TIMESERIES
            elif exp_type:
                # Use experiment type if data type not yet set
                if "data_type" not in data_cfg:
                    data_cfg["data_type"] = exp_type

        # 3. Look for timeseries specific windowing and additional_feature_keys
        # Search through all steps for datamodule args
        pipeline = getattr(cfg, "pipeline", None) or cfg.get("pipeline")
        if pipeline and "steps" in pipeline:
            data_cfg.update(self._extract_datamodule_recursive(pipeline.steps))
            # Extract feature generators from sub-pipelines
            feature_generators = self._extract_feature_generators(pipeline.steps)
            if feature_generators:
                data_cfg["feature_generators"] = feature_generators

        # Defaults
        data_cfg.setdefault("data_type", DATA_DEFAULTS.DATA_TYPE)
        data_cfg.setdefault("features", DATA_DEFAULTS.FEATURES)
        data_cfg.setdefault("input_chunk_length", DATA_DEFAULTS.INPUT_CHUNK_LENGTH)
        data_cfg.setdefault("output_chunk_length", DATA_DEFAULTS.OUTPUT_CHUNK_LENGTH)
        data_cfg.setdefault(
            "additional_feature_keys", DATA_DEFAULTS.ADDITIONAL_FEATURE_KEYS
        )
        data_cfg.setdefault("feature_generators", DATA_DEFAULTS.FEATURE_GENERATORS)

        return data_cfg

    def _extract_feature_generators(
        self, steps: List[Any], parent_pipeline_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract feature generator steps based on additional_feature_keys.

        This method finds steps that OUTPUT the keys listed in additional_feature_keys
        from the datamodule step. This is more reliable than detecting by step type.

        Args:
            steps: List of pipeline steps.
            parent_pipeline_id: ID of parent sub_pipeline if any.

        Returns:
            List of feature generator configurations.
        """
        # Step 1: Find additional_feature_keys from datamodule or
        # feature_inference steps
        _ = parent_pipeline_id
        additional_keys = self._find_additional_feature_keys(steps)
        if not additional_keys:
            print("[ApiGenerator] No additional_feature_keys found in config")
            return []

        print(f"[ApiGenerator] Looking for steps that output: {additional_keys}")

        # Step 2: Build a map of output_key -> step for all steps (recursive)
        output_map = self._build_output_key_map(steps)

        # Step 3: Find steps that output the additional_feature_keys
        generators: List[Dict[str, Any]] = []
        for key in additional_keys:
            if key in output_map:
                step = output_map[key]
                fg_info = self._build_feature_generator_info_from_key(step, key)
                if fg_info:
                    generators.append(fg_info)
                    print(f"  Found: {fg_info['step_id']} -> {key}")
            else:
                print(f"  Warning: No step outputs '{key}'")

        return generators

    def _find_additional_feature_keys(self, steps: List[Any]) -> List[str]:
        """Find additional_feature_keys from datamodule or feature_inference steps."""
        keys: List[str] = []

        for step in steps:
            if not step:
                continue

            step_type = getattr(step, "type", None) or step.get("type")

            # Check datamodule step
            if step_type == STEP_CONSTANTS.DATAMODULE:
                additional = step.get("additional_feature_keys")
                if additional:
                    keys.extend(list(additional))

            # Check feature_inference step
            if step_type == STEP_CONSTANTS.FEATURE_INFERENCE:
                additional = step.get("additional_feature_keys")
                if additional:
                    keys.extend(list(additional))

            # Recursively check sub-pipelines
            if step_type == STEP_CONSTANTS.SUB_PIPELINE:
                pipeline = step.get("pipeline")
                if pipeline and "steps" in pipeline:
                    keys.extend(
                        self._find_additional_feature_keys(list(pipeline.steps))
                    )

        return list(set(keys))  # Remove duplicates

    def _build_output_key_map(self, steps: List[Any]) -> Dict[str, Any]:
        """Build a map of output_key -> step for all steps (recursive)."""
        output_map: Dict[str, Any] = {}

        for step in steps:
            if not step:
                continue

            step_type = getattr(step, "type", None) or step.get("type")

            # Extract output keys from wiring
            wiring = step.get("wiring")
            if wiring and wiring.get("outputs"):
                outputs = wiring.outputs
                if hasattr(outputs, "items"):
                    for _, context_key in outputs.items():
                        output_map[str(context_key)] = step

            # Recursively process sub-pipelines
            if step_type == STEP_CONSTANTS.SUB_PIPELINE:
                pipeline = step.get("pipeline")
                if pipeline and "steps" in pipeline:
                    nested_map = self._build_output_key_map(list(pipeline.steps))
                    output_map.update(nested_map)

            elif step_type == STEP_CONSTANTS.PARALLEL:
                nested_steps = step.get("steps") or []
                nested_map = self._build_output_key_map(nested_steps)
                output_map.update(nested_map)

        return output_map

    def _build_feature_generator_info_from_key(
        self, step: Any, output_key: str
    ) -> Optional[Dict[str, Any]]:
        """Build feature generator info from step and known output key."""
        step_id = getattr(step, "id", None) or step.get("id")
        step_type = getattr(step, "type", None) or step.get("type")

        if not step_id:
            return None

        # Determine inference method based on step type
        inference_method = INFERENCE_METHODS.TRANSFORM
        if step_type in [STEP_CONSTANTS.CLUSTERING, STEP_CONSTANTS.TRAINER]:
            inference_method = INFERENCE_METHODS.PREDICT
        elif step_type == STEP_CONSTANTS.INFERENCE:
            # Check step_id for hints
            if "cluster" in step_id.lower():
                inference_method = INFERENCE_METHODS.PREDICT
        elif step.get("run_method"):
            run_method = str(step.run_method)
            if INFERENCE_METHODS.PREDICT in run_method:
                inference_method = INFERENCE_METHODS.PREDICT

        # # Determine fg_type from step_id or step_type
        # fg_type = "transform"
        # step_id_lower = step_id.lower()
        # if step_type == "clustering" or "cluster" in step_id_lower:
        #     fg_type = "clustering"
        # elif "pca" in step_id_lower:
        #     fg_type = "pca"

        return {
            "step_id": step_id,
            "model_key": f"{CONTEXT_KEYS.FITTED_PREFIX}{step_id}",
            "artifact_name": step_id,
            "output_key": output_key,
            "inference_method": inference_method,
            "step_type": step_type,
        }

    def _is_feature_generator(self, step: Any) -> bool:
        """Check if step is a feature generator.

        A step is a feature generator if it:
        - Has output_as_feature=True
        - Is a clustering step with output key
        - Is a dynamic_adapter with artifact_type=preprocess and output_as_feature
        - Has wiring.outputs that produces feature-like keys

        Args:
            step: Pipeline step to check.

        Returns:
            True if step generates features for downstream use.
        """
        if not step:
            return False

        if step.get("output_as_feature", False):
            return True

        step_type = getattr(step, "type", None) or step.get("type")

        if step_type == STEP_CONSTANTS.CLUSTERING:
            return self._check_clustering_generator(step)

        if step_type == STEP_CONSTANTS.DYNAMIC_ADAPTER:
            return self._check_dynamic_adapter_generator(step)

        if step_type == STEP_CONSTANTS.INFERENCE:
            return self._check_inference_generator(step)

        return False

    def _check_clustering_generator(self, step: Any) -> bool:
        """Check if clustering step is a feature generator."""
        wiring = step.get("wiring")
        if wiring and wiring.get("outputs"):
            outputs = wiring.outputs
            # Has feature output (not just model)
            if self.matcher.resolve_output_key(outputs):
                return True
        return True  # Clustering typically generates features

    def _check_dynamic_adapter_generator(self, step: Any) -> bool:
        """Check if dynamic adapter step is a feature generator."""
        class_path = step.get("class_path", "").lower()

        # EXCLUDE preprocessors that modify columns in place
        if any(x in class_path for x in MODEL_PATTERNS.PREPROCESSOR_PATTERNS):
            return False

        # PCA, clustering add NEW columns - these are feature generators
        if any(x in class_path for x in MODEL_PATTERNS.FEATURE_GENERATOR_PATTERNS):
            return True

        # Check output_as_feature flag for other dynamic adapters
        if step.get("output_as_feature", False):
            return True
        return False

    def _check_inference_generator(self, step: Any) -> bool:
        """Check if inference step is a feature generator."""
        step_id = getattr(step, "id", None) or step.get("id", "")
        step_id_lower = step_id.lower()

        # EXCLUDE preprocessors (impute, scale) - they don't add new columns
        preprocessor_keywords = (
            MODEL_PATTERNS.IMPUTATION_KEYWORDS + MODEL_PATTERNS.SCALING_KEYWORDS
        )
        if any(x in step_id_lower for x in preprocessor_keywords):
            return False

        wiring = step.get("wiring")
        if wiring and wiring.get("outputs"):
            outputs = wiring.outputs
            # Check if it outputs features (not just predictions)
            if "features" in outputs:
                return True

        # Only cluster and pca add new columns
        feature_gen_keywords = (
            MODEL_PATTERNS.CLUSTERING_KEYWORDS
            + MODEL_PATTERNS.DIMENSIONALITY_REDUCTION_KEYWORDS
        )
        if any(x in step_id_lower for x in feature_gen_keywords):
            return True
        return False

    def _build_feature_generator_info(self, step: Any) -> Optional[Dict[str, Any]]:
        """Build feature generator info from step.

        Args:
            step: Feature generator step.

        Returns:
            Feature generator configuration dict or None.
        """
        step_id = getattr(step, "id", None) or step.get("id")
        step_type = getattr(step, "type", None) or step.get("type")

        if not step_id:
            return None

        # Determine output key
        output_key = self._determine_fg_output_key(step, step_id)

        # Determine inference method
        inference_method = self._determine_fg_inference_method(step, step_type, step_id)

        # Determine step type for categorization
        fg_type = self._determine_fg_type(step, step_type, step_id)

        return {
            "step_id": step_id,
            "model_key": f"{CONTEXT_KEYS.FITTED_PREFIX}{step_id}",
            "artifact_name": step_id,
            "output_key": output_key,
            "inference_method": inference_method,
            "step_type": fg_type,
        }

    def _determine_fg_output_key(self, step: Any, step_id: str) -> str:
        """Determine output key for feature generator."""
        output_key = None
        wiring = step.get("wiring")
        if wiring and wiring.get("outputs"):
            outputs = wiring.outputs
            # Try common output keys using matcher
            output_key = self.matcher.resolve_output_key(outputs)
            # Fallback to first output if matcher didn't find anything
            if not output_key and hasattr(outputs, "keys"):
                keys = list(outputs.keys())
                if keys:
                    output_key = outputs[keys[0]]

        if not output_key:
            output_key = f"{step_id}{CONTEXT_KEYS.FEATURES_SUFFIX}"
        return output_key

    def _determine_fg_inference_method(
        self, step: Any, step_type: str, step_id: str
    ) -> str:
        """Determine inference method for feature generator."""
        inference_method = INFERENCE_METHODS.TRANSFORM
        if step_type == STEP_CONSTANTS.CLUSTERING:
            inference_method = INFERENCE_METHODS.PREDICT
        elif step_type == STEP_CONSTANTS.INFERENCE:
            # For transformed inference steps, check step_id for type hints
            step_id_lower = step_id.lower()
            if any(x in step_id_lower for x in MODEL_PATTERNS.CLUSTERING_KEYWORDS):
                inference_method = INFERENCE_METHODS.PREDICT
            else:
                inference_method = INFERENCE_METHODS.TRANSFORM
        elif step.get("run_method"):
            run_method = step.run_method
            if INFERENCE_METHODS.TRANSFORM in run_method:
                inference_method = INFERENCE_METHODS.TRANSFORM
            elif INFERENCE_METHODS.PREDICT in run_method:
                inference_method = INFERENCE_METHODS.PREDICT
        return inference_method

    def _determine_fg_type(self, step: Any, step_type: str, step_id: str) -> str:
        """Determine feature generator type."""
        fg_type = INFERENCE_METHODS.TRANSFORM
        if step_type == STEP_CONSTANTS.CLUSTERING:
            fg_type = STEP_CONSTANTS.CLUSTERING
        elif step_type == STEP_CONSTANTS.INFERENCE:
            # For transformed inference steps, infer type from step_id
            step_id_lower = step_id.lower()
            if any(x in step_id_lower for x in MODEL_PATTERNS.CLUSTERING_KEYWORDS):
                fg_type = STEP_CONSTANTS.CLUSTERING
            elif any(
                x in step_id_lower
                for x in MODEL_PATTERNS.DIMENSIONALITY_REDUCTION_KEYWORDS
            ):
                fg_type = "pca"
            elif any(x in step_id_lower for x in MODEL_PATTERNS.IMPUTATION_KEYWORDS):
                fg_type = "imputer"
            elif any(x in step_id_lower for x in MODEL_PATTERNS.SCALING_KEYWORDS):
                fg_type = "scaler"
        elif step_type == STEP_CONSTANTS.DYNAMIC_ADAPTER:
            class_path = step.get("class_path", "").lower()
            if any(
                x in class_path
                for x in MODEL_PATTERNS.DIMENSIONALITY_REDUCTION_KEYWORDS
            ):
                fg_type = "pca"
            elif any(x in class_path for x in MODEL_PATTERNS.IMPUTATION_KEYWORDS):
                fg_type = "imputer"
            elif any(x in class_path for x in MODEL_PATTERNS.SCALING_KEYWORDS):
                fg_type = "scaler"
            elif any(x in class_path for x in MODEL_PATTERNS.CLUSTERING_KEYWORDS):
                fg_type = STEP_CONSTANTS.CLUSTERING
        return fg_type

    def _extract_datamodule_recursive(self, steps: List[Any]) -> Dict[str, Any]:
        """Recursively search for datamodule configuration in steps."""
        data_cfg: Dict[str, Any] = {}
        for step in steps:
            if not step:
                continue

            step_type = getattr(step, "type", None) or step.get("type")

            # Check for datamodule step with additional_feature_keys
            if step_type == STEP_CONSTANTS.DATAMODULE:
                additional_keys = step.get("additional_feature_keys")
                if additional_keys:
                    data_cfg["additional_feature_keys"] = list(additional_keys)

            dm_args = None
            args = getattr(step, "args", None) or step.get("args")
            if args and "datamodule" in args:
                dm_args = args.datamodule
            elif "datamodule" in step:
                dm_args = step.datamodule

            if dm_args:
                if "input_chunk_length" in dm_args:
                    data_cfg["input_chunk_length"] = dm_args.input_chunk_length
                if "output_chunk_length" in dm_args:
                    data_cfg["output_chunk_length"] = dm_args.output_chunk_length
                # Don't return early if we found dm_args but need to continue searching
                # for additional_feature_keys

            # Recursive search
            if step_type == STEP_CONSTANTS.BRANCH:
                res = self._extract_datamodule_recursive(
                    [step.get("if_true"), step.get("if_false")]
                )
                if res:
                    data_cfg.update(res)
            elif step_type in [STEP_CONSTANTS.SUB_PIPELINE, STEP_CONSTANTS.PARALLEL]:
                nested_steps = step.get("steps") or (
                    step.get("pipeline", {}).get("steps")
                )
                if nested_steps:
                    res = self._extract_datamodule_recursive(nested_steps)
                    if res:
                        data_cfg.update(res)

        return data_cfg
