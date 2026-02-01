"""Feature pipeline parser with sub-pipeline support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class EngineeredFeature:
    """Engineered feature from any source."""

    id: str  # Unique feature ID
    source_step_id: str  # Step that produces this feature
    output_key: str  # Context key where feature is stored
    parent_pipeline: Optional[str] = None  # Parent sub-pipeline ID if nested
    depends_on: List[str] = field(default_factory=list)


@dataclass
class FeaturePipeline:
    """Complete feature pipeline with nested support."""

    base_source: str  # Base preprocessor step
    base_output_key: str = "preprocessed_data"
    engineered: List[EngineeredFeature] = field(default_factory=list)

    def get_all_source_ids(self) -> Set[str]:
        """Get all source step IDs."""
        sources = {self.base_source}
        sources.update(f.source_step_id for f in self.engineered)
        return sources

    def get_feature_keys(self) -> List[str]:
        """Get all feature output keys."""
        keys = [self.base_output_key]
        keys.extend(f.output_key for f in self.engineered)
        return keys


class FeaturePipelineParser:
    """Parse feature pipeline with sub-pipeline flattening."""

    @staticmethod
    def parse_from_steps(train_steps: List[Any]) -> Optional[FeaturePipeline]:
        """Auto-detect feature pipeline from steps.

        Handles:
        - Top-level steps with output_as_feature
        - Steps inside sub-pipelines with output_as_feature
        - additional_feature_keys in datamodule steps

        Parameters
        ----------
        train_steps : List[Any]
            Training pipeline steps.

        Returns
        -------
        Optional[FeaturePipeline]
            Detected feature pipeline or None.
        """
        # Find base preprocessor
        base = FeaturePipelineParser._find_base_preprocessor(train_steps)
        if not base:
            return None

        # Collect all engineered features
        engineered = FeaturePipelineParser._collect_engineered_features(train_steps)

        if not engineered:
            return None

        return FeaturePipeline(
            base_source=base["source"],
            base_output_key=base["output_key"],
            engineered=engineered,
        )

    @staticmethod
    def _find_base_preprocessor(
        steps: List[Any],
    ) -> Optional[Dict[str, str]]:
        """Find base preprocessor step."""
        for step in steps:
            if step.get("type") == "preprocessor":
                output_key = "preprocessed_data"
                if "wiring" in step and "outputs" in step.wiring:
                    output_key = step.wiring.outputs.get(
                        "features", "preprocessed_data"
                    )

                return {"source": step["id"], "output_key": output_key}
        return None

    @staticmethod
    def _collect_engineered_features(
        steps: List[Any],
        parent_pipeline: Optional[str] = None,
    ) -> List[EngineeredFeature]:
        """Recursively collect engineered features.

        Parameters
        ----------
        steps : List[Any]
            Steps to search.
        parent_pipeline : Optional[str]
            Parent sub-pipeline ID if nested.

        Returns
        -------
        List[EngineeredFeature]
            All engineered features found.
        """
        features: List[EngineeredFeature] = []

        for step in steps:
            step_type = step.get("type")

            # Check for sub-pipeline
            if step_type == "sub_pipeline":
                nested_features = FeaturePipelineParser._extract_from_sub_pipeline(step)
                features.extend(nested_features)
                continue

            # Check for feature producer
            if FeaturePipelineParser._is_feature_producer(step):
                feat = FeaturePipelineParser._create_feature_from_step(
                    step, parent_pipeline
                )
                features.append(feat)

        return features

    @staticmethod
    def _extract_from_sub_pipeline(
        sub_pipeline_step: Any,
    ) -> List[EngineeredFeature]:
        """Extract features from sub-pipeline."""
        if not hasattr(sub_pipeline_step, "pipeline"):
            return []

        pipeline = sub_pipeline_step.pipeline
        if not hasattr(pipeline, "steps"):
            return []

        parent_id = sub_pipeline_step["id"]

        # Recursively collect from nested steps
        return FeaturePipelineParser._collect_engineered_features(
            pipeline.steps, parent_pipeline=parent_id
        )

    @staticmethod
    def _is_feature_producer(step: Any) -> bool:
        """Check if step produces features."""
        # Explicit flag
        if step.get("output_as_feature", False):
            return True

        # Clustering models auto-produce features
        if step.get("type") == "clustering":
            return True

        # Dynamic adapters with specific artifact types
        if step.get("type") == "dynamic_adapter":
            if step.get("artifact_type") == "preprocess":
                return True
            if step.get("log_artifact", False):
                return True

        return False

    @staticmethod
    def _create_feature_from_step(
        step: Any, parent_pipeline: Optional[str]
    ) -> EngineeredFeature:
        """Create EngineeredFeature from step."""
        step_id = step["id"]

        # Infer output key from wiring
        output_key = FeaturePipelineParser._infer_output_key(step)

        # Get dependencies
        depends_on = step.get("depends_on", [])

        return EngineeredFeature(
            id=f"feat_{step_id}",
            source_step_id=step_id,
            output_key=output_key,
            parent_pipeline=parent_pipeline,
            depends_on=depends_on,
        )

    @staticmethod
    def _infer_output_key(step: Any) -> str:
        """Infer output key from step configuration."""
        # Check wiring first
        if "wiring" in step and "outputs" in step.wiring:
            outputs = step.wiring.outputs

            # Try common keys
            for key in ["features", "output", "predictions"]:
                if key in outputs:
                    return outputs[key]

        # Default based on step type
        step_type = step.get("type")
        step_id = step["id"]

        if step_type == "clustering":
            return f"{step_id}_features"

        if step_type == "dynamic_adapter":
            return f"{step_id}_output"

        return f"{step_id}_features"

    @staticmethod
    def extract_additional_feature_keys(
        steps: List[Any],
    ) -> Set[str]:
        """Extract all additional_feature_keys references.

        This helps validate that all referenced features exist.

        Parameters
        ----------
        steps : List[Any]
            Pipeline steps.

        Returns
        -------
        Set[str]
            All feature keys referenced in additional_feature_keys.
        """
        keys: Set[str] = set()

        for step in steps:
            if "additional_feature_keys" in step:
                keys.update(step["additional_feature_keys"])

        return keys
