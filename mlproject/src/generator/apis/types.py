"""Type definitions for API generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FeatureGenerator:
    """Configuration for a feature generator (sub-pipeline step)."""

    step_id: str
    model_key: str  # Key in context/MLflow, e.g., "fitted_cluster_type_1"
    artifact_name: str  # MLflow artifact name, e.g., "cluster_type_1"
    output_key: str  # Output key in context, e.g., "cluster_1_features"
    inference_method: str = "transform"  # or "predict"
    step_type: str = "transform"  # "transform", "clustering", "pca", etc.


@dataclass
class DataConfig:
    """Configuration for data type and features."""

    data_type: str
    features: List[str]
    input_chunk_length: int
    output_chunk_length: int
    path: str
    entity_key: str
    additional_feature_keys: List[str] = field(default_factory=list)
    feature_generators: List[FeatureGenerator] = field(default_factory=list)

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]]) -> DataConfig:
        """Create from dictionary with defaults."""
        if config is None:
            config = {}

        # Parse feature generators
        feature_generators = []
        for fg_dict in config.get("feature_generators", []):
            feature_generators.append(
                FeatureGenerator(
                    step_id=fg_dict.get("step_id", ""),
                    model_key=fg_dict.get("model_key", ""),
                    artifact_name=fg_dict.get("artifact_name", ""),
                    output_key=fg_dict.get("output_key", ""),
                    inference_method=fg_dict.get("inference_method", "transform"),
                    step_type=fg_dict.get("step_type", "transform"),
                )
            )

        return cls(
            data_type=config.get("data_type", "timeseries"),
            features=config.get("features", []),
            input_chunk_length=config.get("input_chunk_length", 24),
            output_chunk_length=config.get("output_chunk_length", 6),
            path=config.get("path", ""),
            entity_key=config.get("entity_key", ""),
            additional_feature_keys=config.get("additional_feature_keys", []),
            feature_generators=feature_generators,
        )

    @property
    def is_feast(self) -> bool:
        """Check if data source is Feast."""
        return self.path.startswith("feast://")

    @property
    def feast_repo_path(self) -> str:
        """Extract Feast repository path from URI."""
        if not self.is_feast:
            return ""
        # Format: feast://path/to/repo?query
        return self.path.split("feast://")[1].split("?")[0]

    @property
    def has_feature_generators(self) -> bool:
        """Check if pipeline has feature generators (sub-pipeline)."""
        return len(self.feature_generators) > 0 or len(self.additional_feature_keys) > 0


@dataclass(frozen=True)
class GenerationContext:
    """Context for code generation."""

    pipeline_name: str
    load_map: Dict[str, str]
    preprocessor: Optional[Dict[str, Any]]
    inference_steps: List[Dict[str, Any]]
    experiment_config_path: str
    data_config: DataConfig
    model_keys: List[str]
    alias: str = "production"
