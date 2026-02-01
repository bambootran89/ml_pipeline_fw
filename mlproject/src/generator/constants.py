"""Constants for ML pipeline generator.

This module centralizes all hardcoded string values, keys, and patterns
used throughout the generator package to improve maintainability and
allow for easier configuration.
"""
# pylint: disable=invalid-name

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StepConstants:
    """Constants related to pipeline steps and identifiers."""

    # Step IDs
    INIT_ARTIFACTS_ID: str = "init_artifacts"
    FINAL_PROFILING_ID: str = "final_profiling"
    LOAD_DATA_ID: str = "load_data"
    PREPROCESS_ID: str = "preprocess"

    # Step types
    DATA_LOADER: str = "data_loader"
    PREPROCESSOR: str = "preprocessor"
    DYNAMIC_ADAPTER: str = "dynamic_adapter"
    MLFLOW_LOADER: str = "mlflow_loader"
    INFERENCE: str = "inference"
    FEATURE_INFERENCE: str = "feature_inference"
    EVALUATOR: str = "evaluator"
    PROFILING: str = "profiling"
    LOGGER: str = "logger"
    TRAINER: str = "trainer"
    CLUSTERING: str = "clustering"
    MODEL: str = "model"
    FRAMEWORK_MODEL: str = "framework_model"
    DATAMODULE: str = "datamodule"
    SUB_PIPELINE: str = "sub_pipeline"
    BRANCH: str = "branch"
    PARALLEL: str = "parallel"

    # Step type groups for windowing
    WINDOWING_STEP_TYPES: List[str] = field(
        default_factory=lambda: ["trainer", "clustering", "framework_model"]
    )

    # Step types that should not apply windowing
    NON_WINDOWING_KEYWORDS: List[str] = field(
        default_factory=lambda: ["impute", "pca", "scaler"]
    )


@dataclass
class ContextKeyConstants:
    """Constants for context keys used in pipeline execution."""

    # Primary context keys
    PREPROCESSED_DATA: str = "preprocessed_data"
    TARGET_DATA: str = "target_data"
    RAW_DATA: str = "raw_data"
    TRANSFORM_MANAGER: str = "transform_manager"
    EVALUATION_METRICS: str = "evaluation_metrics"

    # Key naming patterns
    FITTED_PREFIX: str = "fitted_"
    PREDICTIONS_SUFFIX: str = "_predictions"
    METRICS_SUFFIX: str = "_metrics"
    FEATURES_SUFFIX: str = "_features"
    INFERENCE_SUFFIX: str = "_inference"
    EVALUATE_SUFFIX: str = "_evaluate"

    # Priority lists for key resolution
    FEATURE_KEY_PRIORITY: List[str] = field(
        default_factory=lambda: ["X", "features", "data", "input"]
    )

    OUTPUT_KEY_PRIORITY: List[str] = field(
        default_factory=lambda: ["features", "cluster_labels", "predictions", "data"]
    )

    # Keys to exclude from profiling
    PROFILING_EXCLUDE_KEYS: List[str] = field(
        default_factory=lambda: [
            "cfg",
            "preprocessor",
            "df",
            "train_df",
            "val_df",
            "test_df",
        ]
    )


@dataclass
class ModelTypePatterns:
    """Patterns for detecting model types from model keys or class paths."""

    # Model type detection patterns
    DEEP_LEARNING_PATTERNS: List[str] = field(
        default_factory=lambda: [
            "darts",
            "pytorch",
            "tensorflow",
            "keras",
            "lstm",
            "tft",
            "nhits",
            "nbeats",
            "tcn",
            "rnn",
            "gru",
        ]
    )

    MACHINE_LEARNING_PATTERNS: List[str] = field(
        default_factory=lambda: [
            "sklearn",
            "xgboost",
            "catboost",
            "lightgbm",
            "regression",
            "forest",
            "tree",
            "svm",
            "knn",
        ]
    )

    # Feature generator detection patterns
    PREPROCESSOR_PATTERNS: List[str] = field(
        default_factory=lambda: [
            "imputer",
            "scaler",
            "normalizer",
            "encoder",
            "transformer",
        ]
    )

    FEATURE_GENERATOR_PATTERNS: List[str] = field(
        default_factory=lambda: ["pca", "cluster", "kmeans", "decomposition"]
    )

    # Step ID detection keywords
    CLUSTERING_KEYWORDS: List[str] = field(
        default_factory=lambda: ["cluster", "kmeans", "dbscan", "hierarchical"]
    )

    DIMENSIONALITY_REDUCTION_KEYWORDS: List[str] = field(
        default_factory=lambda: ["pca", "svd", "tsne", "umap", "lda"]
    )

    IMPUTATION_KEYWORDS: List[str] = field(
        default_factory=lambda: ["impute", "fillna", "missing"]
    )

    SCALING_KEYWORDS: List[str] = field(
        default_factory=lambda: ["scale", "scaler", "normalize", "standardize"]
    )


@dataclass
class DataConfigDefaults:
    """Default values for data configuration."""

    DATA_TYPE: str = "timeseries"
    INPUT_CHUNK_LENGTH: int = 24
    OUTPUT_CHUNK_LENGTH: int = 6
    FEATURES: List[str] = field(default_factory=list)
    ADDITIONAL_FEATURE_KEYS: List[str] = field(default_factory=list)
    FEATURE_GENERATORS: List[Dict] = field(default_factory=list)

    # Data types
    TIMESERIES: str = "timeseries"
    TABULAR: str = "tabular"

    # Experiment types
    CLASSIFICATION: str = "classification"
    REGRESSION: str = "regression"
    MULTIVARIATE: str = "multivariate"
    UNIVARIATE: str = "univariate"


@dataclass
class APIDefaults:
    """Default configuration for generated API code."""

    # FastAPI defaults
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000

    # Ray Serve defaults
    RAY_PREPROCESS_REPLICAS: int = 2
    RAY_PREPROCESS_CPUS: float = 0.5
    RAY_MODEL_REPLICAS: int = 1
    RAY_MODEL_CPUS: float = 1.0

    # API metadata
    API_VERSION: str = "1.0.0"
    API_TITLE_SUFFIX: str = " API"


@dataclass
class InferenceMethodDefaults:
    """Default inference methods for different step types."""

    TRANSFORM: str = "transform"
    PREDICT: str = "predict"
    PREDICT_PROBA: str = "predict_proba"
    FIT_TRANSFORM: str = "fit_transform"

    # Method mapping for step types
    CLUSTERING_METHOD: str = "predict"
    TRAINER_METHOD: str = "predict"
    PREPROCESSOR_METHOD: str = "transform"
    DYNAMIC_ADAPTER_METHOD: str = "transform"


# Global singleton instances
STEP_CONSTANTS = StepConstants()
CONTEXT_KEYS = ContextKeyConstants()
MODEL_PATTERNS = ModelTypePatterns()
DATA_DEFAULTS = DataConfigDefaults()
API_DEFAULTS = APIDefaults()
INFERENCE_METHODS = InferenceMethodDefaults()


__all__ = [
    "StepConstants",
    "ContextKeyConstants",
    "ModelTypePatterns",
    "DataConfigDefaults",
    "APIDefaults",
    "InferenceMethodDefaults",
    "STEP_CONSTANTS",
    "CONTEXT_KEYS",
    "MODEL_PATTERNS",
    "DATA_DEFAULTS",
    "API_DEFAULTS",
    "INFERENCE_METHODS",
]
