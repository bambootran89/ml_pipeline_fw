"""Pipeline steps package.

This package provides all pipeline step implementations.
Importing this package triggers registration of all step types.

Import Order:
1. base.py - BasePipelineStep (no dependencies)
2. factory.py - StepFactory (empty registry)
3. All step files - each imports factory and registers itself
"""

# Advanced steps
from mlproject.src.pipeline.steps.control.advanced import (
    BranchStep,
    ParallelStep,
    SubPipelineStep,
)
from mlproject.src.pipeline.steps.control.dynamic_adapter import DynamicAdapterStep

# Base class (no circular import issues)
from mlproject.src.pipeline.steps.core.base import BasePipelineStep

# Constants and utilities (no circular dependencies)
from mlproject.src.pipeline.steps.core.constants import (
    ColumnNames,
    ConfigPaths,
    ContextKeys,
    DataTypes,
    DefaultValues,
    ModelTypes,
)

# Factory (defines empty registry)
from mlproject.src.pipeline.steps.core.factory import StepFactory
from mlproject.src.pipeline.steps.core.utils import (
    ConfigAccessor,
    ConfigMerger,
    SampleAligner,
    WindowBuilder,
)

# Import all steps to trigger registration
# Each step file imports StepFactory and calls register() at module load
from mlproject.src.pipeline.steps.data.data_loader import DataLoaderStep
from mlproject.src.pipeline.steps.data.datamodule import DataModuleStep
from mlproject.src.pipeline.steps.data.preprocessor import PreprocessorStep
from mlproject.src.pipeline.steps.features.inference import FeatureInferenceStep
from mlproject.src.pipeline.steps.handlers.data_handlers import (
    DataTypeHandler,
    DataTypeHandlerFactory,
    TabularHandler,
    TimeseriesHandler,
)
from mlproject.src.pipeline.steps.inference.evaluator import EvaluatorStep
from mlproject.src.pipeline.steps.inference.inference import InferenceStep
from mlproject.src.pipeline.steps.mlops.logger import LoggerStep
from mlproject.src.pipeline.steps.mlops.mlflow_loader import MLflowLoaderStep
from mlproject.src.pipeline.steps.mlops.profiling import ProfilingStep
from mlproject.src.pipeline.steps.models.framework_model import (
    ClusteringModelStep,
    FrameworkModelStep,
)
from mlproject.src.pipeline.steps.models.trainer import TrainerStep
from mlproject.src.pipeline.steps.models.tuner import TunerStep

__all__ = [
    # Base
    "BasePipelineStep",
    # Factory
    "StepFactory",
    # Constants
    "ContextKeys",
    "ColumnNames",
    "DataTypes",
    "ModelTypes",
    "DefaultValues",
    "ConfigPaths",
    # Utilities
    "ConfigAccessor",
    "ConfigMerger",
    "WindowBuilder",
    "SampleAligner",
    # Data Handlers
    "DataTypeHandler",
    "DataTypeHandlerFactory",
    "TimeseriesHandler",
    "TabularHandler",
    # Basic steps
    "DataLoaderStep",
    "PreprocessorStep",
    "TrainerStep",
    "EvaluatorStep",
    "InferenceStep",
    "LoggerStep",
    "TunerStep",
    "ProfilingStep",
    # Framework model steps
    "FrameworkModelStep",
    "ClusteringModelStep",
    # Advanced steps
    "ParallelStep",
    "BranchStep",
    "SubPipelineStep",
    "DynamicAdapterStep",
    "MLflowLoaderStep",
    "DataModuleStep",
    "FeatureInferenceStep",
]
