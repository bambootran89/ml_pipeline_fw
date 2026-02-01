"""
Constants for pipeline steps.

This module centralizes all magic strings and hardcoded values
used across pipeline steps to improve maintainability and
reduce errors from typos.
"""

from __future__ import annotations


class ContextKeys:
    """Context key constants for pipeline execution.

    These keys are used to store and retrieve data from the pipeline context
    dictionary that is passed between steps.
    """

    # Internal metadata keys
    ARTIFACT_REGISTRY = "_artifact_registry"
    COMPOSED_FEATURE_NAMES = "_composed_feature_names"
    ADDITIONAL_FEATURE_KEYS = "_additional_feature_keys"

    # Data processing flags
    IS_SPLITED_INPUT = "is_splited_input"
    FEATURE_COLUMNS_SIZE = "feature_columns_size"

    # Common data keys
    FEATURES = "features"
    TARGETS = "targets"
    PREPROCESSED_DATA = "preprocessed_data"
    DATAMODULE = "datamodule"
    MODEL = "model"
    PREDICTIONS = "predictions"
    METRICS = "metrics"

    @staticmethod
    def step_metrics(step_id: str) -> str:
        """Generate metrics key for a specific step.

        Parameters
        ----------
        step_id : str
            Step identifier.

        Returns
        -------
        str
            Context key for step metrics.
        """
        return f"{step_id}_metrics"

    @staticmethod
    def step_best_params(step_id: str) -> str:
        """Generate best parameters key for a specific step.

        Parameters
        ----------
        step_id : str
            Step identifier.

        Returns
        -------
        str
            Context key for step best parameters.
        """
        return f"{step_id}_best_params"


class ColumnNames:
    """Column name constants for data processing.

    These are standard column names used across the pipeline
    for data splits, entity identification, etc.
    """

    # Data split marker column
    DATASET = "dataset"

    # Common dataset split values
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DataTypes:
    """Data type constants.

    Supported data types for pipeline processing.
    """

    TIMESERIES = "timeseries"
    TABULAR = "tabular"

    @staticmethod
    def normalize(data_type: str) -> str:
        """Normalize data type string to lowercase.

        Parameters
        ----------
        data_type : str
            Raw data type string.

        Returns
        -------
        str
            Normalized lowercase data type.
        """
        return str(data_type).lower()

    @staticmethod
    def is_timeseries(data_type: str) -> bool:
        """Check if data type is timeseries.

        Parameters
        ----------
        data_type : str
            Data type to check.

        Returns
        -------
        bool
            True if timeseries, False otherwise.
        """
        return DataTypes.normalize(data_type) == DataTypes.TIMESERIES

    @staticmethod
    def is_tabular(data_type: str) -> bool:
        """Check if data type is tabular.

        Parameters
        ----------
        data_type : str
            Data type to check.

        Returns
        -------
        bool
            True if tabular, False otherwise.
        """
        return DataTypes.normalize(data_type) == DataTypes.TABULAR


class ModelTypes:
    """Model type constants.

    Supported model types for pipeline processing.
    """

    ML = "ml"
    DL = "dl"
    TIMESERIES = "timeseries"
    MULTIVARIATE = "multivariate"

    # Model types that require 3D input (sequence models)
    SEQUENCE_MODEL_TYPES = frozenset([DL, TIMESERIES, MULTIVARIATE])

    @staticmethod
    def is_sequence_model(model_type: str) -> bool:
        """Check if model type requires sequence (3D) input.

        Parameters
        ----------
        model_type : str
            Model type to check.

        Returns
        -------
        bool
            True if model requires sequence input, False otherwise.
        """
        return model_type in ModelTypes.SEQUENCE_MODEL_TYPES


class DefaultValues:
    """Default values for configuration parameters.

    Centralized location for all default values used throughout
    the pipeline steps.
    """

    # Entity/Group identification
    ENTITY_KEY = "location_id"

    # Window parameters for timeseries
    INPUT_CHUNK_LENGTH = 24
    OUTPUT_CHUNK_LENGTH = 6
    STRIDE = 1

    # MLflow configuration
    MLFLOW_ALIAS = "latest"
    MLFLOW_RUN_NAME_SUFFIX = "_run"
    TUNING_RUN_NAME = "Hparam_Tuning_Experiment"

    # Parallel execution
    MAX_WORKERS = 4

    # Cross-validation
    N_SPLITS = 3

    # Artifact configuration
    ARTIFACTS_DIR = "artifacts/models"
    ARTIFACT_TYPE = "component"


class ConfigPaths:
    """Configuration path constants.

    Standard paths used to access configuration values
    from OmegaConf DictConfig objects.
    """

    # Experiment configuration
    EXPERIMENT = "experiment"
    EXPERIMENT_MODEL = "experiment.model"
    EXPERIMENT_MODEL_TYPE = "experiment.model_type"
    EXPERIMENT_HYPERPARAMS = "experiment.hyperparams"

    # Data configuration
    DATA = "data"
    DATA_TYPE = "data.type"
    DATA_FEATURES = "data.features"
    DATA_TARGET_COLUMNS = "data.target_columns"
    DATA_ENTITY_KEY = "data.entity_key"

    # Training configuration
    TRAINING = "training"
    TRAINING_ARTIFACTS_DIR = "training.artifacts_dir"

    # Tuning configuration
    TUNING = "tuning"
    TUNING_OPTIMIZE_METRIC = "tuning.optimize_metric"
    TUNING_N_SPLITS = "tuning.n_splits"
