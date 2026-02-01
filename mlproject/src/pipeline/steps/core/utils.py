"""
Utility classes for pipeline steps.

This module provides reusable utilities for common operations
across pipeline steps, reducing code duplication and improving
maintainability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.pipeline.steps.core.constants import DataTypes, DefaultValues


class ConfigAccessor:
    """Centralized configuration accessor.

    Provides consistent interface for accessing configuration values
    with proper defaults and type handling.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize config accessor.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object.
        """
        self.cfg = cfg

    def get_data_type(self) -> str:
        """Get data type (timeseries or tabular).

        Returns
        -------
        str
            Normalized data type.
        """
        data_cfg = self.cfg.get("data", {})
        data_type = str(data_cfg.get("type", DataTypes.TIMESERIES))
        return DataTypes.normalize(data_type)

    def is_timeseries(self) -> bool:
        """Check if data type is timeseries.

        Returns
        -------
        bool
            True if timeseries, False otherwise.
        """
        return DataTypes.is_timeseries(self.get_data_type())

    def is_tabular(self) -> bool:
        """Check if data type is tabular.

        Returns
        -------
        bool
            True if tabular, False otherwise.
        """
        return DataTypes.is_tabular(self.get_data_type())

    def get_feature_columns(self) -> list[str]:
        """Get feature column names.

        Returns
        -------
        list[str]
            Feature column names.
        """
        data_cfg = self.cfg.get("data", {})
        return list(data_cfg.get("features", []))

    def get_target_columns(self) -> list[str]:
        """Get target column names.

        Returns
        -------
        list[str]
            Target column names.
        """
        data_cfg = self.cfg.get("data", {})
        return list(data_cfg.get("target_columns", []))

    def get_entity_key(self) -> str:
        """Get entity/group key for timeseries.

        Returns
        -------
        str
            Entity key column name.
        """
        data_cfg = self.cfg.get("data", {})
        return str(data_cfg.get("entity_key", DefaultValues.ENTITY_KEY))

    def get_window_config(self) -> Dict[str, int]:
        """Get window configuration for timeseries.

        Returns
        -------
        Dict[str, int]
            Window configuration with keys:
            - input_chunk_length
            - output_chunk_length
            - stride
        """
        hyperparams = self.cfg.experiment.get("hyperparams", {})
        return {
            "input_chunk_length": int(
                hyperparams.get("input_chunk_length", DefaultValues.INPUT_CHUNK_LENGTH)
            ),
            "output_chunk_length": int(
                hyperparams.get(
                    "output_chunk_length", DefaultValues.OUTPUT_CHUNK_LENGTH
                )
            ),
            "stride": int(hyperparams.get("stride", DefaultValues.STRIDE)),
        }

    def get_hyperparams(self) -> Dict[str, Any]:
        """Get all hyperparameters.

        Returns
        -------
        Dict[str, Any]
            Hyperparameters dictionary.
        """
        return dict(self.cfg.experiment.get("hyperparams", {}))

    def get_model_name(self) -> str:
        """Get model name.

        Returns
        -------
        str
            Model name.
        """
        return str(self.cfg.experiment.get("model", "")).lower()

    def get_model_type(self) -> str:
        """Get model type (ml/dl).

        Returns
        -------
        str
            Model type.
        """
        return str(self.cfg.experiment.get("model_type", "ml")).lower()

    def get_artifacts_dir(self) -> str:
        """Get artifacts directory path.

        Returns
        -------
        str
            Artifacts directory path.
        """
        training_cfg = self.cfg.get("training", {})
        return str(training_cfg.get("artifacts_dir", DefaultValues.ARTIFACTS_DIR))

    def get_n_splits(self) -> int:
        """Get number of CV splits for tuning.

        Returns
        -------
        int
            Number of splits.
        """
        tuning_cfg = self.cfg.get("tuning", {})
        return int(tuning_cfg.get("n_splits", DefaultValues.N_SPLITS))

    def get_optimize_metric(self) -> str:
        """Get optimization metric for tuning.

        Returns
        -------
        str
            Metric name to optimize.
        """
        tuning_cfg = self.cfg.get("tuning", {})
        return str(tuning_cfg.get("optimize_metric", "mse"))


class WindowBuilder:
    """Utility for creating sliding windows from timeseries data.

    This class provides a unified interface for window creation,
    eliminating duplicate implementations across multiple steps.
    """

    @staticmethod
    def create_windows(
        data: pd.DataFrame | np.ndarray,
        input_chunk: int,
        output_chunk: int,
        stride: Optional[int] = None,
    ) -> np.ndarray:
        """Create sliding windows from input data.

        Parameters
        ----------
        data : pd.DataFrame | np.ndarray
            Input data to window.
        input_chunk : int
            Window input length (lookback period).
        output_chunk : int
            Window output length (forecast horizon).
        stride : Optional[int]
            Stride for sliding window. If None, uses output_chunk.

        Returns
        -------
        np.ndarray
            Windowed array with shape [n_windows, input_chunk, n_features].

        Raises
        ------
        ValueError
            If data has fewer rows than input_chunk.
        """
        if stride is None:
            stride = output_chunk

        # Convert to numpy if DataFrame
        if isinstance(data, pd.DataFrame):
            arr = data.values
        else:
            arr = data

        n_rows = arr.shape[0]
        if n_rows < input_chunk:
            raise ValueError(
                f"Insufficient data: need at least {input_chunk} rows, got {n_rows}"
            )

        # Create windows
        windows = []
        for start in range(0, n_rows - input_chunk + 1, stride):
            window = arr[start : start + input_chunk]
            windows.append(window)

        return np.array(windows, dtype=np.float32)

    @staticmethod
    def create_windows_with_targets(
        data: pd.DataFrame | np.ndarray,
        input_chunk: int,
        output_chunk: int,
        stride: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows with corresponding target windows.

        Parameters
        ----------
        data : pd.DataFrame | np.ndarray
            Input data to window.
        input_chunk : int
            Window input length.
        output_chunk : int
            Window output length.
        stride : Optional[int]
            Stride for sliding window. If None, uses output_chunk.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - X: Input windows [n_windows, input_chunk, n_features]
            - y: Target windows [n_windows, output_chunk, n_targets]

        Raises
        ------
        ValueError
            If data has fewer rows than input_chunk + output_chunk.
        """
        if stride is None:
            stride = output_chunk

        # Convert to numpy if DataFrame
        if isinstance(data, pd.DataFrame):
            arr = data.values
        else:
            arr = data

        n_rows = arr.shape[0]
        min_required = input_chunk + output_chunk
        if n_rows < min_required:
            raise ValueError(
                f"Insufficient data: need at least {min_required} rows, got {n_rows}"
            )

        # Create windows
        x_windows = []
        y_windows = []

        for start in range(0, n_rows - min_required + 1, stride):
            x_window = arr[start : start + input_chunk]
            y_window = arr[start + input_chunk : start + input_chunk + output_chunk]
            x_windows.append(x_window)
            y_windows.append(y_window)

        return (
            np.array(x_windows, dtype=np.float32),
            np.array(y_windows, dtype=np.float32),
        )

    @staticmethod
    def create_grouped_windows(
        data: pd.DataFrame,
        group_key: str,
        input_chunk: int,
        output_chunk: int,
        stride: Optional[int] = None,
    ) -> np.ndarray:
        """Create windows for grouped timeseries data.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame with group column.
        group_key : str
            Column name for grouping.
        input_chunk : int
            Window input length.
        output_chunk : int
            Window output length.
        stride : Optional[int]
            Stride for sliding window.

        Returns
        -------
        np.ndarray
            Concatenated windows from all groups.
        """
        if group_key not in data.columns:
            # No grouping, process as single series
            return WindowBuilder.create_windows(data, input_chunk, output_chunk, stride)

        windows = []
        for _, group in data.groupby(group_key):
            # Drop group column for windowing
            group_clean = group.drop(columns=[group_key], errors="ignore")
            try:
                group_windows = WindowBuilder.create_windows(
                    group_clean, input_chunk, output_chunk, stride
                )
                windows.append(group_windows)
            except ValueError as e:
                # Skip groups with insufficient data
                print(f"Skipping group due to: {e}")
                continue

        if not windows:
            raise ValueError("No valid windows created from any group")

        return np.vstack(windows).astype(np.float32)


class SampleAligner:
    """Utility for aligning samples across different data sources.

    Handles alignment of features from multiple sources with
    different shapes and dimensions.
    """

    @staticmethod
    def align_samples(
        base_data: pd.DataFrame | np.ndarray,
        additional_data: pd.DataFrame | np.ndarray,
        method: str = "auto",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align base and additional data samples.

        Parameters
        ----------
        base_data : pd.DataFrame | np.ndarray
            Base data to align to.
        additional_data : pd.DataFrame | np.ndarray
            Additional data to align.
        method : str
            Alignment method: "auto", "broadcast", "repeat", "pad".

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Aligned (base, additional) arrays.

        Raises
        ------
        ValueError
            If alignment is not possible with given method.
        """
        # Convert to numpy
        base_arr = (
            base_data.values if isinstance(base_data, pd.DataFrame) else base_data
        )
        add_arr = (
            additional_data.values
            if isinstance(additional_data, pd.DataFrame)
            else additional_data
        )

        # If already aligned, return as-is
        if base_arr.shape[0] == add_arr.shape[0]:
            return base_arr, add_arr

        # Auto-detect method
        if method == "auto":
            if add_arr.shape[0] == 1:
                method = "broadcast"
            elif base_arr.shape[0] % add_arr.shape[0] == 0:
                method = "repeat"
            else:
                method = "pad"

        # Apply alignment
        if method == "broadcast":
            # Broadcast single sample to match base
            aligned_add = np.broadcast_to(add_arr, base_arr.shape)
            return base_arr, aligned_add

        elif method == "repeat":
            # Repeat additional data to match base length
            repeat_factor = base_arr.shape[0] // add_arr.shape[0]
            aligned_add = np.repeat(add_arr, repeat_factor, axis=0)
            # Handle remainder
            if aligned_add.shape[0] < base_arr.shape[0]:
                remainder = base_arr.shape[0] - aligned_add.shape[0]
                aligned_add = np.vstack([aligned_add, add_arr[:remainder]])
            return base_arr, aligned_add

        elif method == "pad":
            # Pad with last value to match length
            if add_arr.shape[0] < base_arr.shape[0]:
                pad_len = base_arr.shape[0] - add_arr.shape[0]
                last_val = add_arr[-1:].copy()
                padding = np.repeat(last_val, pad_len, axis=0)
                aligned_add = np.vstack([add_arr, padding])
            else:
                aligned_add = add_arr[: base_arr.shape[0]]
            return base_arr, aligned_add

        else:
            raise ValueError(f"Unknown alignment method: {method}")

    @staticmethod
    def align_multiple(
        base_data: pd.DataFrame | np.ndarray,
        additional_sources: Dict[str, pd.DataFrame | np.ndarray],
        method: str = "auto",
    ) -> Dict[str, np.ndarray]:
        """Align multiple additional data sources to base.

        Parameters
        ----------
        base_data : pd.DataFrame | np.ndarray
            Base data to align to.
        additional_sources : Dict[str, pd.DataFrame | np.ndarray]
            Dictionary of additional data sources.
        method : str
            Alignment method.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with "base" and aligned additional sources.
        """
        base_arr = (
            base_data.values if isinstance(base_data, pd.DataFrame) else base_data
        )
        result = {"base": base_arr}

        for name, data in additional_sources.items():
            _, aligned = SampleAligner.align_samples(base_data, data, method)
            result[name] = aligned

        return result


class ConfigMerger:
    """Utility for merging OmegaConf configurations.

    Provides unified methods for merging configurations from
    different sources (files, inline dicts, simple overrides).
    """

    @staticmethod
    def merge_external_file(
        base_cfg: DictConfig, file_path: str, search_paths: Optional[list[str]] = None
    ) -> DictConfig:
        """Merge external configuration file.

        Parameters
        ----------
        base_cfg : DictConfig
            Base configuration.
        file_path : str
            Path to external config file.
        search_paths : Optional[list[str]]
            Additional paths to search for the file.

        Returns
        -------
        DictConfig
            Merged configuration.

        Raises
        ------
        FileNotFoundError
            If config file not found.
        """

        config_path = Path(file_path)

        if not config_path.exists():
            # Try search paths
            search_paths = search_paths or [
                "mlproject",
                ".",
            ]

            for search_dir in search_paths:
                alt_path = Path(search_dir) / file_path
                if alt_path.exists():
                    config_path = alt_path
                    break
            else:
                tried_paths = [str(Path(sp) / file_path) for sp in search_paths]
                raise FileNotFoundError(
                    f"Config file not found: '{file_path}'. "
                    f"Tried paths: {tried_paths}"
                )

        # Load and merge
        external_cfg = OmegaConf.load(config_path)
        return cast(DictConfig, OmegaConf.merge(base_cfg, external_cfg))

    @staticmethod
    def merge_inline_config(
        base_cfg: DictConfig, inline_config: Dict[str, Any]
    ) -> DictConfig:
        """Merge inline configuration dictionary.

        Parameters
        ----------
        base_cfg : DictConfig
            Base configuration.
        inline_config : Dict[str, Any]
            Inline configuration to merge.

        Returns
        -------
        DictConfig
            Merged configuration.
        """

        inline_omega = OmegaConf.create(inline_config)
        return cast(DictConfig, OmegaConf.merge(base_cfg, inline_omega))

    @staticmethod
    def apply_simple_overrides(base_cfg: DictConfig, **overrides: Any) -> DictConfig:
        """Apply simple key-value overrides.

        Parameters
        ----------
        base_cfg : DictConfig
            Base configuration.
        **overrides
            Key-value pairs to override.

        Returns
        -------
        DictConfig
            Configuration with overrides applied.
        """

        override_cfg = OmegaConf.create(overrides)
        return cast(DictConfig, OmegaConf.merge(base_cfg, override_cfg))

    @staticmethod
    def merge_model_config(
        base_cfg: DictConfig,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        data_config: Optional[Dict[str, Any]] = None,
    ) -> DictConfig:
        """Merge model-specific configuration.

        Convenience method for merging common model configuration
        parameters into the experiment structure.

        Parameters
        ----------
        base_cfg : DictConfig
            Base configuration.
        model_name : Optional[str]
            Model name to set.
        model_type : Optional[str]
            Model type (ml/dl) to set.
        hyperparams : Optional[Dict[str, Any]]
            Hyperparameters to merge.
        data_config : Optional[Dict[str, Any]]
            Data configuration to merge.

        Returns
        -------
        DictConfig
            Merged configuration.
        """

        # Ensure experiment node exists
        if "experiment" not in base_cfg:
            base_cfg.experiment = {}

        # Apply model name
        if model_name is not None:
            base_cfg.experiment.model = model_name

        # Apply model type
        if model_type is not None:
            base_cfg.experiment.model_type = model_type

        # Merge hyperparams
        if hyperparams is not None:
            if "hyperparams" not in base_cfg.experiment:
                base_cfg.experiment.hyperparams = {}
            for key, value in hyperparams.items():
                base_cfg.experiment.hyperparams[key] = value

        # Merge data config
        if data_config is not None:
            base_cfg = cast(
                DictConfig, OmegaConf.merge(base_cfg, {"data": data_config})
            )

        return base_cfg
