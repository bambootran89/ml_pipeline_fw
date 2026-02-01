"""
Data type handlers for pipeline steps.

This module provides a strategy pattern for handling different
data types (timeseries, tabular) with type-specific logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.pipeline.steps.core.constants import DataTypes
from mlproject.src.pipeline.steps.core.utils import ConfigAccessor, WindowBuilder


class DataTypeHandler(ABC):
    """Abstract base class for data type-specific handlers.

    This provides a strategy pattern for encapsulating data type-specific
    logic that differs between timeseries and tabular data.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize handler.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object.
        """
        self.cfg = cfg
        self.config_accessor = ConfigAccessor(cfg)

    @abstractmethod
    def prepare_model_input(self, features: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Prepare input data for model.

        Parameters
        ----------
        features : pd.DataFrame
            Input features.
        **kwargs
            Additional parameters.

        Returns
        -------
        np.ndarray
            Prepared model input.
        """
        raise NotImplementedError

    @abstractmethod
    def should_attach_targets(self) -> bool:
        """Check if targets should be attached to features.

        Returns
        -------
        bool
            True if targets should be attached, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def requires_windowing(self) -> bool:
        """Check if data requires windowing.

        Returns
        -------
        bool
            True if windowing required, False otherwise.
        """
        raise NotImplementedError


class TimeseriesHandler(DataTypeHandler):
    """Handler for timeseries data.

    Implements timeseries-specific logic including windowing,
    grouped processing, and sequence preparation.
    """

    def prepare_model_input(
        self,
        features: pd.DataFrame,
        entity_key: Optional[str] = None,
        window_config: Optional[Dict[str, int]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Prepare timeseries input with windowing.

        Parameters
        ----------
        features : pd.DataFrame
            Input features.
        entity_key : Optional[str]
            Column for grouping timeseries.
        window_config : Optional[Dict[str, int]]
            Window configuration with input_chunk_length, output_chunk_length.
        **kwargs
            Additional parameters.

        Returns
        -------
        np.ndarray
            Windowed input array.
        """
        if entity_key is None:
            entity_key = self.config_accessor.get_entity_key()

        if window_config is None:
            window_config = self.config_accessor.get_window_config()

        input_chunk = window_config.get("input_chunk_length", 24)
        output_chunk = window_config.get("output_chunk_length", 6)
        stride = window_config.get("stride", output_chunk)

        # Check if we have entity/group column
        if entity_key in features.columns:
            return WindowBuilder.create_grouped_windows(
                features, entity_key, input_chunk, output_chunk, stride
            )

        # Single timeseries
        return WindowBuilder.create_windows(features, input_chunk, output_chunk, stride)

    def create_windows_for_training(
        self,
        features: pd.DataFrame,
        targets: Optional[pd.DataFrame] = None,
        entity_key: Optional[str] = None,
        window_config: Optional[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create windows for training with X and y.

        Parameters
        ----------
        features : pd.DataFrame
            Input features.
        targets : Optional[pd.DataFrame]
            Target values.
        entity_key : Optional[str]
            Column for grouping.
        window_config : Optional[Dict[str, int]]
            Window configuration.

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            X windows and optional y windows.
        """
        if entity_key is None:
            entity_key = self.config_accessor.get_entity_key()

        if window_config is None:
            window_config = self.config_accessor.get_window_config()

        input_chunk = window_config.get("input_chunk_length", 24)
        output_chunk = window_config.get("output_chunk_length", 6)

        # Combine features and targets for windowing
        if targets is not None:
            combined = pd.concat([features, targets], axis=1)
        else:
            combined = features

        # Create windows
        if entity_key in combined.columns:
            return self._create_grouped_windows(
                combined, entity_key, targets is not None, input_chunk, output_chunk
            )

        # Single timeseries
        if targets is not None:
            return WindowBuilder.create_windows_with_targets(
                combined, input_chunk, output_chunk
            )

        return WindowBuilder.create_windows(combined, input_chunk, output_chunk), None

    def _create_grouped_windows(
        self,
        combined: pd.DataFrame,
        entity_key: str,
        has_targets: bool,
        input_chunk: int,
        output_chunk: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create windows for grouped data."""
        x_all = []
        y_all = []

        for _, group in combined.groupby(entity_key):
            group_clean = group.drop(columns=[entity_key], errors="ignore")

            try:
                if has_targets:
                    x_win, y_win = WindowBuilder.create_windows_with_targets(
                        group_clean, input_chunk, output_chunk
                    )
                    x_all.append(x_win)
                    y_all.append(y_win)
                else:
                    x_win = WindowBuilder.create_windows(
                        group_clean, input_chunk, output_chunk
                    )
                    x_all.append(x_win)
            except ValueError:
                continue

        x_windows = np.vstack(x_all) if x_all else np.array([])
        y_windows = np.vstack(y_all) if y_all and has_targets else None

        return x_windows, y_windows

    def should_attach_targets(self) -> bool:
        """Timeseries typically doesn't attach targets separately.

        Returns
        -------
        bool
            False for timeseries.
        """
        return False

    def requires_windowing(self) -> bool:
        """Timeseries requires windowing.

        Returns
        -------
        bool
            True for timeseries.
        """
        return True


class TabularHandler(DataTypeHandler):
    """Handler for tabular data.

    Implements tabular-specific logic with straightforward
    array conversion and target attachment.
    """

    def prepare_model_input(self, features: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Prepare tabular input (simple conversion to array).

        Parameters
        ----------
        features : pd.DataFrame
            Input features.
        **kwargs
            Additional parameters (unused).

        Returns
        -------
        np.ndarray
            Feature array.
        """
        return features.values.astype(np.float32)

    def prepare_with_targets(
        self, features: pd.DataFrame, targets: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine features and targets for tabular data.

        Parameters
        ----------
        features : pd.DataFrame
            Input features.
        targets : pd.DataFrame
            Target values.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame.
        """
        return pd.concat([features, targets], axis=1)

    def should_attach_targets(self) -> bool:
        """Tabular data attaches targets to features.

        Returns
        -------
        bool
            True for tabular.
        """
        return True

    def requires_windowing(self) -> bool:
        """Tabular data doesn't require windowing.

        Returns
        -------
        bool
            False for tabular.
        """
        return False


class DataTypeHandlerFactory:
    """Factory for creating data type handlers.

    Provides a central point for instantiating the correct
    handler based on data type configuration.
    """

    @staticmethod
    def create(cfg: DictConfig) -> DataTypeHandler:
        """Create handler for configured data type.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object.

        Returns
        -------
        DataTypeHandler
            Handler instance for the data type.

        Raises
        ------
        ValueError
            If data type is not supported.
        """
        accessor = ConfigAccessor(cfg)
        data_type = accessor.get_data_type()

        if DataTypes.is_timeseries(data_type):
            return TimeseriesHandler(cfg)
        elif DataTypes.is_tabular(data_type):
            return TabularHandler(cfg)
        else:
            raise ValueError(
                f"Unsupported data type: {data_type}. "
                f"Supported types: {DataTypes.TIMESERIES}, {DataTypes.TABULAR}"
            )

    @staticmethod
    def create_from_type(data_type: str, cfg: DictConfig) -> DataTypeHandler:
        """Create handler for specific data type.

        Parameters
        ----------
        data_type : str
            Data type name.
        cfg : DictConfig
            Configuration object.

        Returns
        -------
        DataTypeHandler
            Handler instance.

        Raises
        ------
        ValueError
            If data type is not supported.
        """
        normalized_type = DataTypes.normalize(data_type)

        if normalized_type == DataTypes.TIMESERIES:
            return TimeseriesHandler(cfg)
        elif normalized_type == DataTypes.TABULAR:
            return TabularHandler(cfg)
        else:
            raise ValueError(
                f"Unsupported data type: {data_type}. "
                f"Supported types: {DataTypes.TIMESERIES}, {DataTypes.TABULAR}"
            )
