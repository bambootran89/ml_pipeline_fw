"""
Feature retrieval strategies for different data types.

This module implements the Strategy pattern to handle timeseries
and tabular data loading from Feast with zero duplication.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from mlproject.src.features.base import BaseFeatureStore
from mlproject.src.features.timeseries import TimeSeriesFeatureStore


class FeatureRetrievalStrategy(ABC):
    """
    Abstract strategy for feature retrieval from Feast.

    Implementations must define how to fetch features for a
    specific data type (timeseries, tabular, etc).
    """

    @abstractmethod
    def retrieve(
        self,
        store: BaseFeatureStore,
        features: List[str],
        entity_key: str,
        entity_ids: List[Union[int, str]],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Retrieve features from the feature store.

        Parameters
        ----------
        store : BaseFeatureStore
            Initialized feature store client.
        features : List[str]
            Fully qualified feature references.
        entity_key : str
            Entity join key name.
        entity_ids : List[Union[int, str]]
            Entity identifier value.
        config : Dict[str, Any]
            Additional configuration parameters.

        Returns
        -------
        pd.DataFrame
            Retrieved features.
        """


class TimeseriesRetrievalStrategy(FeatureRetrievalStrategy):
    """Strategy for timeseries feature retrieval with sequence windows."""

    def retrieve(
        self,
        store: BaseFeatureStore,
        features: List[str],
        entity_key: str,
        entity_ids: List[Union[int, str]],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Retrieve timeseries features as a sequence window."""
        ts_store = TimeSeriesFeatureStore(
            store=store,
            default_entity_key=entity_key,
            default_entity_id=entity_ids[0],
        )

        # Extract time range
        start_date: datetime = datetime.fromisoformat(config["start_date"])
        end_date: datetime = datetime.fromisoformat(config["end_date"])

        return ts_store.get_sequence_by_range(
            features=features,
            start_date=start_date,
            end_date=end_date,
        )


class TabularRetrievalStrategy(FeatureRetrievalStrategy):
    """Strategy for tabular feature retrieval with entity-based joins."""

    def retrieve(
        self,
        store: BaseFeatureStore,
        features: List[str],
        entity_key: str,
        entity_ids: List[Union[int, str]],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Retrieve tabular features via historical or online lookup."""
        entity_data_path: str = config.get("entity_data", "")

        if not entity_data_path:
            raise ValueError("Tabular strategy requires 'entity_data' path")

        # Load entity reference data
        if entity_data_path.endswith(".csv"):
            entity_df = pd.read_csv(entity_data_path)
        elif entity_data_path.endswith(".parquet"):
            entity_df = pd.read_parquet(entity_data_path)
        else:
            raise ValueError(f"Unsupported format: {entity_data_path}")

        index_col: str = config.get("index_col", "event_timestamp")
        cols = [entity_key, index_col] + config.get("target_columns", [])
        entity_df = entity_df[cols]

        df = store.get_historical_features(
            entity_df=entity_df,
            features=features,
        )

        return df


class OnlineRetrievalStrategy(FeatureRetrievalStrategy):
    """
    Strategy for online/real-time feature retrieval.

    Supports two modes:
    1. Tabular: Single point from Online Store (get_online_features)
    2. Timeseries: Latest sequence window (get_latest_n_sequence)

    This strategy is used by serving pipelines for real-time inference.
    """

    def __init__(self, time_point: Optional[str] = "now"):
        """
        Initialize online retrieval strategy.

        Parameters
        ----------
        time_point : Optional[str], default="now"
            Time point for timeseries sequence retrieval.
            Can be "now", ISO datetime string, or Unix timestamp.
        """
        self.time_point = time_point

    def retrieve(
        self,
        store: BaseFeatureStore,
        features: List[str],
        entity_key: str,
        entity_ids: List[Union[int, str]],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Retrieve features for online serving.

        Parameters
        ----------
        store : BaseFeatureStore
            Feature store instance.
        features : List[str]
            Fully qualified feature references.
        entity_key : str
            Entity column name.
        entity_id : Union[int, str]
            Entity identifier value.
        config : Dict[str, Any]
            Configuration dictionary containing:
            - data_type: "timeseries" or "tabular"
            - featureview: Feature view name
            - features: List of feature names

            For timeseries only:
            - input_chunk_length: Sequence window size
            - frequency_hours: Data frequency in hours
            - index_col: Timestamp column name
            - end_date: Fallback date if no data at time_point

        Returns
        -------
        pd.DataFrame
            Retrieved features ready for preprocessing.

        Raises
        ------
        ValueError
            If required config keys are missing or no data found.
        """
        # Validate required config keys
        features_from_config = config.get("features", [])
        featureview = config.get("featureview")

        if not features_from_config or not featureview:
            raise ValueError("Config must contain 'features' and 'featureview'")

        data_type = config.get("data_type", "timeseries")

        if data_type != "timeseries":
            return self._retrieve_tabular(store, features, entity_key, entity_ids)

        return self._retrieve_timeseries(
            store, features, entity_key, entity_ids, config
        )

    def _retrieve_tabular(
        self,
        store: BaseFeatureStore,
        features: List[str],
        entity_key: str,
        entity_ids: List[Union[int, str]],
    ) -> pd.DataFrame:
        """
        Retrieve single point from Online Store.

        Parameters
        ----------
        store : BaseFeatureStore
            Feature store instance.
        features : List[str]
            Feature references in format "view:feature".
        entity_key : str
            Entity column name.
        entity_id : Union[int, str]
            Entity identifier value.

        Returns
        -------
        pd.DataFrame
            Single row with latest feature values.

        Raises
        ------
        ValueError
            If no online data found for the entity.
        """
        online_data = store.get_online_features(
            entity_rows=[{entity_key: entity_id} for entity_id in entity_ids],
            features=features,
        )

        if not online_data:
            raise ValueError(f"No online data found for {entity_key}={entity_ids}")

        return pd.DataFrame(online_data)

    def _retrieve_timeseries(
        self,
        store: BaseFeatureStore,
        features: List[str],
        entity_key: str,
        entity_ids: List[Union[int, str]],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Retrieve sequence window for timeseries serving.

        Parameters
        ----------
        store : BaseFeatureStore
            Feature store instance.
        features : List[str]
            Feature references in format "view:feature".
        entity_key : str
            Entity column name.
        entity_id : Union[int, str]
            Entity identifier value.
        config : Dict[str, Any]
            Configuration with timeseries-specific parameters.

        Returns
        -------
        pd.DataFrame
            Indexed DataFrame with sequence window.

        Raises
        ------
        ValueError
            If no data found at time_point or fallback end_date.
        """
        win_size = config.get("input_chunk_length", 24)
        frequency_hours = config.get("frequency_hours", 1)
        end_date = config.get("end_date")
        index_col = config.get("index_col", "event_timestamp")

        # Wrap store with timeseries interface
        ts_store = TimeSeriesFeatureStore(
            store=store,
            default_entity_key=entity_key,
            default_entity_ids=entity_ids,
        )

        # Retrieve sequence at time_point
        assert self.time_point is not None
        df = ts_store.get_latest_n_sequence(
            features=features,
            n_points=win_size + (24 // frequency_hours),
            frequency_hours=frequency_hours,
            time_point=self.time_point,
            entity_ids=entity_ids,
        )

        # Fallback to config end_date if no data
        cols = config.get("features", [])
        if (df.empty or df[cols].isna().all().all()) and end_date:
            df = ts_store.get_latest_n_sequence(
                features=features,
                n_points=win_size + (24 // frequency_hours),
                frequency_hours=frequency_hours,
                time_point=end_date,
                entity_ids=entity_ids,
            )

        if df.empty:
            raise ValueError(f"No timeseries data at time_point={self.time_point}")

        # Set index
        df = df.set_index(index_col)
        return df


class StrategyFactory:
    """
    Factory class to instantiate feature retrieval strategy.

    Depending on the data type and mode, this factory returns a
    strategy instance that handles feature retrieval.
    """

    @staticmethod
    def create(
        data_type: str,
        mode: str = "historical",
        time_point: Optional[str] = None,
    ) -> FeatureRetrievalStrategy:
        """
        Create a feature retrieval strategy based on data type and mode.

        Parameters
        ----------
        data_type : str
            Type of the data ('timeseries' or 'tabular').
        mode : str, default="historical"
            Retrieval mode, either 'historical' or 'online'.
        time_point : Optional[str], default=None
            Timestamp for online retrieval (only used if mode='online').

        Returns
        -------
        FeatureRetrievalStrategy
            An instance of a FeatureRetrievalStrategy subclass.

        Raises
        ------
        ValueError
            If data_type is not recognized.
        """
        if mode == "online":
            return OnlineRetrievalStrategy(time_point)

        if data_type == "timeseries":
            return TimeseriesRetrievalStrategy()
        if data_type == "tabular":
            return TabularRetrievalStrategy()

        raise ValueError(f"Invalid data_type: {data_type}")
