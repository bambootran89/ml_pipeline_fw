"""
Unified facade for feature store operations.

This module provides a simplified, high-level interface for
loading features from Feast regardless of data type.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import pandas as pd
from omegaconf import DictConfig

from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.strategies import FeatureRetrievalStrategy, StrategyFactory


class FeatureStoreFacade:
    """
    Simplified facade for Feast feature retrieval.

    This class eliminates complexity by routing data type-specific
    logic to appropriate strategies while maintaining a clean API.
    """

    def __init__(
        self,
        cfg: DictConfig,
        mode: str = "historical",
    ) -> None:
        """
        Initialize facade with configuration.

        Parameters
        ----------
        cfg : DictConfig
            Configuration containing data path and feature specs.
        mode : str, default="historical"
            Retrieval mode: "historical" (training/eval) or
            "online" (serving).
        """
        self.cfg = cfg
        self.data_cfg = self.cfg.get("data", {})
        self.mode = mode
        self._validate_config()

    def load_features(
        self,
        time_point: Optional[str] = None,
        entity_ids: Optional[List[Union[int, str]]] = None,
    ) -> pd.DataFrame:
        """
        Load features from Feast using configuration metadata.

        Parameters
        ----------
        time_point : Optional[str], default=None
            Time point for online timeseries retrieval.
            Only used when mode="online" and data_type="timeseries".
            Can be "now", ISO datetime string, or Unix timestamp.
        entity_ids: Optional[List[Union[int, str]]]
        Returns
        -------
        pd.DataFrame
            Retrieved features ready for preprocessing.

        Raises
        ------
        ValueError
            If configuration is incomplete or invalid.
        """
        # Parse Feast URI
        uri: str = self.data_cfg.get("path", "")
        parsed = urlparse(uri)

        if not parsed.scheme or parsed.scheme != "feast":
            raise ValueError(f"Invalid Feast URI: {uri}")

        repo_name: str = parsed.netloc

        # Validate required config
        self._validate_config()

        # Initialize store
        store = FeatureStoreFactory.create(
            store_type="feast",
            repo_path=repo_name,
        )

        # Build feature references
        featureview: str = self.data_cfg["featureview"]
        features: List[str] = [f"{featureview}:{f}" for f in self.data_cfg["features"]]
        # target_columns: List[str] = [
        #     f"{featureview}:{f}" for f in self.data_cfg.get("target_columns", [])
        # ]
        # features = list(set(only_features + target_columns))

        # Extract entity metadata
        entity_key, cfg_entity_ids = self._resolve_entity()
        if entity_ids is None:
            entity_ids = cfg_entity_ids

        # Prepare strategy config
        strategy_config = self._build_strategy_config()

        # Create strategy (not pass store!)
        strategy = self._create_strategy(time_point)

        # Retrieve features (pass store here!)
        df = strategy.retrieve(
            store=store,
            features=features,
            entity_key=entity_key,
            entity_ids=entity_ids,
            config=strategy_config,
        )

        return self._post_process(df)

    def _create_strategy(
        self,
        time_point: Optional[str] = None,
    ) -> FeatureRetrievalStrategy:
        """
        Create appropriate retrieval strategy.

        Parameters
        ----------
        time_point : Optional[str], default=None
            Time point for online mode.

        Returns
        -------
        FeatureRetrievalStrategy
            Strategy instance based on mode and data_type.
        """
        data_type = self.cfg.data.get("type", "timeseries")

        return StrategyFactory.create(
            data_type=data_type,
            mode=self.mode,
            time_point=time_point,
        )

    def _validate_config(self) -> None:
        """Validate required configuration keys."""
        required = ["featureview", "features"]
        missing = [k for k in required if k not in self.data_cfg]

        if missing:
            raise ValueError(f"Missing config keys: {missing}")

    def _resolve_entity(self) -> tuple[str, List[Union[int, str]]]:
        """
        Resolve entity key and ID from configuration.

        Returns
        -------
        tuple[str, Union[int, str]]
            Entity key name and entity ID value.
        """
        entity_key: str = self.data_cfg.get("entity_key", "location_id")
        entity_ids: List[Union[int, str]] = self.data_cfg.get(
            "entity_ids", [1, 2, 3, 4]
        )
        return entity_key, entity_ids

    def _build_strategy_config(self) -> Dict[str, Any]:
        """
        Build configuration dict for strategy execution.

        Returns
        -------
        Dict[str, Any]
            Strategy-specific configuration.
        """
        config = {
            "start_date": self.data_cfg.get("start_date", ""),
            "end_date": self.data_cfg.get("end_date", ""),
            "entity_data": self.data_cfg.get("entity_data", ""),
            "index_col": self.data_cfg.get("index_col", "event_timestamp"),
            "data_type": self.data_cfg.get("type", "timeseries"),
        }

        # Add online-specific config for timeseries
        if self.mode == "online":
            exp_cfg = self.cfg.get("experiment", {})
            hyperparams = exp_cfg.get("hyperparams", {})

            config.update(
                {
                    "input_chunk_length": hyperparams.get("input_chunk_length", 24),
                    "frequency_hours": self.data_cfg.get("frequency_hours", 1),
                    "featureview": self.data_cfg.get("featureview", ""),
                    "features": self.data_cfg.get("features", []),
                }
            )
        else:
            config.update({"target_columns": self.data_cfg.get("target_columns", [])})

        return config

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply post-processing to retrieved features.

        Parameters
        ----------
        df : pd.DataFrame
            Raw features from Feast.

        Returns
        -------
        pd.DataFrame
            Processed features ready for modeling.
        """
        index_col = self.data_cfg.get("index_col", "event_timestamp")

        # Set index if not already set
        if index_col in df.columns:
            df = df.set_index(index_col)

        # # Filter to requested features
        # feature_cols = self.data_cfg.get("features", [])
        # target_cols = self.data_cfg.get("target_columns", [])
        # requested = set(feature_cols + target_cols)
        # available = [c for c in requested if c in df.columns]

        # if available:
        #     return df[available]

        return df
