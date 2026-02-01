"""
Feast Feature Store dataset loader for offline historical time-series features.

Supports:
- 'timeseries': Fetch historical sequences using TimeSeriesFeatureStore.
- 'tabular': Fetch latest or historical features for entities.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from omegaconf import DictConfig

from mlproject.src.dataio.base import BaseDatasetLoader
from mlproject.src.features.facade import FeatureStoreFacade

logger = logging.getLogger(__name__)


class FeastDatasetLoader(BaseDatasetLoader):
    """Dataset loader retrieving historical
    time-series and tabular features from Feast."""

    SUPPORTED_TYPE: list[str] = ["timeseries", "tabular"]

    def load(
        self,
        cfg: DictConfig,
        path: str,
        *,
        index_col: Optional[str] = None,
        data_type: str,
    ) -> pd.DataFrame:
        """
        Load historical features from Feast using unified facade.

        Args:
            cfg: DictConfig with feature metadata.
            path: Feast URI (feast://<repo>?entity=<key>&id=<val>).
            index_col: Optional index column name.
            data_type: 'timeseries' or 'tabular'.

        Returns:
            DataFrame with requested features.
        """
        self._validate_data_type(data_type)

        # Use facade for clean feature loading
        facade = FeatureStoreFacade(cfg)
        df = facade.load_features()

        logger.info("Loaded %d rows from Feast via facade.", len(df))
        return df

    def _validate_data_type(self, data_type: str) -> None:
        """Validate data_type input."""
        if data_type not in self.SUPPORTED_TYPE:
            raise ValueError(
                f"FeastDatasetLoader only supports types: {self.SUPPORTED_TYPE}"
            )
