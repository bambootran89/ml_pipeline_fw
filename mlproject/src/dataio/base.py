from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from omegaconf import DictConfig


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.

    A DatasetLoader is responsible for:
    - Loading raw data from a data source (file, database, etc.)
    - Returning the data as a pandas DataFrame

    It must NOT:
    - Perform feature engineering
    - Perform train/validation/test splitting
    - Contain business or model-specific logic
    """

    @abstractmethod
    def load(
        self,
        cfg: DictConfig,
        path: str,
        *,
        index_col: Optional[str] = None,
        data_type: str,
    ) -> pd.DataFrame:
        """
        Load a dataset from the given source.

        Parameters
        ----------
        path : str
            Path to the data source or connection string.
        index_col : Optional[str], default=None
            Column name to be used as DataFrame index.
        data_type : str
            Dataset type (e.g., "timeseries", "tabular").

        Returns
        -------
        pd.DataFrame
            Loaded raw dataset.
        """
        raise NotImplementedError
