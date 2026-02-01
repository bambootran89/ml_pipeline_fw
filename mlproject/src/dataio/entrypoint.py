from __future__ import annotations

from typing import Optional

import pandas as pd
from omegaconf import DictConfig

from mlproject.src.dataio.registry import DatasetLoaderRegistry


def load_dataset(
    cfg: DictConfig,
    path: str,
    *,
    index_col: Optional[str] = None,
    data_type: str,
) -> pd.DataFrame:
    """
    Load a dataset using the appropriate DatasetLoader.

    This function serves as the public entry point for
    pipelines and data modules, abstracting away loader
    resolution and instantiation logic.

    Parameters
    ----------
    path : str
        Data source path.
    index_col : Optional[str], default=None
        Column to be used as index.
    data_type : str
        Dataset type (e.g., "timeseries", "tabular").

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    loader = DatasetLoaderRegistry.get_loader(path)

    return loader.load(
        cfg,
        path,
        index_col=index_col,
        data_type=data_type,
    )
