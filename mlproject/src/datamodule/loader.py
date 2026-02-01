from __future__ import annotations

from typing import Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.dataio.entrypoint import load_dataset


def resolve_datasets_from_cfg(
    cfg: DictConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Resolve and load raw datasets from configuration.

    This function handles:
    - Single-path datasets (no split yet)
    - Explicit train/val/test datasets

    It performs orchestration only and delegates actual I/O
    to the dataio layer.
    """
    cfg = cfg if cfg is not None else OmegaConf.create()
    data_cfg = cfg.get("data", OmegaConf.create())

    path = data_cfg.get("path")
    train_path = data_cfg.get("train_path")
    val_path = data_cfg.get("val_path")
    test_path = data_cfg.get("test_path")

    data_type = str(data_cfg.get("type", "timeseries")).lower()
    index_col = data_cfg.get("index_col")

    if data_type == "timeseries" and not index_col:
        index_col = "date"
    elif data_type == "tabular":
        index_col = None

    if path:
        full_df = load_dataset(
            cfg,
            path,
            index_col=index_col,
            data_type=data_type,
        )
        return full_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if not train_path:
        raise ValueError("`data.train_path` must be specified")

    if not test_path:
        raise ValueError("`data.test_path` must be specified")

    train_df = load_dataset(
        cfg,
        train_path,
        index_col=index_col,
        data_type=data_type,
    )

    test_df = load_dataset(
        cfg,
        test_path,
        index_col=index_col,
        data_type=data_type,
    )

    val_df = (
        load_dataset(cfg, val_path, index_col=index_col, data_type=data_type)
        if val_path
        else test_df.copy()
    )

    return pd.DataFrame(), train_df, val_df, test_df
