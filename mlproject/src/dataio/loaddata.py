from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from mlproject.src.features.facade import FeatureStoreFacade


def load_from_feast(cfg: DictConfig, time_point: str) -> pd.DataFrame:
    """
    Load features from Feast using unified facade.

    Returns
    -------
    pd.DataFrame
        Features ready for preprocessing.
        - Tabular: Single row from Online Store
        - Timeseries: Indexed sequence window

    Raises
    ------
    ValueError
        If Feast URI is invalid or data loading fails.
    """
    facade = FeatureStoreFacade(cfg, mode="online")
    return facade.load_features(time_point=time_point)


def load_csv_data(input_path: str) -> pd.DataFrame:
    """Load data from CSV file.

    Parameters
    ----------
    input_path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If file not found.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[SERVE] Loading data from CSV: {input_path}")
    df = pd.read_csv(input_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    print(f"[SERVE] Loaded CSV data shape: {df.shape}")
    return df
