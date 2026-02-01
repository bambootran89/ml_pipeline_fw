"""Feature engineering helpers for Feast preparation."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """Add lag features for selected time-series columns.

    The function preserves the original column values and appends new
    lag-shifted columns in the format: ``<column>_lag<period>``.

    Args:
        df:
            Input DataFrame containing time-series data.
        columns:
            List of column names to generate lag features for.
        lags:
            List of integer lag periods (e.g., ``[1, 24, 168]``).

    Returns:
        DataFrame with added lag feature columns.
    """
    df_out = df.copy()

    for col in columns:
        if col not in df_out.columns:
            raise KeyError(f"Column '{col}' not found in dataframe")

        for lag in lags:
            df_out[f"{col}_lag{lag}"] = df_out[col].shift(lag)
    assert len(df) == len(df_out)
    return df_out


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    agg: str = "mean",
) -> pd.DataFrame:
    """Add rolling window aggregation features for selected columns.

    New columns are appended in the format:
    ``<column>_roll<window>_<aggregation>``.

    Args:
        df:
            Input DataFrame containing numerical time-series data.
        columns:
            List of column names to compute rolling aggregations on.
        windows:
            List of rolling window sizes as integers
            (e.g., ``[6, 12, 24]``).
        agg:
            Aggregation function name supported by Pandas rolling
            (e.g., ``"mean"``, ``"std"``, ``"min"``, ``"max"``).

    Returns:
        DataFrame with rolling feature columns added.
    """
    df_out = df.copy()

    if agg not in {"mean", "std", "min", "max"}:
        raise ValueError(f"Aggregation '{agg}' is not supported")

    for col in columns:
        if col not in df_out.columns:
            raise KeyError(f"Column '{col}' not found in dataframe")

        for window in windows:
            df_out[f"{col}_roll{window}_{agg}"] = df_out[col].rolling(window).agg(agg)
    assert len(df) == len(df_out)
    return df_out


def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Add cyclical time-based features for forecasting models.

    Generated features:
    - ``hour_sin``, ``hour_cos`` → cyclical hour of day encoding
    - ``dow_sin``, ``dow_cos`` → cyclical day of week encoding

    Args:
        df:
            Input DataFrame containing a timestamp column.
        timestamp_col:
            Name of the column containing datetime values.

    Returns:
        DataFrame with added cyclical time features.
    """
    df_out = df.copy()

    if timestamp_col not in df_out.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found")

    ts = pd.to_datetime(df_out[timestamp_col])

    # Hour of day (24-hour cycle)
    df_out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df_out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)

    # Day of week (7-day cycle)
    df_out["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df_out["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    assert len(df) == len(df_out)
    return df_out
