"""
Station data transformation utilities for memory optimization and cleaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def transform_station_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transformations to station-level data for modeling and memory efficiency.

    This function operates on a DataFrame copy and enforces optimized numeric
    types where applicable. Example use cases include cleaning, downcasting, or
    converting high-cardinality numeric columns to memory-efficient formats such
    as float32.

    Args:
        df: Raw input station feature table.

    Returns:
        A transformed DataFrame with optimized dtypes.
    """
    df = df.copy()

    if "production" in df.columns:
        df["production"] = df["production"].astype(np.float32)

    return df
