"""
Forecast feature engineering utilities aligned with ts_transformers pipeline.
"""

from __future__ import annotations

import pandas as pd

from mlproject.src.features.transformers.ts_transformers import (
    add_lag_features,
    add_rolling_features,
)


def apply_forecast_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply forecast feature engineering transformations.

    The transformations include:
    - 24-hour lag for `temperature` → generates `temperature_lag24`
    - 12-hour rolling mean for `demand` → generates `demand_roll12_mean`

    Args:
        df: Input DataFrame containing at least `temperature` and `demand`.

    Returns:
        DataFrame copy with engineered forecast features.
    """
    df = df.copy()

    df = add_lag_features(df, columns=["temperature"], lags=[24])
    df = add_rolling_features(df, columns=["demand"], windows=[12], agg="mean")

    return df
