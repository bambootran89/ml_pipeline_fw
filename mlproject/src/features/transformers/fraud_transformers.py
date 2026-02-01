"""
Fraud feature engineering utilities.

This module computes fraud-related indicators from raw transaction data.
It preserves input data integrity by working on a defensive copy and
returns a normalized, inference-ready DataFrame.
"""

from __future__ import annotations

import pandas as pd


def add_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fraud detection features from raw transaction data.

    Transformations applied:
    1. high_amount_flag: Float indicator for large transactions (> 1000).
    2. risk_score: Weighted risk formula using device and transaction velocity.

    Args:
        df: Raw transaction DataFrame containing:
            - amount (numeric)
            - device_risk (numeric)
            - tx_count_30m (numeric)
            - high_amount_flag (derived internally)

    Returns:
        A new DataFrame with fraud features added and values normalized.

    Raises:
        KeyError: If required input columns are missing.
    """
    required_cols = ["amount", "device_risk", "tx_count_30m"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns for fraud features: {missing}")

    df = df.copy()

    df["high_amount_flag"] = (df["amount"] > 1000).astype(float)

    df["risk_score"] = (
        0.6 * df["device_risk"]
        + 0.4 * (df["tx_count_30m"] / 10)
        + 0.2 * df["high_amount_flag"]
    ).clip(0, 1)

    return df
