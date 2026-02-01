"""
Interaction feature transformation pipeline for recommendation and analytics.
"""

from __future__ import annotations

import pandas as pd


def process_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform and sanitize interaction features for modeling.

    Ensures numerical stability and enforces valid ranges where required.
    Example operations may include normalization, ratio computation,
    and score bounding. The function returns a new DataFrame copy
    with updated values.

    Args:
        df: Input interaction feature table.

    Returns:
        A transformed DataFrame with bounded and sanitized columns.
    """
    df = df.copy()

    if "like_ratio" in df.columns:
        df["like_ratio"] = df["like_ratio"].clip(0.0, 1.0)

    return df
