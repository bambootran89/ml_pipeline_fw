"""
Feast feature registration module for ETTH1 time-series dataset.

This module defines the ETTH1 feature schema and provides a helper function
to register entities and feature views into a Feast-compatible feature store.
"""

from __future__ import annotations

from typing import Any, Dict

ETTH1_SCHEMA: Dict[str, str] = {
    "HUFL": "float",
    "MUFL": "float",
    "HUFL_lag24": "float",
    "MUFL_lag24": "float",
    "HUFL_roll12_mean": "float",
    "MUFL_roll12_mean": "float",
    "hour_sin": "float",
    "hour_cos": "float",
    "dow_sin": "float",
    "dow_cos": "float",
    "mobility_inflow": "float",
}


def register_etth1_features(store: Any, entity_col: str, source_path: str) -> None:
    """
    Register ETTH1 entities and feature view metadata into a Feast-compatible store.

    This function centralizes metadata registration logic. When new features are
    added, only ETTH1_SCHEMA or this function needs to be updated.

    Args:
        store: Feast-compatible feature store client or wrapper.
        entity_col: Column name used as the entity join key.
        source_path: Path to the offline source (CSV or Parquet) used by Feast.

    Raises:
        ValueError: If required arguments are empty.
        AttributeError: If the store does not support required registration APIs.
    """
    if not entity_col or not source_path:
        raise ValueError("entity_col and source_path must be non-empty strings.")

    # Register entity metadata
    store.register_entity(name="location", join_key=entity_col, value_type="int")

    # Register feature view metadata
    store.register_feature_view(
        name="etth1_features",
        entities=["location"],
        schema=ETTH1_SCHEMA,
        source_path=source_path,
        ttl_days=365,
    )
