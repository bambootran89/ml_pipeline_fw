"""
RecSys feature registration for Feast-compatible feature stores.

This module defines the recommendation system schema and registers:
1. The 'recsys_user' entity for feature joins.
2. The 'recsys_view' feature view backed by an offline source.

Feast type conversion is delegated to the BaseFeatureStore implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

RECSYS_SCHEMA: Dict[str, str] = {
    "view_count": "float",
    "like_ratio": "float",
    "item_id": "int",
}


def register_recsys_features(
    store: Any,
    entity_col: str = "user_id",
    source_path: str = "",
) -> None:
    """
    Register recommendation system features into a Feast-compatible store.

    Args:
        store: Feature store client implementing BaseFeatureStore interface.
        entity_col: Join key column for the 'recsys_user' entity (default: 'user_id').
        source_path: Offline source path to back the feature view.

    Raises:
        ValueError: If entity_col is empty or source_path is missing.
    """
    if not entity_col:
        raise ValueError("entity_col must be a non-empty string.")

    if not source_path:
        raise ValueError("source_path must be provided to register the feature view.")

    store.register_entity(
        name="recsys_user",
        join_key=entity_col,
        value_type="int",
    )

    store.register_feature_view(
        name="recsys_view",
        entities=["recsys_user"],
        schema=RECSYS_SCHEMA,
        source_path=source_path,
        ttl_days=2,
    )

    logger.info(
        "Registered RecSys features with entity_col='%s' from source='%s'",
        entity_col,
        source_path,
    )
