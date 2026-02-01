"""
Fraud feature registration for Feast-compatible feature stores.

This module defines the fraud feature schema and registers:
1. The 'user' entity used for joins.
2. The 'fraud_view' feature view backed by an offline source.

Type conversions to Feast internal types are handled by the
downstream BaseFeatureStore implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

FRAUD_SCHEMA: Dict[str, str] = {
    "amount": "float",
    "tx_count_30m": "float",
    "risk_score": "float",
}


def register_fraud_features(
    store: Any,
    entity_col: str = "user_id",
    source_path: str = "",
) -> None:
    """
    Register fraud detection features into a Feast-compatible feature store.

    Args:
        store: Feature store client implementing BaseFeatureStore interface.
        entity_col: Join key column name for the 'user' entity (default: 'user_id').
        source_path: Offline source path used to back the feature view.

    Raises:
        ValueError: If entity_col is empty or source_path is not provided.
    """
    if not entity_col:
        raise ValueError("entity_col must be a non-empty string.")

    if not source_path:
        raise ValueError("source_path must be provided to register the feature view.")

    store.register_entity(
        name="user",
        join_key=entity_col,
        description="User identifier for fraud detection",
        value_type="int",
    )

    store.register_feature_view(
        name="fraud_view",
        entities=["user"],
        schema=FRAUD_SCHEMA,
        source_path=source_path,
        ttl_days=30,
    )

    logger.info(
        "Registered fraud features using entity_col='%s' and source_path='%s'",
        entity_col,
        source_path,
    )
