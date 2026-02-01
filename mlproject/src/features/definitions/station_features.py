"""
Feast registration definitions for production stations.
"""

from __future__ import annotations

from typing import Any

STATION_SCHEMA = {
    "production": "float",
}


def register_station_definitions(store: Any, entity_col: str, source_path: str) -> None:
    """
    Register station entity and its associated feature view metadata to Feast.

    This function defines the station join key and registers a lightweight
    feature view (`station_stats`) for periodic materialization. It is designed
    to support batch ingestion and online feature retrieval for inference APIs.

    Args:
        store: Initialized Feast-compatible feature store client.
        entity_col: Column name used as the join key for station entity.
        source_path: Absolute or local path to offline feature source.
    """
    store.register_entity(
        name="station",
        join_key=entity_col,
        description="Production station identifier",
        value_type="int",
    )

    store.register_feature_view(
        name="station_stats",
        entities=["station"],
        schema=STATION_SCHEMA,
        source_path=source_path,
        ttl_days=1,
    )
