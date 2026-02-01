"""
Feast registration metadata for time-series forecasting feature views.
"""

from __future__ import annotations

from typing import Any, Mapping

FORECAST_SCHEMA: Mapping[str, str] = {
    "temperature": "float",
    "demand": "float",
    "temperature_lag24": "float",
    "demand_roll12_mean": "float",
}


def register_forecast_definitions(
    store: Any, entity_col: str, source_path: str
) -> None:
    """
    Register entity and feature view metadata for time-series forecasting.

    The registered components include:
    - Entity: `location` using the provided `entity_col` as join key
    - Feature view: `forecast_view` with lag and rolling window schema

    Args:
        store: Initialized Feast feature store instance.
        entity_col: Column name used as the entity join key.
        source_path: Path to the processed Parquet feature source.
    """
    store.register_entity(
        name="location",
        join_key=entity_col,
        description="Sensor location entity",
        value_type="int",
    )

    store.register_feature_view(
        name="forecast_view",
        entities=["location"],
        schema=FORECAST_SCHEMA,
        source_path=source_path,
        ttl_days=7,
    )
