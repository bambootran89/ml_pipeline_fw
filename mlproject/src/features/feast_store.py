"""
Feast-backed implementation of a typed Feature Store client for Feast v0.58.0+.

This module provides a concrete ``FeastFeatureStore`` class implementing the
``BaseFeatureStore`` contract. It normalizes type inconsistencies between
``ValueType`` (for ``Entity``) and ``feast.types`` (for ``Field`` schemas), while
ensuring PIT-safe offline joins and deterministic pivoted online responses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

from mlproject.src.features.base import BaseFeatureStore

logger = logging.getLogger(__name__)


class FeastFeatureStore(BaseFeatureStore):
    """Feast v0.58.0+ implementation with hybrid type normalization."""

    def __init__(self, repo_path: str = "feature_repo") -> None:
        """Initialize a Feast feature store client."""
        self.repo_path = repo_path
        self.store = FeatureStore(repo_path=repo_path)

    def register_entity(
        self,
        name: str,
        join_key: str,
        description: str = "",
        value_type: str = "int",
    ) -> None:
        """Register a Feast ``Entity`` using ``ValueType`` enum."""
        v_type = ValueType.INT64 if value_type == "int" else ValueType.STRING

        entity = Entity(
            name=name,
            join_keys=[join_key],
            description=description,
            value_type=v_type,
        )
        self.store.apply([entity])
        logger.info("Registered entity: %s", name)

    def register_feature_view(
        self,
        name: str,
        entities: List[str],
        schema: Dict[str, str],
        source_path: str,
        ttl_days: Optional[int] = None,
    ) -> None:
        """Register a ``FeatureView`` backed by a ``FileSource``."""
        source = FileSource(path=source_path, timestamp_field="event_timestamp")

        type_map = {"float": Float32, "int": Int64, "string": String}
        feast_fields = [
            Field(name=fn, dtype=type_map.get(ft, Float32)) for fn, ft in schema.items()
        ]

        entity_objs = [self.store.get_entity(e_name) for e_name in entities]

        view = FeatureView(
            name=name,
            entities=entity_objs,
            schema=feast_fields,
            source=source,
            ttl=pd.Timedelta(days=ttl_days) if ttl_days else None,
        )
        self.store.apply([view])
        logger.info("Registered feature view: %s", name)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
    ) -> pd.DataFrame:
        """Retrieve PIT-safe historical features for training."""
        return self.store.get_historical_features(
            entity_df=entity_df,
            features=features,
        ).to_df()

    def get_online_features(
        self,
        entity_rows: List[Dict[str, Any]],
        features: List[str],
    ) -> List[Dict[str, Any]]:
        """Retrieve online features and pivot into record dictionaries."""
        response = self.store.get_online_features(
            entity_rows=entity_rows,
            features=features,
        )
        data = response.to_dict()

        keys = list(data.keys())
        values = list(data.values())
        return [dict(zip(keys, row)) for row in zip(*values)]

    def materialize(self, start_date: Any, end_date: Any) -> None:
        """Materialize offline features into the online store."""
        self.store.materialize(start_date, end_date)
        logger.info(
            "Materialized features from %s to %s", str(start_date), str(end_date)
        )
