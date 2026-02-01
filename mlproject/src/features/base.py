"""
Vendor-agnnostic abstract base definitions for Feature Store operations.

This module defines the minimal, typed contracts required for modern
feature store clients that support entity registration, schema-aware
feature view creation, offline historical feature retrieval with
point-in-time (PIT) semantics, batch materialization, and online
feature lookups for inference workloads.

The abstraction aligns with Feast v0.58.0+ operational semantics,
including timezone-aware PIT keys, schema awareness, and explicit
offline-to-online synchronization expectations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class BaseFeatureStore(ABC):
    """
    Abstract interface for a feature store client exposing both offline and online
    feature access patterns with schema-validated registration and deterministic
    batch materialization.

    Required capabilities defined by this interface:
    - Entity registration for primary join keys.
    - Feature view registration with logical schemas and optional TTL.
    - Offline historical feature retrieval honoring PIT (point-in-time) semantics.
    - Online feature retrieval returning latest materialized values for inference.
    - Batch materialization to synchronize offline sources into an online store.

    Implementations must enforce timezone-aware PIT keys (e.g., UTC) and ensure
    schema correctness to support reliable joins and lookups in Feast v0.58.0+.
    """

    @abstractmethod
    def register_entity(
        self,
        name: str,
        join_key: str,
        description: str = "",
        value_type: str = "int",
    ) -> None:
        """
        Register an entity representing an indexable object used as a join key.

        Entities serve as the primary index for resolving feature values in both
        offline PIT joins and online inference lookups.

        Args:
            name:
                Unique logical name of the entity.
            join_key:
                Column name used for offline/online feature joins and lookups.
            description:
                Optional human-readable description of the entity.
            value_type:
                Logical type of the join key (default: "int").
        """

    @abstractmethod
    def register_feature_view(
        self,
        name: str,
        entities: List[str],
        schema: Dict[str, str],
        source_path: str,
        ttl_days: Optional[int] = None,
    ) -> None:
        """
        Register a schema-aware feature view backed by an offline data source.

        Feature views define typed feature mappings that can be resolved at PIT
        during training or served online after batch materialization.

        Args:
            name:
                Unique name of the feature view.
            entities:
                List of entity names this view is joined on.
            schema:
                Mapping of feature names to logical types.
            source_path:
                URI or local path of the offline data source backing the view.
            ttl_days:
                Optional time-to-live (in days) for online staleness control.
        """

    @abstractmethod
    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
    ) -> pd.DataFrame:
        """
        Retrieve offline features resolved using PIT (point-in-time) join semantics.

        This method is intended for training workloads and must guarantee that
        feature values are resolved without future-information leakage relative
        to ``event_timestamp``.

        Args:
            entity_df:
                DataFrame containing entity join keys and PIT ``event_timestamp``.
            features:
                Fully qualified feature references to resolve at PIT.

        Returns:
            DataFrame of PIT-aligned feature values for training workloads.
        """

    @abstractmethod
    def get_online_features(
        self,
        entity_rows: List[Dict[str, Any]],
        features: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent online feature values for inference workloads.

        Online lookups must return the latest successfully materialized value
        per entity join key and honor TTL if configured by the implementation.

        Args:
            entity_rows:
                List of dictionaries containing entity join key values.
            features:
                Fully qualified feature references to fetch for inference.

        Returns:
            List of resolved feature dictionaries for inference workloads.
        """

    @abstractmethod
    def materialize(
        self,
        start_date: Any,
        end_date: Any,
    ) -> None:
        """
        Materialize offline features into an online store for inference.

        The materialization window must include all valid feature source records
        from ``start_date`` (inclusive) up to ``end_date`` (inclusive). Implementations
        should enforce timezone-aware PIT keys when applicable (e.g., UTC for Feast).

        Args:
            start_date:
                Beginning of the materialization window (inclusive).
            end_date:
                End of the materialization window (inclusive).
        """
