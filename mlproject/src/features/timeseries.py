"""
Specialized Feature Store wrapper for Time-Series Sequence operations.

This module provides a high-level API to retrieve sequences of data
for single or multiple entities. It abstracts away entity ID handling
while supporting efficient bulk queries.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union

import pandas as pd

from mlproject.src.features.base import BaseFeatureStore


class TimeSeriesFeatureStore:
    """
    Wrapper for BaseFeatureStore with multi-entity sequence retrieval.

    Supports both single-entity (backward compatible) and multi-entity
    queries with optimal performance through bulk operations.
    """

    TIMESTAMP_FIELD = "event_timestamp"

    def __init__(
        self,
        store: BaseFeatureStore,
        default_entity_key: str = "location_id",
        default_entity_id: Optional[Union[int, str]] = None,
        default_entity_ids: Optional[Union[int, str, List[Union[int, str]]]] = None,
    ):
        """
        Initialize wrapper with store and entity metadata.

        Parameters
        ----------
        store : BaseFeatureStore
            Feature store implementation instance.
        default_entity_key : str, default="location_id"
            Entity join key name.
        default_entity_id : Optional[Union[int, str]], default=None
            Single entity ID (deprecated, use default_entity_ids).
        default_entity_ids : Optional[Union[int, str, List[...]]
            Entity ID(s) to query. Can be:
            - Single value: 1 or "location_A"
            - List: [1, 2, 3] or ["A", "B", "C"]
            If None, uses default_entity_id or [1].

        Examples
        --------
        Single entity (backward compatible):
        >>> store = TimeSeriesFeatureStore(
        ...     feast_store,
        ...     default_entity_id=1
        ... )

        Multiple entities (recommended):
        >>> store = TimeSeriesFeatureStore(
        ...     feast_store,
        ...     default_entity_ids=[1, 2, 3, 4, 5]
        ... )
        """
        self.store = store
        self.default_entity_key = default_entity_key

        # Handle entity IDs with backward compatibility
        if default_entity_ids is not None:
            # New parameter provided
            if isinstance(default_entity_ids, (int, str)):
                self.default_entity_ids = [default_entity_ids]
            else:
                self.default_entity_ids = list(default_entity_ids)
        elif default_entity_id is not None:
            # Deprecated parameter provided
            self.default_entity_ids = [default_entity_id]
        else:
            # No parameters provided
            self.default_entity_ids = [1]

    def get_sequence_by_range(
        self,
        features: List[str],
        start_date: datetime,
        end_date: datetime,
        frequency: str = "1H",
        entity_ids: Optional[List[Union[int, str]]] = None,
        entity_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieve continuous sequences for time range (multi-entity).

        Parameters
        ----------
        features : List[str]
            Feature names in 'view:feature' format.
        start_date : datetime
            Start of time range (inclusive).
        end_date : datetime
            End of time range (inclusive).
        frequency : str, default="1H"
            Pandas frequency string (e.g., "1H", "30min").
        entity_ids : Optional[List[Union[int, str]]], default=None
            Entity IDs to query. If None, uses default_entity_ids.
        entity_key : Optional[str], default=None
            Entity key name. If None, uses default_entity_key.

        Returns
        -------
        pd.DataFrame
            Features sorted by [entity_key, event_timestamp].
            For N entities and T timestamps, returns N×T rows.

        Examples
        --------
        Query multiple locations:
        >>> df = store.get_sequence_by_range(
        ...     features=["temp:value", "humidity:value"],
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 1, 7),
        ...     entity_ids=[1, 2, 3],
        ... )
        >>> len(df)  # 3 entities × 168 hours
        504
        """
        target_ids = entity_ids if entity_ids is not None else self.default_entity_ids
        target_key = entity_key if entity_key is not None else self.default_entity_key

        # Generate timestamps
        timestamps = pd.date_range(
            start=start_date,
            end=end_date,
            freq=frequency,
            tz=timezone.utc,
        )

        # Build Cartesian product: entities × timestamps
        entity_rows = []
        for entity_id in target_ids:
            for ts in timestamps:
                entity_rows.append(
                    {
                        target_key: entity_id,
                        self.TIMESTAMP_FIELD: ts,
                    }
                )

        # Single bulk query
        result = self.store.get_historical_features(
            entity_df=pd.DataFrame(entity_rows),
            features=features,
        )

        return result.sort_values([target_key, self.TIMESTAMP_FIELD])

    def get_latest_n_sequence(
        self,
        features: List[str],
        n_points: int,
        frequency_hours: int = 1,
        entity_ids: Optional[List[Union[int, str]]] = None,
        entity_key: Optional[str] = None,
        time_point: str = "now",
    ) -> pd.DataFrame:
        """
        Retrieve N most recent points for entities (multi-entity).

        Parameters
        ----------
        features : List[str]
            Feature names in 'view:feature' format.
        n_points : int
            Number of historical points per entity.
        frequency_hours : int, default=1
            Hourly interval between points.
        entity_ids : Optional[List[Union[int, str]]], default=None
            Entity IDs to query. If None, uses default_entity_ids.
        entity_key : Optional[str], default=None
            Entity key name. If None, uses default_entity_key.
        time_point : str, default="now"
            Reference time. Can be:
            - "now": Current time (rounded to hour)
            - ISO datetime string: "2024-01-01T12:00:00"

        Returns
        -------
        pd.DataFrame
            Features sorted by [entity_key, event_timestamp].
            For N entities, returns N×n_points rows.

        Examples
        --------
        Single entity (backward compatible):
        >>> df = store.get_latest_n_sequence(
        ...     features=["temp:value"],
        ...     n_points=24,
        ... )
        >>> len(df)
        24

        Multiple entities (5x faster than 5 separate queries):
        >>> df = store.get_latest_n_sequence(
        ...     features=["temp:value"],
        ...     n_points=24,
        ...     entity_ids=[1, 2, 3, 4, 5],
        ... )
        >>> len(df)
        120
        """
        target_ids = entity_ids if entity_ids is not None else self.default_entity_ids
        target_key = entity_key if entity_key is not None else self.default_entity_key

        # Parse time_point
        if time_point == "now":
            timestamps = [
                datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
                - timedelta(hours=i * frequency_hours)
                for i in range(n_points)
            ]
        else:
            timestamps = [
                datetime.fromisoformat(time_point).astimezone(timezone.utc)
                - timedelta(hours=i * frequency_hours)
                for i in range(n_points)
            ]

        # Generate timestamps (descending from base_time)

        # Build Cartesian product: entities × timestamps

        entity_rows = []
        for entity_id in target_ids:
            for ts in timestamps:
                entity_rows.append(
                    {
                        target_key: entity_id,
                        self.TIMESTAMP_FIELD: ts,
                    }
                )
        # Single bulk query for all entities
        entity_df = pd.DataFrame(entity_rows)
        result = self.store.get_historical_features(
            entity_df=entity_df,
            features=features,
        )

        return result.sort_values([target_key, self.TIMESTAMP_FIELD])

    def get_latest_n_sequence_single(
        self,
        features: List[str],
        n_points: int,
        frequency_hours: int = 1,
        entity_id: Optional[Union[int, str]] = None,
        entity_key: Optional[str] = None,
        time_point: str = "now",
    ) -> pd.DataFrame:
        """
        Convenience method for single-entity queries.

        This is syntactic sugar for get_latest_n_sequence with
        entity_ids=[entity_id].

        Parameters
        ----------
        entity_id : Optional[Union[int, str]], default=None
            Single entity ID to query.
        (other parameters same as get_latest_n_sequence)

        Returns
        -------
        pd.DataFrame
            Features for single entity (n_points rows).
        """
        target_id = entity_id if entity_id is not None else self.default_entity_ids[0]

        return self.get_latest_n_sequence(
            features=features,
            n_points=n_points,
            frequency_hours=frequency_hours,
            entity_ids=[target_id],
            entity_key=entity_key,
            time_point=time_point,
        )
