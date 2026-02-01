"""
Feast time-series retrieval demo for training and inference.

This script demonstrates offline sequence retrieval by time range and
latest-N point lookup using a TimeSeriesFeatureStore wrapper around a
Feast-compatible feature store client.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from mlproject.src.features.examples.ingest_station import run_pipeline
from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.timeseries import TimeSeriesFeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _profile_df(df: pd.DataFrame, tag: str) -> None:
    """
    Print minimal profiling information for a DataFrame.

    Args:
        df: DataFrame to profile.
        tag: Label tag describing the profiling stage.
    """
    null_counts = df.isna().sum()
    logger.info(
        "[Profile-%s] rows=%d cols=%d nulls=%d earliest=%s latest=%s",
        tag,
        df.shape[0],
        df.shape[1],
        int(null_counts.sum()),
        df.get("event_timestamp", pd.Series([None])).min(),
        df.get("event_timestamp", pd.Series([None])).max(),
    )
    if null_counts.sum() > 0:
        logger.info("[Null-%s] %s", tag, null_counts[null_counts > 0].to_dict())


def run_retrieval_demo() -> None:
    """
    Run Feast retrieval demo for time-range sequence and latest-N points.
    """
    run_pipeline()
    base_store = FeatureStoreFactory.create(
        store_type="feast",
        repo_path="ts_gen_repo",
    )

    ts_store = TimeSeriesFeatureStore(
        base_store,
        default_entity_key="station_id",
        default_entity_id=101,
    )

    logger.info("Fetching sequence range (last 10 hours)...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=10)

    df_range = ts_store.get_sequence_by_range(
        features=["station_stats:production"],
        start_date=start,
        end_date=end,
    )
    print("\n--- Training Sequence Result ---")
    print(df_range.head())

    _profile_df(df_range, "range")

    logger.info("Fetching latest 5 points for inference...")
    df_latest = ts_store.get_latest_n_sequence(
        features=["station_stats:production"],
        n_points=5,
    )
    print("\n--- Latest N Points Result ---")
    print(df_latest)

    _profile_df(df_latest, "latest")


if __name__ == "__main__":
    run_retrieval_demo()
