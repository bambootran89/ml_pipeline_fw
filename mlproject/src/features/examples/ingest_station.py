"""
Production station ingestion pipeline.

This script simulates raw station production data, applies transformations,
persists features to an offline Parquet store, and registers metadata to Feast
for subsequent materialization and online inference API usage.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from mlproject.src.features.definitions.station_features import (
    register_station_definitions,
)
from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager
from mlproject.src.features.transformers.station_transformers import (
    transform_station_data,
)

logger = logging.getLogger(__name__)


def simulate_station_raw_data(
    periods: int = 200,
    station_id: Union[int, str] = 101,
) -> pd.DataFrame:
    """
    Simulate raw hourly production data for a single station.

    Args:
        periods: Number of hourly records to generate.
        station_id: Identifier for the production station.

    Returns:
        DataFrame containing simulated raw station production data.
    """
    now = datetime.now(timezone.utc)
    df = pd.DataFrame(
        {
            "station_id": [station_id] * periods,
            "event_timestamp": pd.date_range(
                end=now,
                periods=periods,
                freq="1H",
                tz=timezone.utc,
            ),
            "production": np.random.uniform(50, 100, size=periods).astype(np.float32),
        }
    )
    return df


def run_pipeline(
    *,
    repo_name: str = "ts_gen_repo",
    index_col: Optional[str] = None,
    materialize_days: Optional[int] = 7,
) -> Optional[Path]:
    """
    Run the station feature ingestion and registration pipeline.

    This pipeline performs:
      1. Raw data simulation.
      2. Feature transformation.
      3. Offline Parquet persistence.
      4. Feast metadata registration.
      5. Optional materialization to online store.

    Args:
        repo_name: Name or path of the Feast feature repository.
        index_col: Optional column name to set as DataFrame index.
        materialize_days: If provided, materialize recent N days to online store.

    Returns:
        Path to the offline Parquet file, or None if pipeline fails.
    """
    try:
        FeastRepositoryManager.initialize_repo(repo_name)

        df_raw = simulate_station_raw_data()
        df_features = transform_station_data(df_raw)

        if index_col and index_col in df_features.columns:
            df_features = df_features.set_index(index_col)

        df_features = df_features.sort_values("event_timestamp")

        for col in df_features.columns:
            if df_features[col].isna().sum() > 0:
                df_features[col] = df_features[col].ffill()

        data_path = Path(repo_name) / "data" / "production.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_parquet(data_path)

        store = FeatureStoreFactory.create(
            store_type="feast",
            repo_path=repo_name,
        )

        register_station_definitions(store, "station_id", str(data_path.absolute()))

        if materialize_days:
            start = datetime.now(timezone.utc) - timedelta(days=materialize_days)
            end = datetime.now(timezone.utc)
            logger.info(
                "Materializing last %d days: %s -> %s", materialize_days, start, end
            )
            store.materialize(start, end)

        logger.info("Station ingestion pipeline completed: %s", data_path)
        print(f"Station ingestion pipeline finished: {data_path}")

        return data_path

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        print("Pipeline failed. Check logs.")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = run_pipeline()
    if not path:
        sys.exit(1)
