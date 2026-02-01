"""
Batch feature engineering and Feast registration pipeline for recommendation
interaction data. Generates synthetic interaction logs, processes features,
persists them to an offline store, and registers metadata definitions to Feast.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlproject.src.features.definitions.recsys_features import register_recsys_features
from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager
from mlproject.src.features.transformers.recsys_transformers import (
    process_interaction_features,
)

logger = logging.getLogger(__name__)


def generate_interaction_data(hours: int = 72) -> pd.DataFrame:
    """
    Generate synthetic hourly user-item interaction logs.

    Args:
        hours: Number of hours of interaction history to simulate.

    Returns:
        A DataFrame containing synthetic interaction feature columns.
    """
    now: datetime = datetime.now(timezone.utc)

    ts: pd.DatetimeIndex = pd.date_range(
        start=now - timedelta(hours=hours),
        periods=hours,
        freq="H",
        tz=timezone.utc,
    )

    n_rows: int = len(ts)

    return pd.DataFrame(
        {
            "user_id": (np.arange(n_rows) % 4 + 1).astype(int),
            "item_id": (np.arange(n_rows) % 10 + 101).astype(int),
            "event_timestamp": ts,
            "view_count": np.random.poisson(5, n_rows).astype(float),
            "like_ratio": np.random.beta(2, 5, n_rows).astype(float),
        }
    )


def run_pipeline() -> None:
    """
    Execute batch feature processing, offline persistence, and Feast metadata
    registration for recommendation system interaction features.
    """
    repo_name: str = "recsys_repo"
    FeastRepositoryManager.initialize_repo(repo_name)

    df_raw: pd.DataFrame = generate_interaction_data()
    df_features: pd.DataFrame = process_interaction_features(df_raw)

    data_path: Path = Path(repo_name) / "data" / "interactions.parquet"
    data_path.parent.mkdir(exist_ok=True)

    df_features.to_parquet(data_path)
    logger.info("Saved offline interaction features to %s", data_path)

    store: Any = FeatureStoreFactory.create(store_type="feast", repo_path=repo_name)
    register_recsys_features(store, "user_id", str(data_path.absolute()))

    print(f"RecSys ingestion completed. Offline store path: {data_path}")


if __name__ == "__main__":
    run_pipeline()
