"""
Batch ingestion pipeline for fraud feature engineering and Feast metadata registration.

This script simulates raw transaction data, applies fraud transformations, persists
the results to a Parquet file (offline store), registers entity and feature view
definitions to Feast, and ensures static quality compliance with pylint and mypy.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mlproject.src.features.definitions.fraud_reatures import register_fraud_features
from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager
from mlproject.src.features.transformers.fraud_transformers import add_fraud_features


def simulate_raw_data(n: int = 5000) -> pd.DataFrame:
    """
    Generate simulated raw transaction data for fraud feature ingestion.

    Args:
        n: Number of rows to simulate.

    Returns:
        DataFrame containing raw transaction events with user, timestamp,
        transaction amount, device risk, and transaction velocity metrics.
    """
    now = datetime.now(timezone.utc)
    start_offset = timedelta(hours=n // 60 + 1)
    timestamps = pd.date_range(
        start=now - start_offset,
        periods=n,
        freq="min",
        tz=timezone.utc,
    )

    return pd.DataFrame(
        {
            "user_id": np.random.randint(1, 50, size=n),
            "event_timestamp": timestamps,
            "amount": np.random.exponential(scale=200, size=n),
            "tx_count_30m": np.random.poisson(3, size=n).astype(float),
            "device_risk": np.random.rand(n),
        }
    )


def run_ingestion() -> None:
    """
    Execute batch fraud feature ingestion and metadata registration to Feast.

    Workflow:
    1. Initialize Feast repository structure.
    2. Load simulated raw data.
    3. Apply fraud feature transformations.
    4. Persist transformed data to offline store as Parquet.
    5. Register entity and feature view definitions in Feast.
    6. Log ingestion summary.
    """
    repo_name: str = "fraud_repo"
    FeastRepositoryManager.initialize_repo(repo_name)

    df_raw: pd.DataFrame = simulate_raw_data()
    df_features: pd.DataFrame = add_fraud_features(df_raw)

    data_path: Path = Path(repo_name) / "data" / "fraud.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(data_path)

    store = FeatureStoreFactory.create(
        store_type="feast",
        repo_path=repo_name,
    )

    register_fraud_features(
        store,
        "user_id",
        str(data_path.absolute()),
    )

    print(f"Ingested {len(df_features)} rows into {data_path}")


if __name__ == "__main__":
    run_ingestion()
