"""
Sensor data ingestion pipeline with lag and rolling window feature engineering,
Parquet persistence, and Feast feature store registration.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlproject.src.features.definitions.forecast_features import (
    register_forecast_definitions,
)
from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager
from mlproject.src.features.transformers.forecast_transformers import (
    apply_forecast_engineering,
)


def generate_raw_sensor_data(periods: int = 200) -> pd.DataFrame:
    """
    Generate synthetic raw sensor time-series data without lag/rolling computation.

    Args:
        periods: Number of hourly data points to generate.

    Returns:
        pd.DataFrame: Raw sensor data containing temperature and demand signals.
    """
    now: datetime = datetime.now(timezone.utc)
    ts = pd.date_range(
        start=now - timedelta(hours=periods),
        periods=periods,
        freq="H",
        tz=timezone.utc,
    )

    return pd.DataFrame(
        {
            "location_id": np.ones(periods, dtype=np.int32),
            "event_timestamp": ts,
            "temperature": (np.random.randn(periods) * 10 + 25).astype(np.float64),
            "demand": (np.random.randn(periods) * 100 + 500).astype(np.float64),
        }
    )


def run_ingestion() -> None:
    """
    Execute the end-to-end ingestion pipeline:
    - Generate raw sensor data
    - Apply lag/rolling feature engineering
    - Persist to Parquet (offline store)
    - Register entity and feature view in Feast
    """
    try:
        repo_name: str = "ts_repo"

        # Initialize Feast repo and ensure data directory exists.
        FeastRepositoryManager.initialize_repo(repo_name)
        data_dir: Path = Path(repo_name) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate raw sensor data.
        df_raw: pd.DataFrame = generate_raw_sensor_data(periods=200)

        # 2. Apply feature engineering (lag + rolling windows).
        df_features: pd.DataFrame = apply_forecast_engineering(df_raw)

        # 3. Persist engineered features to Parquet.
        out_path: Path = data_dir / "ts_features.parquet"
        df_features.to_parquet(out_path)

        # 4. Create Feast feature store and register definitions.
        store: Any = FeatureStoreFactory.create(store_type="feast", repo_path=repo_name)
        register_forecast_definitions(
            store=store,
            entity_col="location_id",
            source_path=str(out_path.resolve()),
        )

        print(f"[INFO] Ingested and registered forecast features: {out_path}")

    except Exception as exc:  # Safe catch for pylint E0712 compliance
        print(f"[ERROR] Ingestion pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_ingestion()
