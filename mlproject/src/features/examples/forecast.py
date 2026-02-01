"""
Demo for Feast feature retrieval: historical lookup and online feature serving
after materialization to the online store.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from mlproject.src.features.examples.ingest_forecast import run_ingestion
from mlproject.src.features.factory import FeatureStoreFactory

logger = logging.getLogger(__name__)


def demo() -> None:
    """Run historical and online feature retrieval demo."""
    run_ingestion()
    try:
        store: Any = FeatureStoreFactory.create(store_type="feast", repo_path="ts_repo")
        now: datetime = datetime.now(timezone.utc)

        # 1. Historical feature retrieval
        entity_df: pd.DataFrame = pd.DataFrame(
            {
                "location_id": [1, 1, 1],
                "event_timestamp": [
                    now - timedelta(hours=1),
                    now - timedelta(hours=25),
                    now - timedelta(hours=50),
                ],
            }
        )

        hist_df: pd.DataFrame = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "forecast_view:temperature_lag24",
                "forecast_view:demand_roll12_mean",
            ],
        )  # Convert Feast RetrievalJob to DataFrame if using Feast Python API

        print("--- Historical Features ---\n", hist_df.head())

        # 2. Materialize to online store
        store.materialize(
            start_date=now - timedelta(days=10),
            end_date=now + timedelta(minutes=5),
        )

        # 3. Online feature retrieval
        online_res: Any = store.get_online_features(
            entity_rows=[{"location_id": 1}],
            features=["forecast_view:temperature_lag24"],
        )

        print(f"\n--- Online Feature (location_id=1) ---\n{online_res}")

    except Exception as exc:
        logger.error("Feature retrieval failed: %s", exc)
        print(f"[ERROR] Demo failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
