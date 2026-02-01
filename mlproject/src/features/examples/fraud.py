"""
Feast offline and online feature retrieval demo for fraud detection.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from mlproject.src.features.examples.ingest_fraud import run_ingestion
from mlproject.src.features.factory import FeatureStoreFactory


def demo() -> None:
    """
    Demonstrate offline (historical) and online feature retrieval using Feast.

    Offline mode is used for training pipelines via entity DataFrame joins.
    Online mode is used for real-time inference via direct entity row queries.
    """
    run_ingestion()
    store = FeatureStoreFactory.create(store_type="feast", repo_path="fraud_repo")
    now: datetime = datetime.now(timezone.utc)

    entity_df: pd.DataFrame = pd.DataFrame(
        {
            "user_id": [5, 10],
            "event_timestamp": [
                now - timedelta(minutes=1),
                now - timedelta(minutes=5),
            ],
        }
    )

    hist_df = store.get_historical_features(
        entity_df,
        features=[
            "fraud_view:risk_score",
            "fraud_view:amount",
        ],
    )

    print("--- Offline Sample ---")
    print(hist_df)

    online_feats = store.get_online_features(
        entity_rows=[{"user_id": 5}],
        features=[
            "fraud_view:risk_score",
            "fraud_view:device_risk",
        ],
    )

    print("\n--- Online Sample ---")
    print(online_feats)


if __name__ == "__main__":
    demo()
