"""
Online feature materialization and retrieval demo for recommendation system
inference using Feast. Performs a short materialization window and fetches
real-time user-item interaction features for prediction input.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from mlproject.src.features.examples.ingest_recsys import run_pipeline
from mlproject.src.features.factory import FeatureStoreFactory

logger = logging.getLogger(__name__)


def demo_retrieval() -> None:
    """
    Run a materialization and online feature retrieval example.

    Connects to an existing Feast feature store, materializes a short
    time window to the online store, and retrieves online features for
    a list of users to simulate model inference input.
    """
    run_pipeline()
    store: Any = FeatureStoreFactory.create(
        store_type="feast",
        repo_path="recsys_repo",
    )

    now: datetime = datetime.now(timezone.utc)

    store.materialize(
        start_date=now - timedelta(days=7),
        end_date=now + timedelta(minutes=10),
    )

    entity_rows: List[Dict[str, Any]] = [
        {"user_id": 1},
        {"user_id": 2},
    ]

    results = store.get_online_features(
        entity_rows=entity_rows,
        features=[
            "recsys_view:like_ratio",
            "recsys_view:view_count",
            "recsys_view:item_id",
        ],
    )

    print("--- RecSys Online Prediction Input ---")

    for res in results:
        user_id = res.get("user_id")
        item_id = res.get("item_id")
        like_ratio = res.get("like_ratio")

        if user_id is None or item_id is None or like_ratio is None:
            logger.warning("Missing feature values for entity row: %s", res)
            continue

        print(
            f"User {user_id} -> Recommend Item: {item_id} "
            f"(Like Ratio: {float(like_ratio):.2f})"
        )


if __name__ == "__main__":
    demo_retrieval()
