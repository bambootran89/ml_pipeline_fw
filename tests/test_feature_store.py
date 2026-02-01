"""
Professional integration tests for a Feast Feature Store client
(Feast v0.58.0+). Validates PIT join correctness, offline-to-online
materialization, API-ready online record pivoting, and behavior on
missing entities.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Generator, List

import pandas as pd
import pytest

from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager
from mlproject.src.features.strategies import BaseFeatureStore


@pytest.fixture(scope="function")
def repo_path(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Create a temporary Feast repository directory and ensure teardown
    after test execution.
    """
    path: Path = tmp_path / "mlops_feature_repo"
    FeastRepositoryManager.initialize_repo(str(path))
    yield path
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture(scope="function")
def sample_data(repo_path: Path) -> Path:
    """
    Generate a UTC-aware offline data source (Parquet) containing
    multiple feature values across different timestamps.
    """
    data_file: Path = repo_path / "data" / "user_stats.parquet"
    data_file.parent.mkdir(parents=True, exist_ok=True)

    now: datetime = datetime.now(timezone.utc)
    df: pd.DataFrame = pd.DataFrame(
        {
            "user_id": [101, 101, 102],
            "event_timestamp": [
                now - timedelta(days=5),
                now - timedelta(days=1),
                now - timedelta(days=2),
            ],
            "credit_score": [600.0, 750.0, 800.0],
            "is_active": [1, 1, 0],
        }
    )
    df.to_parquet(data_file)
    return data_file


@pytest.fixture(scope="function")
def store(repo_path: Path, sample_data: Path) -> BaseFeatureStore:
    """
    Initialize a FeastFeatureStore via factory and register:
    - A join-key entity (``user_id``).
    - A PIT-safe feature view (``user_stats``).
    """
    store_inst = FeatureStoreFactory.create("feast", str(repo_path))

    store_inst.register_entity(
        name="user",
        join_key="user_id",
        description="Primary join key for user features.",
        value_type="int",
    )

    store_inst.register_feature_view(
        name="user_stats",
        entities=["user"],
        schema={"credit_score": "float", "is_active": "int"},
        source_path=str(sample_data.absolute()),
        ttl_days=30,
    )
    return store_inst


def test_pit_join_accuracy(store: BaseFeatureStore) -> None:
    """
    Validate point-in-time (PIT) join semantics by asserting that the
    closest past feature record relative to the query timestamp is
    returned (no future leakage).
    """
    now: datetime = datetime.now(timezone.utc)

    entity_df: pd.DataFrame = pd.DataFrame(
        {"user_id": [101], "event_timestamp": [now - timedelta(days=4)]}
    )

    hist: pd.DataFrame = store.get_historical_features(
        entity_df=entity_df, features=["user_stats:credit_score"]
    )

    assert hist["credit_score"].iloc[0] == pytest.approx(600.0)


def test_online_serving_consistency(store: BaseFeatureStore) -> None:
    """
    Validate offline-to-online materialization and ensure that online
    feature responses are returned as a list of dictionaries, ready
    for deterministic API serving.
    """
    now: datetime = datetime.now(timezone.utc)

    store.materialize(
        start_date=now - timedelta(days=10),
        end_date=now + timedelta(minutes=5),
    )

    online_data: List[Dict[str, object]] = store.get_online_features(
        entity_rows=[{"user_id": 101}, {"user_id": 102}],
        features=["user_stats:credit_score", "user_stats:is_active"],
    )

    assert isinstance(online_data, list)
    assert len(online_data) == 2

    user_101: Dict[str, object] = next(
        row for row in online_data if row.get("user_id") == 101
    )
    assert user_101.get("credit_score") == pytest.approx(750.0)
    assert user_101.get("is_active") == 1


def test_empty_retrieval_behavior(store: BaseFeatureStore) -> None:
    """
    Ensure that requests for non-existent entity keys do not raise
    runtime errors and return ``None`` for unresolved features.
    """
    online_data: List[Dict[str, object]] = store.get_online_features(
        entity_rows=[{"user_id": 999}],
        features=["user_stats:credit_score"],
    )

    assert online_data[0].get("credit_score") is None
