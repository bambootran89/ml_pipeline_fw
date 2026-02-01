"""
Batch ingestion pipeline for ETTH1 time-series features.

This script performs:
1. Raw CSV loading and timestamp normalization.
2. Time-series feature engineering (lag, rolling, cyclic time features).
3. Offline persistence to a Feast repository (Parquet).
4. Feature metadata registration into a Feast-compatible store.

The implementation preserves loader routing logic using the "feast://" prefix
in the configuration layer and avoids breaking external pipeline components.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List

import pandas as pd

from mlproject.src.features.definitions.etth1_features import register_etth1_features
from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager
from mlproject.src.features.transformers.ts_transformers import (
    add_lag_features,
    add_rolling_features,
    add_time_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def load_raw(csv_path: str, entity_col: str) -> pd.DataFrame:
    """
    Load raw ETTH1 data from CSV and normalize timestamp column.

    Args:
        csv_path: Local path to the raw CSV file.
        entity_col: Column name to be used as the Feast entity join key.

    Returns:
        DataFrame with "event_timestamp" set and the original date column removed.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the entity column name is empty.
    """
    file = Path(csv_path)
    if not file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not entity_col:
        raise ValueError("entity_col must be a non-empty string.")

    df = pd.read_csv(csv_path)
    df["event_timestamp"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop(columns=["date"])
    df[entity_col] = 1

    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ETTH1 time-series feature engineering using shared transformers.

    Transformations:
    - Lag features (24h) for HUFL and MUFL.
    - Rolling 12-step mean windows.
    - Cyclic time encoding for timestamp.

    Args:
        df: DataFrame containing at least HUFL, MUFL and "event_timestamp".

    Returns:
        Engineered DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required: List[str] = ["HUFL", "MUFL", "event_timestamp"]
    missing: List[str] = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns for engineering: {missing}")

    df = add_lag_features(df, ["HUFL", "MUFL"], lags=[24])
    df = add_rolling_features(df, ["HUFL", "MUFL"], windows=[12], agg="mean")
    df = add_time_features(df, "event_timestamp")

    return df


def save_to_offline_store(df: pd.DataFrame, repo_path: str) -> Path:
    """
    Persist engineered features into a Feast offline repository as Parquet.

    Args:
        df: Engineered DataFrame to persist.
        repo_path: Path to the Feast feature repository.

    Returns:
        Path to the written Parquet feature source.

    Raises:
        OSError: If repository initialization fails.
    """
    if not repo_path:
        raise ValueError("repo_path must be a non-empty string.")

    FeastRepositoryManager.initialize_repo(repo_path)

    out_path = Path(repo_path) / "data" / "features.parquet"
    out_path.parent.mkdir(exist_ok=True, parents=True)

    df.to_parquet(out_path)

    logger.info(
        "Saved engineered features to offline Feast store at '%s'", str(out_path)
    )

    return out_path


def main() -> None:
    """
    Run the ETTH1 batch ingestion and Feast metadata registration workflow.

    This function:
    - Parses CLI arguments.
    - Loads and engineers features.
    - Writes to offline store.
    - Registers entities and feature view metadata to Feast.

    Raises:
        SystemExit: If ingestion or registration fails.
    """
    parser = argparse.ArgumentParser(
        description="ETTH1 Feast batch ingestion and registration."
    )

    parser.add_argument("--csv", required=True, help="Path to raw ETTH1 CSV.")
    parser.add_argument("--repo", default="feature_repo", help="Feast repo path.")
    parser.add_argument("--entity", default="location_id", help="Entity join key.")

    args = parser.parse_args()

    try:
        df = load_raw(args.csv, args.entity)
        df = engineer(df)
        feat_source = save_to_offline_store(df, args.repo)

        store: Any = FeatureStoreFactory.create(
            store_type="feast",
            repo_path=args.repo,
        )

        register_etth1_features(store, args.entity, str(feat_source.absolute()))

        logger.info("Batch ingestion and registration completed.")
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
