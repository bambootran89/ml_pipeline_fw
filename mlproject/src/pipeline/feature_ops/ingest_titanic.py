"""
Batch ingestion pipeline for Titanic tabular features.

This script:
1. Loads raw CSV and adds timestamp/entity columns.
2. Persists features to Feast offline store (Parquet).
3. Registers metadata into Feast.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from mlproject.src.features.definitions.titanic_features import (
    register_titanic_features,
)
from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_raw(csv_path: str, entity_col: str) -> pd.DataFrame:
    """
    Load raw Titanic data from CSV and normalize timestamp column.

    Args:
        csv_path: Local path to the Titanic CSV file.
        entity_col: Column name to be used as the Feast entity join key.

    Returns:
        DataFrame with "event_timestamp" column and entity_col added.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If entity_col is empty.
    """
    file = Path(csv_path)
    if not file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not entity_col:
        raise ValueError("entity_col must be a non-empty string.")

    df = pd.read_csv(csv_path)
    df["Sex"] = df["Sex"].astype(str)
    df["Embarked"] = df["Embarked"].astype(str)
    df["Sex"] = df["Sex"].fillna("unknown")
    df["Embarked"] = df["Embarked"].fillna("unknown")

    df["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    df[entity_col] = df["PassengerId"].astype(int)

    return df


def save_to_offline_store(df: pd.DataFrame, repo_path: str) -> Path:
    """
    Persist Titanic features to a Feast offline repository as Parquet.

    Args:
        df: DataFrame with engineered/preprocessed features.
        repo_path: Path to the Feast feature repository.

    Returns:
        Path to the saved Parquet file.

    Raises:
        ValueError: If repo_path is empty.
        OSError: If repository initialization fails.
    """
    if not repo_path:
        raise ValueError("repo_path must be a non-empty string.")

    FeastRepositoryManager.initialize_repo(repo_path)

    out_path = Path(repo_path) / "data" / "titanic.parquet"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(out_path)

    logger.info("Saved Titanic features to offline store at '%s'", str(out_path))

    return out_path


def main() -> None:
    """
    Run Titanic batch ingestion and Feast registration workflow.

    Steps:
    - Parse CLI arguments.
    - Load raw CSV and preprocess features.
    - Persist to offline store.
    - Register feature definitions to Feast.

    Raises:
        SystemExit: If any step fails.
    """
    parser = argparse.ArgumentParser(
        description="Titanic Feast batch ingestion and registration."
    )
    parser.add_argument("--csv", required=True, help="Path to raw Titanic CSV.")
    parser.add_argument("--repo", default="titanic_repo", help="Feast repo path.")
    parser.add_argument("--entity", default="passenger_id", help="Entity join key.")

    args = parser.parse_args()

    try:
        df = load_raw(args.csv, args.entity)
        save_to_offline_store(df, args.repo)

        store: Any = FeatureStoreFactory.create(store_type="feast", repo_path=args.repo)
        register_titanic_features(store, args.entity, "data/titanic.parquet")

        logger.info("Batch ingestion and registration completed.")
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
