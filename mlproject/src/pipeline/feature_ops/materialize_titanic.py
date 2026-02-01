"""
Materialization pipeline for Titanic features to Online Store.

This script:
1. Initializes a Feast-compatible feature store using the repository path.
2. Detects the materialization range from Parquet or CLI dates.
3. Pushes features into the online store.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import pandas as pd

from mlproject.src.features.factory import FeatureStoreFactory


def detect_range_from_parquet(data_path: str) -> Tuple[datetime, datetime]:
    """
    Infer materialization time range from a Parquet file.

    Args:
        data_path: Path to Parquet with "event_timestamp".

    Returns:
        Tuple of (start_date, end_date) UTC timezone-aware datetimes.

    Raises:
        FileNotFoundError: If file not found.
        ValueError: If timestamp column missing or empty.
    """
    file = Path(data_path)
    if not file.exists():
        raise FileNotFoundError(f"Parquet file not found: {data_path}")

    df = pd.read_parquet(data_path)
    if "event_timestamp" not in df.columns:
        raise ValueError("Missing 'event_timestamp' in Parquet.")

    if df["event_timestamp"].isnull().all():
        raise ValueError("'event_timestamp' contains only null values.")

    start_date = df["event_timestamp"].min().to_pydatetime()
    end_date = df["event_timestamp"].max().to_pydatetime()
    return start_date.astimezone(timezone.utc), end_date.astimezone(timezone.utc)


def parse_dates(start: str, end: str) -> Tuple[datetime, datetime]:
    """
    Convert CLI-provided start/end strings into UTC timezone-aware datetimes.

    Args:
        start: Start date string in ISO format.
        end: End date string in ISO format.

    Returns:
        Tuple of (start_date, end_date) UTC timezone-aware datetimes.

    Raises:
        ValueError: If parsing fails or start > end.
    """
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    if start_dt > end_dt:
        raise ValueError("start date must be earlier than or equal to end date.")

    return start_dt, end_dt


def main() -> None:
    """
    Run materialization from offline → online store for Titanic features.

    Raises:
        SystemExit: If sync fails due to invalid inputs or store errors.
    """
    parser = argparse.ArgumentParser(
        description="Titanic Feast offline-to-online materialization."
    )
    parser.add_argument("--repo", default="titanic_repo", help="Feast repo path.")
    parser.add_argument(
        "--data", required=False, help="Parquet path to detect materialization range."
    )
    parser.add_argument("--start", required=False, help="Start date (YYYY-MM-DD) UTC.")
    parser.add_argument("--end", required=False, help="End date (YYYY-MM-DD) UTC.")

    args = parser.parse_args()

    store: Any = FeatureStoreFactory.create(store_type="feast", repo_path=args.repo)

    if args.data:
        start_date, end_date = detect_range_from_parquet(args.data)
    elif args.start and args.end:
        start_date, end_date = parse_dates(args.start, args.end)
    else:
        print("Provide --data or both --start and --end.")
        return

    print(
        f"Materializing offline features into online store from "
        f"{start_date.isoformat()} to {end_date.isoformat()}..."
    )

    store.materialize(start_date, end_date)
    print("Materialization completed successfully.")


if __name__ == "__main__":
    main()
