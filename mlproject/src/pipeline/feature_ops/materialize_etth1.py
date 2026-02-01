"""
Materialization pipeline to sync offline Feast store into the online store.

This script:
1. Initializes a Feast-compatible feature store using the repository path.
2. Detects the time range either from a Parquet file or CLI-provided dates.
3. Runs materialization to push features into the online store.

The implementation avoids modifying interfaces that could break external
pipeline components relying on BaseDatasetLoader or FeatureStoreFactory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import pandas as pd

from mlproject.src.features.factory import FeatureStoreFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def detect_range_from_parquet(data_path: str) -> Tuple[datetime, datetime]:
    """
    Infer materialization time range from a Parquet file.

    Args:
        data_path: Local file path to engineered Feast feature Parquet.

    Returns:
        (start_date, end_date) timezone-aware UTC datetimes.

    Raises:
        FileNotFoundError: If the Parquet file does not exist.
        ValueError: If the timestamp column is missing or empty.
    """
    file = Path(data_path)
    if not file.exists():
        raise FileNotFoundError(f"Parquet file not found: {data_path}")

    df = pd.read_parquet(data_path)

    if "event_timestamp" not in df.columns:
        raise ValueError("Parquet file missing 'event_timestamp' column.")

    if df["event_timestamp"].isnull().all():
        raise ValueError("'event_timestamp' contains only null values.")

    start_date = df["event_timestamp"].min().to_pydatetime()
    end_date = df["event_timestamp"].max().to_pydatetime()

    return start_date.astimezone(timezone.utc), end_date.astimezone(timezone.utc)


def parse_dates(
    start: str,
    end: str,
) -> Tuple[datetime, datetime]:
    """
    Convert CLI-provided start/end strings into UTC timezone-aware datetimes.

    Args:
        start: Start date string in ISO format (YYYY-MM-DD).
        end: End date string in ISO format (YYYY-MM-DD).

    Returns:
        (start_date, end_date) UTC timezone-aware datetimes.

    Raises:
        ValueError: If parsing fails or date order is invalid.
    """
    try:
        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    except Exception as exc:
        raise ValueError(f"Invalid date format: {exc}") from exc

    if start_dt > end_dt:
        raise ValueError("start date must be earlier than or equal to end date.")

    return start_dt, end_dt


def main() -> None:
    """
    Run the materialization sync process for offline → online Feast store.

    Raises:
        SystemExit: If sync fails due to invalid inputs or store errors.
    """
    parser = argparse.ArgumentParser(
        description="Feast offline-to-online materialization sync."
    )

    parser.add_argument(
        "--repo",
        default="feature_repo",
        help="Path to Feast feature repository.",
    )

    parser.add_argument(
        "--data",
        required=False,
        help="Parquet path to auto-detect materialization range.",
    )

    parser.add_argument(
        "--start",
        required=False,
        help="Start date (YYYY-MM-DD) UTC.",
    )

    parser.add_argument(
        "--end",
        required=False,
        help="End date (YYYY-MM-DD) UTC.",
    )

    args = parser.parse_args()

    try:
        store: Any = FeatureStoreFactory.create(
            store_type="feast",
            repo_path=args.repo,
        )

        if args.data:
            start_date, end_date = detect_range_from_parquet(args.data)
        elif args.start and args.end:
            start_date, end_date = parse_dates(args.start, args.end)
        else:
            logger.error("Provide --data or both --start and --end.")
            sys.exit(1)

        logger.info(
            "Materializing offline features into online store from '%s' to '%s'",
            start_date.date().isoformat(),
            end_date.date().isoformat(),
        )

        store.materialize(start_date, end_date)

        logger.info("Materialization completed successfully.")

    except (FileNotFoundError, ValueError) as input_err:
        logger.error("Input error: %s", input_err)
        sys.exit(1)
    except Exception as exc:
        logger.error("Materialization sync failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
