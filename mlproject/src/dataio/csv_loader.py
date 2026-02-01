from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
from omegaconf import DictConfig
from pandas.errors import EmptyDataError, ParserError

from mlproject.src.dataio.base import BaseDatasetLoader

logger = logging.getLogger(__name__)


class CsvDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for CSV files.

    Supports:
    - Time-series datasets with a datetime index
    - Tabular datasets with an optional index column
    """

    def load(
        self,
        cfg: DictConfig,
        path: str,
        *,
        index_col: Optional[str] = None,
        data_type: str,
    ) -> pd.DataFrame:
        """
        Load a CSV dataset.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        index_col : Optional[str], default=None
            Column to be used as index.
        data_type : str
            Dataset type: "timeseries" or "tabular".

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        _ = cfg
        self._validate_data_type(data_type)

        columns = self._read_columns(path)
        if columns.empty:
            logger.warning("CSV file contains no columns: %s", path)
            return pd.DataFrame()

        if data_type == "timeseries":
            return self._load_timeseries(path, index_col, columns)

        return self._load_tabular(path, index_col)

    # Validation

    @staticmethod
    def _validate_data_type(data_type: str) -> None:
        if data_type not in {"timeseries", "tabular"}:
            raise ValueError(f"Unsupported data_type: {data_type}")

    # IO helpers

    @staticmethod
    def _read_columns(path: str) -> pd.Index:
        """
        Read CSV header only to obtain column names.
        """
        try:
            return pd.read_csv(path, nrows=0).columns
        except EmptyDataError:
            logger.warning("CSV file is empty: %s", path)
            return pd.Index([])
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"CSV file not found: {path}") from exc
        except ParserError as exc:
            raise ValueError(f"Failed to parse CSV header: {path}") from exc

    @staticmethod
    def _read_csv(
        path: str,
        *,
        parse_dates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Read CSV file with optional datetime parsing.
        """
        try:
            return pd.read_csv(path, parse_dates=parse_dates)
        except ParserError as exc:
            raise ValueError(f"Failed to parse CSV file: {path}") from exc

    # Load strategies
    def _load_timeseries(
        self,
        path: str,
        index_col: Optional[str],
        columns: pd.Index,
    ) -> pd.DataFrame:
        """
        Load time-series CSV dataset.
        """
        if not index_col:
            raise ValueError("index_col must be provided for timeseries data")

        if index_col not in columns:
            raise ValueError(
                f"Index column '{index_col}' not found in CSV columns: "
                f"{list(columns)}"
            )

        try:
            df = self._read_csv(path, parse_dates=[index_col])
        except (ValueError, TypeError):
            logger.warning(
                "Failed to parse '%s' as datetime, falling back to raw values",
                index_col,
            )
            df = self._read_csv(path)
        print("\n[CSV Profiling] DataFrame result:")
        print(f"  → rows         : {len(df)}")
        print(f"  → columns      : {list(df.columns)}")

        print("\n[CSV Profiling] head Sample:")
        print(df.head(5))

        print("\n[CSV Profiling] tail Sample:")
        print(df.tail(5))
        return df.set_index(index_col)

    def _load_tabular(
        self,
        path: str,
        index_col: Optional[str],
    ) -> pd.DataFrame:
        """
        Load tabular CSV dataset.
        """
        df = self._read_csv(path)

        if index_col and index_col in df.columns:
            logger.info(
                "Using column '%s' as index for tabular dataset",
                index_col,
            )
            df = df.set_index(index_col)

        return df
