from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlproject.src.dataio.csv_loader import CsvDatasetLoader
from mlproject.src.dataio.entrypoint import load_dataset
from mlproject.src.dataio.registry import DatasetLoaderRegistry


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """
    Create a simple CSV file for testing.
    """
    path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02"],
            "x": [1.0, 2.0],
            "y": [10.0, 20.0],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_registry_resolves_csv_loader(sample_csv: Path) -> None:
    """
    DatasetLoaderRegistry should resolve CSVDatasetLoader for .csv files.
    """
    loader = DatasetLoaderRegistry.get_loader(str(sample_csv))
    assert isinstance(loader, CsvDatasetLoader)


def test_load_timeseries_csv(sample_csv: Path) -> None:
    """
    CSV loader should load time-series data with index column.
    """
    df = load_dataset(
        cfg={},
        path=str(sample_csv),
        index_col="date",
        data_type="timeseries",
    )

    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "date"
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 2


def test_load_tabular_csv(sample_csv: Path) -> None:
    """
    CSV loader should load tabular data without forcing an index.
    """
    df = load_dataset(
        cfg={},
        path=str(sample_csv),
        index_col=None,
        data_type="tabular",
    )

    assert isinstance(df, pd.DataFrame)
    assert "date" in df.columns
    assert df.index.name is None


def test_missing_index_column_raises(sample_csv: Path) -> None:
    """
    Missing index column in time-series mode should raise ValueError.
    """
    with pytest.raises(ValueError):
        load_dataset(
            cfg={},
            path=str(sample_csv),
            index_col="missing_col",
            data_type="timeseries",
        )


def test_empty_csv_returns_empty_df(tmp_path: Path) -> None:
    """
    Empty CSV file should return an empty DataFrame.
    """
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("")

    df = load_dataset(
        cfg={},
        path=str(empty_path),
        index_col="date",
        data_type="timeseries",
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_unsupported_data_source_raises() -> None:
    """
    Unsupported data source should raise ValueError.
    """
    with pytest.raises(ValueError):
        DatasetLoaderRegistry.get_loader("data.unknown")
