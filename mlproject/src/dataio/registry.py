from __future__ import annotations

from typing import Dict, Type

from mlproject.src.dataio.base import BaseDatasetLoader
from mlproject.src.dataio.csv_loader import CsvDatasetLoader
from mlproject.src.dataio.feast_loader import FeastDatasetLoader


class DatasetLoaderRegistry:
    """
    Registry mapping data source types to DatasetLoader implementations.

    This registry resolves the appropriate DatasetLoader
    based on the data source path (e.g., file extension).
    """

    _REGISTRY: Dict[str, Type[BaseDatasetLoader]] = {
        ".csv": CsvDatasetLoader,
        "feast://": FeastDatasetLoader,
    }

    @classmethod
    def get_loader(cls, path: str) -> BaseDatasetLoader:
        """
        Resolve a DatasetLoader based on the given data source path.

        Parameters
        ----------
        path : str
            Data source path.

        Returns
        -------
        BaseDatasetLoader
            Instantiated dataset loader.

        Raises
        ------
        ValueError
            If the data source is unsupported.
        """
        path_lower = path.lower()

        for prefix, loader_cls in cls._REGISTRY.items():
            if path_lower.startswith(prefix):
                return loader_cls()

        for suffix, loader_cls in cls._REGISTRY.items():
            if path_lower.endswith(suffix):
                return loader_cls()

        raise ValueError(f"Unsupported data source: {path}")
