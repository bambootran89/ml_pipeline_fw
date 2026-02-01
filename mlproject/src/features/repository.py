"""
Feast Feature Store repository bootstrap and configuration manager.

This module bootstraps a minimal Feast repository layout on disk and
generates a deterministic `feature_store.yaml` configuration file that
supports PIT-compatible batch retrieval and online feature serving.

Key design choices:
- Uses relative paths for registry and online store persistence.
- Sets entity key serialization version to 3 for stable entity encoding.
- Declares offline and online store types explicitly to avoid runtime
  path guessing failures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


class FeastRepositoryManager:
    """
    Bootstrapper for Feast repository structure and configuration.

    This class must not be instantiated. It exposes static utilities to:
    1. Create a repository root directory.
    2. Ensure a `data/` subdirectory exists for offline sources.
    3. Generate a deterministic `feature_store.yaml` file compatible
       with Feast v0.58.0+ batch PIT joins and online lookups.
    """

    @staticmethod
    def initialize_repo(repo_path: str = "feature_repo") -> None:
        """
        Create a Feast repository directory and write configuration to disk.

        Generated config properties:
        - `project`: Feature store project name.
        - `registry`: Schema and metadata storage backend path.
        - `provider`: Execution environment.
        - `entity_key_serialization_version`: Entity key encoding version.
        - `offline_store.type`: Offline PIT retrieval backend.
        - `online_store.type`: Online serving backend.
        - `online_store.path`: Persistence path for SQLite (if used).

        Args:
            repo_path: Filesystem path to bootstrap the repository root.

        Raises:
            OSError: If directory or configuration file creation fails.
        """
        base_path = Path(repo_path)
        base_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        data_dir = base_path / "data"
        data_dir.mkdir(
            exist_ok=True,
        )

        config: Dict[str, Any] = {
            "project": "mlproject",
            "registry": "data/registry.db",
            "provider": "local",
            "entity_key_serialization_version": 3,
            "offline_store": {
                "type": "file",
            },
            "online_store": {
                "type": "sqlite",
                "path": "data/online.db",
            },
        }

        with open(base_path / "feature_store.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f)


def get_supported_options() -> Dict[str, List[str]]:
    """
    Return commonly supported backend options for offline and online stores.

    These options depend on infrastructure and plugin availability,
    but Feast 0.58.0+ commonly supports the following:

    Returns:
        Dictionary containing offline and online store backend lists.
    """
    return {
        "offline_store_backends": [
            "file",
            "bigquery",
            "snowflake",
            "redshift",
            "spark",
            "trino",
            "trino",
        ],
        "online_store_backends": [
            "sqlite",
            "redis",
            "dynamodb",
            "datastore",
            "bigtable",
            "cassandra",
            "elasticsearch",
        ],
    }
