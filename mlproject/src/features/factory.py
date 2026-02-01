"""
Factory module for instantiating Feature Store clients in a vendor-agnostic,
type-safe manner.

This module exposes a centralized ``FeatureStoreFactory`` that constructs
feature store client instances (e.g., Feast) based on a logical store type and
offline repository path. The factory ensures a consistent return type conforming
to ``BaseFeatureStore`` so that upstream pipelines can rely on static typing and
polymorphic behavior without coupling to a specific provider.

The factory is designed for use in both training (offline PIT joins) and
inference (online lookups) workflows and surfaces unsupported store types as
explicit errors to aid static analysis, CI validation, and development debugging.
"""

from __future__ import annotations

from mlproject.src.features.base import BaseFeatureStore
from mlproject.src.features.feast_store import FeastFeatureStore


class FeatureStoreFactory:
    """
    Factory for creating strongly typed Feature Store client instances based on
    a logical provider identifier.

    The factory currently supports Feast and guarantees that the returned client
    implements the ``BaseFeatureStore`` interface, enabling static validation via
    ``mypy`` and contract enforcement via ``pylint`` in ML pipelines.

    This class must not be instantiated and only exposes static creation methods.
    """

    @staticmethod
    def create(
        store_type: str = "feast",
        repo_path: str = "feature_repo",
    ) -> BaseFeatureStore:
        """
        Create a feature store client instance for the specified provider.

        The method returns a concrete client implementing ``BaseFeatureStore``.
        Unsupported store types raise ``ValueError`` without silent suppression
        to ensure static analyzers and developers detect configuration issues.

        Args:
            store_type:
                Logical identifier of the feature store provider (case-insensitive).
                Default: ``"feast"``.
            repo_path:
                Path or URI to the offline feature repository used by the provider.
                Default: ``"feature_repo"``.

        Returns:
            A concrete ``BaseFeatureStore`` client instance.

        Raises:
            ValueError:
                If ``store_type`` is not supported by the factory implementation.
        """
        if store_type.lower() == "feast":
            return FeastFeatureStore(repo_path=repo_path)

        raise ValueError(f"Unsupported store type: {store_type}")
