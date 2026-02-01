"""Feature composition utilities for flexible pipeline wiring.

This module provides utilities to compose features from multiple sources
with automatic shape alignment and validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class FeatureComposer:
    """Compose features from multiple pipeline sources with shape alignment."""

    @staticmethod
    def compose_features(
        base_features: Any,
        additional_features: Optional[Dict[str, Any]] = None,
        align_method: str = "auto",
    ) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]]]:
        """Compose features from base and additional sources.

        Parameters
        ----------
        base_features : Any
            Primary feature source (DataFrame or ndarray).
        additional_features : Optional[Dict[str, Any]]
            Dict mapping source names to feature arrays/DataFrames.
        align_method : str
            How to align shapes: 'auto', 'broadcast', 'repeat', 'concat'.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Tuple[int, int]]]
            - Composed features as DataFrame
            - Metadata dict mapping source -> (start_idx, end_idx)

        Raises
        ------
        ValueError
            If shapes cannot be aligned.
        """
        # Convert base to DataFrame
        base_df = FeatureComposer._to_dataframe(base_features, "base")
        n_samples = len(base_df)

        # Reset index to ensure proper alignment when concatenating
        # This handles cases where base has Timestamp index but additional
        # features have integer index
        base_df = base_df.reset_index(drop=True)

        # Track feature positions
        metadata: Dict[str, Tuple[int, int]] = {"base": (0, base_df.shape[1])}

        if not additional_features:
            return base_df, metadata
        current_idx = base_df.shape[1]

        # Compose additional features
        for source_name, features in additional_features.items():
            if features is None:
                continue

            aligned = FeatureComposer._align_features(features, n_samples, align_method)

            feature_df = FeatureComposer._to_dataframe(aligned, source_name)
            # Reset index to match base_df
            feature_df = feature_df.reset_index(drop=True)

            # Track position
            n_cols = feature_df.shape[1]
            metadata[source_name] = (current_idx, current_idx + n_cols)
            current_idx += n_cols

            # Concatenate
            base_df = pd.concat([base_df, feature_df], axis=1)

        # Keep original column names - base columns stay as-is,
        # additional features already have prefixed names from _ndarray_to_df
        return base_df, metadata

    @staticmethod
    def _to_dataframe(data: Any, prefix: str) -> pd.DataFrame:
        """Convert data to DataFrame with proper column names.

        Parameters
        ----------
        data : Any
            Input data (DataFrame, ndarray, list).
        prefix : str
            Prefix for auto-generated column names.

        Returns
        -------
        pd.DataFrame
            Normalized DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()

        if isinstance(data, np.ndarray):
            return FeatureComposer._ndarray_to_df(data, prefix)

        if isinstance(data, (list, tuple)):
            return FeatureComposer._ndarray_to_df(np.array(data), prefix)

        raise TypeError(f"Cannot convert {type(data).__name__} to DataFrame")

    @staticmethod
    def _ndarray_to_df(arr: np.ndarray, prefix: str) -> pd.DataFrame:
        """Convert numpy array to DataFrame handling 1D/2D/3D cases.

        Parameters
        ----------
        arr : np.ndarray
            Input array.
        prefix : str
            Column name prefix.

        Returns
        -------
        pd.DataFrame
            Converted DataFrame.

        Raises
        ------
        ValueError
            If array has unsupported dimensions.
        """
        if arr.ndim == 1:
            return pd.DataFrame({f"{prefix}_0": arr})

        if arr.ndim == 2:
            cols = [f"{prefix}_{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols)

        if arr.ndim == 3:
            # Flatten to 2D: (n_samples, steps * features)
            n_samples, n_steps, n_features = arr.shape
            reshaped = arr.reshape(n_samples, n_steps * n_features)
            cols = [
                f"{prefix}_s{s}_f{f}" for s in range(n_steps) for f in range(n_features)
            ]
            return pd.DataFrame(reshaped, columns=cols)

        raise ValueError(
            f"Cannot convert {arr.ndim}D array to DataFrame. " "Supported: 1D, 2D, 3D"
        )

    @staticmethod
    def _align_features(
        features: Any,
        target_samples: int,
        method: str,
    ) -> np.ndarray:
        """Align feature array to target number of samples.

        Parameters
        ----------
        features : Any
            Feature array to align.
        target_samples : int
            Target number of samples.
        method : str
            Alignment method.

        Returns
        -------
        np.ndarray
            Aligned feature array.

        Raises
        ------
        ValueError
            If alignment fails.
        """
        arr = np.asarray(features)
        current_samples = arr.shape[0] if arr.ndim > 0 else 1

        if current_samples == target_samples:
            return arr

        if method == "auto":
            method = FeatureComposer._infer_align_method(
                current_samples, target_samples
            )

        if method == "broadcast":
            return FeatureComposer._broadcast_align(arr, target_samples)

        if method == "repeat":
            return FeatureComposer._repeat_align(arr, target_samples)

        if method == "truncate":
            return FeatureComposer._truncate_align(arr, target_samples)

        if method == "pad_start":
            return FeatureComposer._pad_start_align(arr, target_samples)

        raise ValueError(f"Unknown alignment method: {method}")

    @staticmethod
    def _infer_align_method(current: int, target: int) -> str:
        """Infer best alignment method based on sizes.

        Parameters
        ----------
        current : int
            Current sample size.
        target : int
            Target sample size.

        Returns
        -------
        str
            Inferred method.
        """
        if current == 1:
            return "broadcast"

        if current < target and target % current == 0:
            return "repeat"

        if current > target:
            return "truncate"

        # Pad at start: for windowed features that have fewer samples
        # (e.g., 171 windows from 200 samples due to sliding window)
        # Padding at start because windows start from position input_chunk
        if current < target:
            return "pad_start"

        raise ValueError(
            f"Cannot auto-align {current} samples to {target}. "
            "Specify align_method explicitly."
        )

    @staticmethod
    def _broadcast_align(arr: np.ndarray, target: int) -> np.ndarray:
        """Broadcast single sample to multiple samples.

        Parameters
        ----------
        arr : np.ndarray
            Input array with shape (1, ...).
        target : int
            Target number of samples.

        Returns
        -------
        np.ndarray
            Broadcasted array with shape (target, ...).
        """
        if arr.ndim == 1:
            return np.repeat(arr, target)

        return np.repeat(arr, target, axis=0)

    @staticmethod
    def _repeat_align(arr: np.ndarray, target: int) -> np.ndarray:
        """Repeat array to reach target samples.

        Parameters
        ----------
        arr : np.ndarray
            Input array.
        target : int
            Target number of samples.

        Returns
        -------
        np.ndarray
            Repeated array.
        """
        current = arr.shape[0]
        n_repeats = target // current

        if arr.ndim == 1:
            return np.tile(arr, n_repeats)

        return np.tile(arr, (n_repeats, 1))

    @staticmethod
    def _truncate_align(arr: np.ndarray, target: int) -> np.ndarray:
        """Truncate array to target samples.

        Parameters
        ----------
        arr : np.ndarray
            Input array.
        target : int
            Target number of samples.

        Returns
        -------
        np.ndarray
            Truncated array.
        """
        return arr[:target]

    @staticmethod
    def _pad_start_align(arr: np.ndarray, target: int) -> np.ndarray:
        """Pad array at start to reach target samples.

        Useful for windowed features that have fewer samples due to
        sliding window (e.g., 171 windows from 200 samples).
        Padding at start because windows start from position input_chunk.

        Parameters
        ----------
        arr : np.ndarray
            Input array.
        target : int
            Target number of samples.

        Returns
        -------
        np.ndarray
            Padded array.
        """
        current = arr.shape[0]
        if current >= target:
            return arr[:target]

        n_pad = target - current

        if arr.ndim == 1:
            pad_vals = np.repeat(arr[0], n_pad)
            return np.concatenate([pad_vals, arr])

        # For 2D+ arrays, pad with first row repeated
        pad_vals = np.tile(arr[:1], (n_pad,) + (1,) * (arr.ndim - 1))
        return np.concatenate([pad_vals, arr], axis=0)


class FeatureExtractor:
    """Extract features from pipeline context with wiring support."""

    @staticmethod
    def extract_from_context(
        context: Dict[str, Any],
        feature_keys: List[str],
        required: bool = True,
    ) -> Dict[str, Any]:
        """Extract multiple feature sources from context.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.
        feature_keys : List[str]
            Keys to extract.
        required : bool
            Whether to raise error on missing keys.

        Returns
        -------
        Dict[str, Any]
            Extracted features dict.

        Raises
        ------
        KeyError
            If required key is missing.
        """
        extracted: Dict[str, Any] = {}

        for key in feature_keys:
            value = context.get(key)

            if value is None and required:
                raise KeyError(f"Required feature key '{key}' not found in context")

            if value is not None:
                extracted[key] = value

        return extracted

    @staticmethod
    def get_feature_shape(features: Any) -> Tuple[int, ...]:
        """Get shape of feature array/DataFrame.

        Parameters
        ----------
        features : Any
            Feature data.

        Returns
        -------
        Tuple[int, ...]
            Shape tuple.
        """
        if isinstance(features, pd.DataFrame):
            return features.shape

        if isinstance(features, np.ndarray):
            return features.shape

        if isinstance(features, (list, tuple)):
            return (len(features),)

        return (1,)
