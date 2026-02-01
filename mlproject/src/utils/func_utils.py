"""
Utility functions for handling time-series input/output shapes.

These helpers standardize tensor/array dimensionality for traditional
machine-learning models and provide automatic inference of model
input/output dimensions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def flatten_timeseries(x: np.ndarray) -> np.ndarray:
    """
    Flatten a 3D time-series array into 2D format.

    This is useful for traditional machine-learning models
    (e.g., XGBoost, RandomForest, Linear models) which expect
    2D input matrices.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (batch, seq_len, features) or any numeric array.

    Returns
    -------
    np.ndarray
        If x is 3D, returns array of shape (batch, seq_len * features).
        Otherwise, returns x unchanged.
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 3:
        batch, seq_len, n_feat = x.shape
        return x.reshape(batch, seq_len * n_feat)

    return x


def flatten_metrics_for_mlflow(metrics: dict[str, Any]) -> dict[str, float]:
    """
    Flatten a metrics dictionary into an MLflow-compatible format.

    MLflow only accepts scalar float values for logging. However, many
    time-series metrics may return vector outputs (e.g., multivariate or
    multi-horizon forecasts). This utility converts any `np.ndarray`
    metric values into multiple scalar metrics using the pattern:

        <metric_name>_<index> = float_value

    Example:
        Input:
            {
                "mae": np.array([0.12, 0.20, 0.31]),
                "rmse": 0.55,
            }

        Output:
            {
                "mae_0": 0.12,
                "mae_1": 0.20,
                "mae_2": 0.31,
                "rmse": 0.55,
            }

    Args:
        metrics (dict[str, Any]):
            Dictionary of metrics returned by the evaluator. Values may be
            scalars, numpy arrays, or unsupported types (ignored).

    Returns:
        dict[str, float]:
            A flattened metrics dictionary containing only scalar floats,
            safe for logging with MLflow.

    Notes:
        - Unsupported types are silently skipped.
        - Vector metrics preserve ordering (index corresponds to output dimension).
    """
    result: dict[str, float] = {}
    for key, val in metrics.items():
        if isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                result[f"{key}_{i}"] = float(v)
        elif isinstance(val, (np.floating, float, int)):
            result[key] = float(val)
        else:
            # Skip unsupported types
            continue
    return result


def get_env_path(var_name: str, default: str) -> Path:
    """
    Resolve a filesystem path from an environment variable.

    Parameters
    ----------
    var_name : str
        Name of the environment variable.
    default : str
        Default relative path if the variable is not set.

    Returns
    -------
    Path
        Resolved absolute path.
    """
    return Path(os.getenv(var_name, default)).resolve()
