from typing import Tuple

import numpy as np
from torch.utils.data import Dataset


class NumpyWindowDataset(Dataset):
    """
    PyTorch Dataset wrapper for numpy arrays.

    Stores features X (N, seq_len, feat) and targets y (N, horizon).
    """

    def __init__(self, x_array: np.ndarray, y_array: np.ndarray):
        """
        Args:
            x_array (np.ndarray): Input features of shape (N, seq_len, feat)
            y_array (np.ndarray): Targets of shape (N, horizon)
        """
        self.x_array = x_array.astype("float32")
        self.y_array = y_array.astype("float32")

    def __len__(self):
        return len(self.x_array)

    def __getitem__(self, idx):
        return self.x_array[idx], self.y_array[idx]


class TabularDataset:
    """
    Dataset wrapper for tabular machine learning tasks.

    This class provides a minimal and explicit abstraction for tabular
    datasets used in traditional machine learning workflows (e.g.
    XGBoost, LightGBM, sklearn models).

    It assumes:
        - Feature matrix `x` is strictly 2D
        - Target array `y` is 1D or 2D with a single column
        - No temporal dependency or sliding window logic is applied

    Typical use cases:
        - Regression datasets
        - Classification datasets
        - Cross-validation with index-based splitters
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the tabular dataset.

        Args:
            x (np.ndarray):
                Feature matrix of shape (n_samples, n_features).

            y (np.ndarray):
                Target values of shape (n_samples,) or (n_samples, 1).

        Raises:
            ValueError:
                If input arrays have incompatible shapes or dimensions.
        """
        if x.ndim != 2:
            raise ValueError(
                "Input features `x` must be a 2D array of shape "
                "(n_samples, n_features)."
            )

        if y.ndim not in (1, 2):
            raise ValueError(
                "Target `y` must be a 1D array of shape (n_samples,) "
                "or a 2D array of shape (n_samples, 1)."
            )

        if x.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in `x` and `y` must be identical.")

        self.x = x
        self.y = y

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the feature matrix and target array.

        This method intentionally performs no transformation or windowing.
        It serves as a unified access point for downstream components such as
        data splitters, trainers, and evaluators.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - x: Feature matrix of shape (n_samples, n_features)
                - y: Target array of shape (n_samples,) or (n_samples, 1)
        """
        return self.x, self.y
