from typing import Any, Dict, Union

import numpy as np

from mlproject.src.eval.base import BaseEvaluator


class RegressionEvaluator(BaseEvaluator):
    """Evaluator for general regression tasks, supporting multivariate outputs.

    Methods:
        - mae: Mean Absolute Error
        - mse: Mean Squared Error
        - rmse: Root Mean Squared Error
        - r2: Coefficient of Determination
        - evaluate: Compute all regression metrics at once
    """

    def mae(self, y_true, y_pred) -> Union[float, np.ndarray]:
        """
        Compute MAE per output (column-wise) and optionally aggregate.

        Args:
            y_true: True values (array-like)
            y_pred: Predicted values (array-like)

        Returns:
            float | np.ndarray: Aggregated scalar or per-output MAE
        """
        y_true, y_pred = self._to_numpy(y_true), self._to_numpy(y_pred)
        metric = np.mean(np.abs(y_true - y_pred), axis=0)
        return self._aggregate(metric)

    def mse(self, y_true, y_pred) -> Union[float, np.ndarray]:
        """
        Compute MSE per output.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float | np.ndarray: Aggregated scalar or per-output MSE
        """
        y_true, y_pred = self._to_numpy(y_true), self._to_numpy(y_pred)
        metric = np.mean((y_true - y_pred) ** 2, axis=0)
        return self._aggregate(metric)

    def rmse(self, y_true, y_pred) -> Union[float, np.ndarray]:
        """
        Compute RMSE per output.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float | np.ndarray: Aggregated scalar or per-output RMSE
        """
        return np.sqrt(self.mse(y_true, y_pred))

    def r2(self, y_true, y_pred) -> Union[float, np.ndarray]:
        """
        Compute R^2 score per output.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float | np.ndarray: Aggregated scalar or per-output R^2
        """
        y_true, y_pred = self._to_numpy(y_true), self._to_numpy(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        metric = 1 - ss_res / np.where(ss_tot != 0, ss_tot, 1)
        return self._aggregate(metric)

    def evaluate(
        self,
        y_true,
        y_pred,
        **kwargs: Any,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute all regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            dict: Dictionary with keys ["mae","mse","rmse","r2"] and per-output metrics
        """
        _ = kwargs
        return {
            "mae": self.mae(y_true, y_pred),
            "mse": self.mse(y_true, y_pred),
            "rmse": self.rmse(y_true, y_pred),
            "r2": self.r2(y_true, y_pred),
        }
