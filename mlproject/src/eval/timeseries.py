from typing import Any, Dict, Union

import numpy as np

from mlproject.src.eval.regression import RegressionEvaluator


class TimeSeriesEvaluator(RegressionEvaluator):
    """Evaluator for time-series forecasting tasks.

    Extends RegressionEvaluator by adding:
        - mape: Mean Absolute Percentage Error
        - smape: Symmetric Mean Absolute Percentage Error
    """

    def mape(self, y_true, y_pred) -> Union[float, np.ndarray]:
        """
        Compute MAPE per output.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float | np.ndarray: Aggregated scalar or per-output MAPE in percentage
        """
        y_true, y_pred = self._to_numpy(y_true), self._to_numpy(y_pred)
        denom = np.where(np.abs(y_true) > 0, np.abs(y_true), 1.0)
        metric = np.mean(np.abs((y_true - y_pred) / denom), axis=0) * 100
        return self._aggregate(metric)

    def smape(self, y_true, y_pred) -> Union[float, np.ndarray]:
        """
        Compute sMAPE per output.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float | np.ndarray: Aggregated scalar or per-output sMAPE in percentage
        """
        y_true, y_pred = self._to_numpy(y_true), self._to_numpy(y_pred)
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        diff = np.abs(y_true - y_pred)
        metric = (
            np.mean(
                np.divide(diff, denom, out=np.zeros_like(diff), where=denom != 0),
                axis=0,
            )
            * 100
        )
        return self._aggregate(metric)

    def evaluate(
        self,
        y_true,
        y_pred,
        **kwargs: Any,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute regression + time-series-specific metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            dict: Dictionary with keys ["mae","mse","rmse","r2","mape","smape"]
                  and per-output metrics if multivariate
        """
        _ = kwargs
        metrics = super().evaluate(y_true, y_pred)
        metrics.update(
            {
                "mape": self.mape(y_true, y_pred),
                "smape": self.smape(y_true, y_pred),
            }
        )
        return metrics
