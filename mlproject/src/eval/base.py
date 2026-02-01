from typing import Any, Dict, Optional, Union

import numpy as np
import torch


class BaseEvaluator:
    """
    Generic base evaluator for predictive models.
    Not tied to any task (regression, classification, time-series, etc.).

    Supports:
        - Multivariate outputs
        - Flexible aggregation over outputs: mean, median, or none
        - Input as numpy arrays, torch tensors, or lists
    """

    def __init__(
        self,
        aggregate: Optional[str] = "mean",
    ):
        """
        Args:
            aggregate: Aggregation strategy for multivariate outputs:
                - 'mean'  : average metric over outputs
                - 'median': median over outputs
                - None    : return per-output metrics
        """
        self.aggregate = aggregate

    @staticmethod
    def _to_numpy(y: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(y, torch.Tensor):
            return y.detach().cpu().numpy()
        return np.array(y)

    def _aggregate(self, metric: np.ndarray) -> Union[float, np.ndarray]:
        """Aggregate metric over outputs if requested."""
        if self.aggregate == "mean":
            return float(np.mean(metric))
        elif self.aggregate == "median":
            return float(np.median(metric))
        return metric

    def evaluate(
        self,
        y_true: Any,
        y_pred: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generic evaluation interface.
        Must be implemented in subclasses.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            dict: {metric_name: value}
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
