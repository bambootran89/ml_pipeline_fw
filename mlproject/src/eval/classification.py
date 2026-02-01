from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlproject.src.eval.base import BaseEvaluator


class ClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for binary classification models.

    This class computes standard classification metrics and
    automatically handles both predicted class labels and
    predicted probabilities.
    """

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            y_true (np.ndarray):
                Ground-truth binary labels of shape (n_samples,).
                Values must be 0 or 1.

            y_pred (np.ndarray):
                Model outputs. Can be either:
                - Predicted class labels of shape (n_samples,)
                - Predicted probabilities of shape (n_samples,), values in [0, 1]

        Returns:
            Dict[str, float]:
                Dictionary containing computed metrics:
                - accuracy
                - f1
                - precision
                - recall
                - roc_auc (only if probabilities are provided)
        """
        _ = kwargs
        self._validate_inputs(y_true, y_pred)

        y_pred_label: np.ndarray
        y_pred_proba: Optional[np.ndarray]

        if self._is_probability_output(y_pred):
            y_pred_proba = y_pred.astype(float)
            y_pred_label = (y_pred_proba >= 0.5).astype(int)
        else:
            y_pred_label = y_pred.astype(int)
            y_pred_proba = None

        metrics: Dict[str, float] = {
            "accuracy": accuracy_score(y_true, y_pred_label),
            "f1": f1_score(y_true, y_pred_label, average="weighted"),
            "precision": precision_score(
                y_true, y_pred_label, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred_label, average="weighted", zero_division=0
            ),
        }

        if y_pred_proba is not None and self._can_compute_auc(y_true):
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        return metrics

    @staticmethod
    def _is_probability_output(y_pred: np.ndarray) -> bool:
        """
        Check whether predictions represent probabilities.

        Args:
            y_pred (np.ndarray):
                Model predictions.

        Returns:
            bool:
                True if values are in the range [0, 1] and dtype is float.
        """
        if not np.issubdtype(y_pred.dtype, np.floating):
            return False

        return bool(np.all((y_pred >= 0.0) & (y_pred <= 1.0)))

    @staticmethod
    def _can_compute_auc(y_true: np.ndarray) -> bool:
        """
        Determine whether ROC-AUC can be computed.

        ROC-AUC requires at least two distinct classes.

        Args:
            y_true (np.ndarray):
                Ground-truth labels.

        Returns:
            bool:
                True if more than one unique class is present.
        """
        return np.unique(y_true).size > 1

    @staticmethod
    def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Validate input arrays by checking sample length consistency.

        This validation is intentionally minimal and only ensures that
        ``y_true`` and ``y_pred`` contain the same number of samples.
        Shape, dimensionality, and value semantics are validated elsewhere
        if needed.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels.
        y_pred : np.ndarray
            Model predictions.

        Raises
        ------
        ValueError
            If the number of samples in ``y_true`` and ``y_pred`` differs.
        """
        n_true = len(y_true.flatten())
        n_pred = len(y_pred.flatten())

        if n_true != n_pred:
            raise ValueError(
                "Length mismatch between y_true and y_pred: " f"{n_true} != {n_pred}."
            )
