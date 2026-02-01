from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from mlproject.src.eval.base import BaseEvaluator


class ClusteringEvaluator(BaseEvaluator):
    """
    Evaluator for clustering models.

    This evaluator computes intrinsic clustering metrics that do not
    require ground-truth labels and provides human-readable explanations
    for each metric.
    """

    _METRIC_EXPLANATIONS: Dict[str, str] = {
        "inertia": (
            "Within-cluster sum of squares. Measures cluster compactness. "
            "Lower values indicate tighter clusters."
        ),
        "silhouette_score": (
            "Mean silhouette coefficient over all samples, ranging from -1 to 1. "
            "Higher values indicate better cluster separation."
        ),
        "calinski_harabasz": (
            "Ratio of between-cluster dispersion to within-cluster dispersion. "
            "Higher values indicate better defined clusters."
        ),
        "davies_bouldin": (
            "Average similarity between each cluster and its most similar cluster. "
            "Lower values indicate better clustering."
        ),
        "n_clusters": ("Total number of clusters identified in the clustering result."),
        "min_cluster_size": ("Number of samples in the smallest cluster."),
        "max_cluster_size": ("Number of samples in the largest cluster."),
        "std_cluster_size": (
            "Standard deviation of cluster sizes, indicating size imbalance."
        ),
    }

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute clustering metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Unused. Present for interface compatibility.
        y_pred : np.ndarray
            Cluster labels.
        **kwargs
            Optional extra information:
            - x: feature matrix of shape (n_samples, n_features)
            - model: fitted clustering model (to extract inertia)

        Returns
        -------
        Dict[str, float]
            Dictionary of clustering metric values.
        """
        _ = y_true
        x = kwargs.get("x")
        labels = np.asarray(y_pred)

        metrics: Dict[str, float] = {}

        model = kwargs.get("model")
        if model is not None and hasattr(model, "inertia_"):
            metrics["inertia"] = float(model.inertia_)

        n_clusters = np.unique(labels).size

        if x is not None and x.ndim == 2 and n_clusters > 1:
            metrics["silhouette_score"] = float(silhouette_score(x, labels))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(x, labels))
            metrics["davies_bouldin"] = float(davies_bouldin_score(x, labels))

        cluster_sizes = np.bincount(labels.astype(int))
        if cluster_sizes.size > 0:
            metrics["n_clusters"] = float(cluster_sizes.size)
            metrics["min_cluster_size"] = float(cluster_sizes.min())
            metrics["max_cluster_size"] = float(cluster_sizes.max())
            metrics["std_cluster_size"] = float(cluster_sizes.std())

        return metrics

    def explain(self, metrics: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Attach explanations to clustering metrics.

        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary of computed clustering metrics.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping metric names to their values and explanations.
        """
        explained: Dict[str, Dict[str, Any]] = {}

        for name, value in metrics.items():
            explained[name] = {
                "value": value,
                "explanation": self._METRIC_EXPLANATIONS.get(
                    name, "No explanation available."
                ),
            }

        return explained
