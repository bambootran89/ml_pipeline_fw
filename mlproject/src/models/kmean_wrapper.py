from typing import Any, Optional

import numpy as np
from sklearn.cluster import KMeans

from mlproject.src.models.base import MLModelWrapper
from mlproject.src.utils.func_utils import flatten_timeseries


class KMeansWrapper(MLModelWrapper):
    """
    Wrapper class for scikit-learn KMeans clustering.

    This wrapper adapts `sklearn.cluster.KMeans` to the unified
    `MLModelWrapper` interface used across the project.

    Notes
    -----
    - KMeans is an unsupervised algorithm; the `y` argument is ignored.
    - Time-series inputs with more than 2 dimensions are flattened
      automatically before fitting or prediction.
    """

    def build(self, model_type: str) -> Any:
        """
        Build and initialize the underlying KMeans model.

        Model hyperparameters are read from `self.model_cfg`.

        Supported configuration keys:
        - n_clusters (int, default=8)
        - init (str, default="k-means++")
        - random_state (int, default=42)
        - max_iter (int, default=300)
        - tol (float, default=1e-4)

        Returns:
            Any:
                Initialized `sklearn.cluster.KMeans` instance.
        """
        args = self.cfg.get("args", {})
        n_clusters: int = args.get("n_clusters", 8)
        init_method: str = args.get("init", "k-means++")
        random_state: int = args.get("random_state", 42)
        max_iter: int = args.get("max_iter", 300)
        tol: float = args.get("tol", 1e-4)
        self.model_type = model_type
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init_method,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_init="auto",  # Recommended for recent scikit-learn versions
        )

    def fit(
        self,
        x,
        y,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Any:
        """
        Fit the KMeans model on input data.

        Parameters:
            X (np.ndarray):
                Training data of shape (n_samples, n_features) or
                higher-dimensional time-series data.
            y (Optional[np.ndarray], optional):
                Ignored. Present for API compatibility.
            **kwargs:
                Additional keyword arguments (ignored).

        Returns:
            KMeansWrapper:
                The fitted model wrapper.
        """
        del y, sample_weight, kwargs
        if self.model is None:
            self.build(model_type="")
        self.ensure_built()
        if x.ndim > 2:
            x = flatten_timeseries(x)
        if self.model is None:
            raise RuntimeError("Model not built/trained yet.")
        self.model.fit(x.astype(np.float32))

    def predict(self, x: Any, **kwargs: Any) -> Any:
        """
        Predict cluster indices for new data.

        Parameters:
            X (np.ndarray):
                Input data of shape (n_samples, n_features) or
                higher-dimensional time-series data.

        Returns:
            np.ndarray:
                Array of cluster labels for each sample.
        """
        if self.model is None:
            raise RuntimeError("Model not built/trained yet.")
        if x.ndim > 2:
            x = flatten_timeseries(x)
        return self.model.predict(x.astype(np.float32))
