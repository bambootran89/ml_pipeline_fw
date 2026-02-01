from typing import Any, Optional, cast

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator

from mlproject.src.models.base import MLModelWrapper
from mlproject.src.utils.func_utils import flatten_timeseries


class CatBoostWrapper(MLModelWrapper):
    """
    Wrapper for CatBoost regressor/classifier using MLModelWrapper as base.
    """

    def build(self, model_type: str) -> None:
        """
        Initialize the CatBoost model based on configuration.
        """
        args = self.cfg.get("args", {})

        # Determine model type (regression vs classification)
        if len(model_type) == 0:
            model_type = self.cfg.get("type", "regression")

        # CatBoost specific: avoid creating many log files if not specified
        if "allow_writing_files" not in args:
            args["allow_writing_files"] = False

        if model_type == "regression":
            self.model = CatBoostRegressor(**args)
        else:
            self.model = CatBoostClassifier(**args)

        self.model_type = model_type

    def fit(
        self,
        x,
        y,
        sample_weight: Optional[np.ndarray] = None,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Train model with sklearn-style estimator interface."""
        if self.model is None:
            self.build(model_type="")

        self.ensure_built()

        # Flatten time-series input (Batch, Time, Feat) -> (Batch, Time*Feat)
        x_reshaped = flatten_timeseries(x)
        y_reshaped = flatten_timeseries(y)
        fit_params = kwargs.copy()

        # Handle validation set for CatBoost
        if x_val is not None and y_val is not None:
            x_val_reshaped = flatten_timeseries(x_val)
            y_val_reshaped = flatten_timeseries(y_val)
            # CatBoost supports eval_set as a list of tuples
            fit_params["eval_set"] = [(x_val_reshaped, y_val_reshaped)]

            # Default to verbose=False (silent) if not specified to keep logs clean
            if "verbose" not in fit_params:
                fit_params["verbose"] = False

            # Optional: Add early_stopping_rounds if usually used in your project
            # if "early_stopping_rounds" not in fit_params:
            #     fit_params["early_stopping_rounds"] = 50

        # Note: If you have categorical features, you can pass 'cat_features=[indices]'
        # inside **kwargs from the Trainer or Config.
        model = cast(BaseEstimator, self.model)
        model.fit(x_reshaped, y_reshaped, sample_weight=sample_weight, **fit_params)

    def predict(self, x: Any, **kwargs: Any) -> Any:
        """
        Predict using the trained CatBoost model.

        Args:
            x (Any): Input features.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model not built/trained yet.")

        # Ensure input is numpy array
        x_arr = np.asarray(x, dtype=np.float32)

        shape = x_arr.shape
        assert len(shape) <= 3

        # Flatten 3D input (Batch, Time, Feature) -> 2D (Batch, Time*Feature)
        if len(shape) == 3:
            x_arr = x_arr.reshape(-1, shape[1] * shape[2])

        # Handle single sample prediction
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        preds = self.model.predict(x_arr)
        if self.n_targets == 1:
            return preds
        else:
            preds = preds.reshape(len(preds), -1, self.n_targets)
        return np.asarray(preds)
