from typing import Any, Optional, cast

import numpy as np
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBRegressor

from mlproject.src.models.base import MLModelWrapper
from mlproject.src.utils.func_utils import flatten_timeseries


class XGBWrapper(MLModelWrapper):
    """
    Wrapper for XGBoost regressor using MLModelWrapper as base.
    """

    def build(self, model_type: str) -> None:
        args = self.cfg.get("args", {})
        if len(model_type) == 0:
            model_type = self.cfg.get("type", "regression")
        if model_type == "regression":
            self.model = XGBRegressor(**args)
        else:
            self.model = XGBClassifier(**args)
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
        """Train model with sklearn-style estimator."""
        if self.model is None:
            self.build(model_type="")  # Dùng giá trị int đã tính toán

        self.ensure_built()
        x_reshaped = flatten_timeseries(x)
        y_reshaped = flatten_timeseries(y)
        # Tạo eval_set nếu có validation data
        fit_params = kwargs.copy()
        if x_val is not None and y_val is not None:
            x_val_reshaped = flatten_timeseries(x_val)
            y_val_reshaped = flatten_timeseries(y_val)
            fit_params["eval_set"] = [(x_val_reshaped, y_val_reshaped)]
            fit_params["verbose"] = False
        model = cast(BaseEstimator, self.model)
        model.fit(x_reshaped, y_reshaped, sample_weight=sample_weight, **fit_params)

    def predict(self, x: Any, **kwargs: Any) -> Any:
        """
        Predict using the trained XGBRegressor.

        Args:
            x (Any): Input features.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model not built/trained yet.")

        x_arr = np.asarray(x, dtype=np.float32)

        shape = x_arr.shape
        assert len(shape) <= 3

        if len(shape) == 3:
            x_arr = x_arr.reshape(-1, shape[1] * shape[2])

        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        preds = self.model.predict(x_arr)
        if self.n_targets == 1:
            return preds
        else:
            preds = preds.reshape(len(preds), -1, self.n_targets)
        return np.asarray(preds)
