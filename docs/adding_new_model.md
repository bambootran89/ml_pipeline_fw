# Adding new Model Guide (Example: CatBoost)

This section demonstrates how to integrate a new model implementation into the project using CatBoost as an example.
The same pattern applies to any other algorithm (LightGBM, TabNet, custom Torch models, etc.).

## Step 1: Implement a Model Wrapper

Create a new wrapper that conforms to the project's unified MLModelWrapper interface.

File: mlproject/src/models/catboost_wrapper.py

```python
from typing import Any, Optional, cast

import numpy as np
from sklearn.base import BaseEstimator
from catboost import CatBoostClassifier, CatBoostRegressor

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

        fit_params = kwargs.copy()

        # Handle validation set for CatBoost
        if x_val is not None and y_val is not None:
            x_val_reshaped = flatten_timeseries(x_val)
            # CatBoost supports eval_set as a list of tuples
            fit_params["eval_set"] = [(x_val_reshaped, y_val)]

            # Default to verbose=False (silent) if not specified to keep logs clean
            if "verbose" not in fit_params:
                fit_params["verbose"] = False

            # Optional: Add early_stopping_rounds if usually used in your project
            # if "early_stopping_rounds" not in fit_params:
            #     fit_params["early_stopping_rounds"] = 50

        # Note: If you have categorical features, you can pass 'cat_features=[indices]'
        # inside **kwargs from the Trainer or Config.
        model = cast(BaseEstimator, self.model)
        model.fit(x_reshaped, y, sample_weight=sample_weight, **fit_params)

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
        return np.asarray(preds)

```

## Step 2: Register the Model in model.yaml

Tell the system how to locate and instantiate the new model.
File: mlproject/configs/base/model.yaml

```yaml
catboost:
    module: "mlproject.src.models.catboost_wrapper"
    class: "CatBoostWrapper"
    datamodule_type: "ml"
```

## Step 3: Create an Experiment Configuration

Define a new experiment that uses CatBoost.
File: mlproject/configs/experiments/exp1.yaml
```bash
defaults:
  - ../base/preprocessing.yaml
  - ../base/model.yaml
  - ../base/training.yaml
  - ../base/evaluation.yaml
  - ../base/mlflow.yaml
  - ../base/tuning.yaml

data:
  path: "mlproject/data/ETTh1.csv"   # optional demo
  index_col: "date"
  target_columns: ["HUFL","MUFL",]
  features: ["HUFL", "MUFL", "mobility_inflow"]
  return_type: pandas
  type: timeseries

experiment:
  name: "catboost_tuning"
  type: "timeseries"
  model: "catboost"
  model_type: "ml"
  hyperparams:
    # Initial guess (will be overridden by tuning)
    input_chunk_length: 24
    output_chunk_length: 6
    args:
      iterations: 1000
      learning_rate: 0.05
      depth: 6
      loss_function: "MultiRMSE"
      task_type: "CPU"
      early_stopping_rounds: 20
      verbose: 100
      allow_writing_files: False

# Override tuning settings
tuning:
  n_trials: 30
  n_splits: 3
  test_size: 24
  optimize_metric: "mae_mean"
  direction: "minimize"
  n_jobs: 1

```

## Step 4 (Optional): Enable Hyperparameter Tuning

To allow Optuna-based hyperparameter tuning, extend the global tuning configuration.

File: mlproject/configs/base/tuning.yaml
```yaml
catboost:
  iterations:
    type: "int"
    range: [100, 1000]
    step: 100
  learning_rate:
    type: "float"
    range: [0.001, 0.1]
    log: true
  depth:
    type: "int"
    range: [4, 10]
    step: 1
  l2_leaf_reg:
    type: "float"
    range: [1, 10]
    log: false
```
