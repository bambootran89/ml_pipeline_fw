#!/usr/bin/env python
"""
Abstract base class for hyperparameter tuners.

This module defines a unified interface for tuning frameworks such as
Optuna, Ray Tune, and others. Concrete tuner implementations must define
the optimization objective and the main tuning loop.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from omegaconf import DictConfig

# from mlproject.src.datamodule.splitters.timeseries import TimeSeriesFoldSplitter
from mlproject.src.datamodule.splitters.base import BaseSplitter
from mlproject.src.tracking.mlflow_manager import MLflowManager

# Forward reference for static type checking only
if TYPE_CHECKING:
    pass


class BaseTuner(ABC):
    """
    Base abstract tuner interface.

    Subclasses must implement:
    - objective(): the optimization function executed per trial
    - tune(): the main hyperparameter search loop
    """

    def __init__(
        self,
        cfg: DictConfig,
        splitter: BaseSplitter,
        mlflow_manager: Optional[MLflowManager] = None,
        metric_name: str = "mae_mean",
        direction: str = "minimize",
    ):
        """
        Initialize the tuner.

        Args:
            cfg: Experiment configuration object.
            splitter: Time-series cross-validation splitter instance.
            mlflow_manager: Optional MLflow tracking helper.
            metric_name: Name of the metric to optimize.
            direction: Optimization direction, either "minimize" or
                "maximize".
        """
        self.cfg = cfg
        self.splitter = splitter
        self.mlflow_manager = mlflow_manager
        self.metric_name = metric_name
        self.direction = direction

        if direction not in ["minimize", "maximize"]:
            raise ValueError(f"Invalid direction: {direction}")

    @abstractmethod
    def objective(self, trial: Any) -> float:
        """
        Define the objective function executed for each trial.

        Args:
            trial: Backend-specific trial object (Optuna trial, Ray config,
                or similar).

        Returns:
            float: Metric value to be optimized.
        """
        raise NotImplementedError

    @abstractmethod
    def tune(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the main hyperparameter search loop.

        Args:
            n_trials: Maximum number of optimization trials.
            timeout: Optional timeout in seconds for the tuning run.
            **kwargs: Additional backend-specific tuning arguments.

        Returns:
            dict: A dictionary containing tuning results:
                - best_params: Best hyperparameters found.
                - best_value: Best metric value achieved.
                - study/tuner: Backend-specific study/tuner object.
        """
        raise NotImplementedError

    def get_search_space(self, model_name: str) -> Dict[str, Any]:
        """
        Retrieve the search space definition for the given model.

        The method first checks whether the YAML configuration provides
        a search space. If not, it falls back to the default registry.

        Args:
            model_name: Name of the target model (e.g., "xgboost",
                "nlinear").

        Returns:
            dict: Search-space definition for the model.
        """
        # YAML-defined search space takes priority
        search_space = (
            self.cfg.get("tuning", {}).get("search_space", {}).get(model_name, {})
        )

        if search_space:
            return dict(search_space)
        raise ValueError(
            f"No hyperparameter search_space defined for model '{model_name}'. "
            "Please configure it under tuning.search_space in your YAML file."
        )
