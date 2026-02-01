from copy import deepcopy
from typing import Any, Dict, Optional

import optuna
from omegaconf import DictConfig

# from mlproject.src.datamodule.splitters.timeseries import TimeSeriesFoldSplitter
from mlproject.src.datamodule.splitters.base import BaseSplitter
from mlproject.src.pipeline.compat.v1.cv import CrossValidationPipeline
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.tuning.base import BaseTuner


class OptunaTuner(BaseTuner):
    """
    Hyperparameter tuner using Optuna for time-series forecasting models.

    Supports:
    - Automatic search-space suggestion via cfg
    - MLflow integration for tracking parameters, metrics, and nested runs
    - Nested cross-validation evaluation
    """

    def __init__(
        self,
        cfg: DictConfig,
        splitter: BaseSplitter,
        mlflow_manager: Optional[MLflowManager] = None,
        metric_name: str = "mae_mean",
        direction: str = "minimize",
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
    ):
        """
        Initialize the OptunaTuner.

        Args:
            cfg (DictConfig): Full experiment configuration.
            splitter (TimeSeriesSplitter): CV splitter for time series data.
            mlflow_manager (Optional[MLflowManager]): MLflow manager instance.
            metric_name (str): Metric to optimize. Defaults to "mae_mean".
            direction (str):
                Optimization direction, "minimize"
                or "maximize". Defaults to "minimize".
            model_name (str, optional):
                Override model name from cfg.experiment.model.
            model_type (str, optional):
                Override model type from cfg.experiment.model_type.
        """
        super().__init__(cfg, splitter, mlflow_manager, metric_name, direction)

        # Store model overrides
        self.model_name_override = model_name
        self.model_type_override = model_type

        # CV pipeline used inside objective function
        self.cv_pipeline = CrossValidationPipeline(cfg, splitter)

    def _suggest_params(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a given trial using the registered search space.

        Args:
            trial (optuna.Trial): Optuna trial object.
            model_name (str): Name of the model to tune.

        Returns:
            Dict[str, Any]: Suggested hyperparameters.
        """
        search_space = self.get_search_space(model_name)
        params: Dict[str, Any] = {}

        for name, spec in search_space.items():
            param_type = spec["type"]
            param_range = spec["range"]

            if param_type == "int":
                step = spec.get("step", 1)
                params[name] = trial.suggest_int(
                    name, param_range[0], param_range[1], step=step
                )

            elif param_type == "float":
                log = spec.get("log", False)
                params[name] = trial.suggest_float(
                    name, param_range[0], param_range[1], log=log
                )

            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, param_range)

            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        return params

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function evaluated for each Optuna trial.
        """
        # Use override if provided, otherwise fall back to config
        model_name = (
            self.model_name_override
            if self.model_name_override
            else self.cfg.experiment.model.lower()
        )
        model_type = (
            self.model_type_override
            if self.model_type_override
            else self.cfg.experiment.model_type.lower()
        )
        hyperparams = self._suggest_params(trial, model_name)

        # Include fixed hyperparameters
        fixed = dict(self.cfg.experiment.hyperparams)
        hyperparams = {**hyperparams, **fixed}

        # Update config for this trial
        trial_cfg = deepcopy(self.cfg)
        trial_cfg.experiment.hyperparams.update(hyperparams)
        self.cv_pipeline.cfg = trial_cfg

        run_name = f"Trial_{trial.number:03d}"
        assert self.mlflow_manager is not None
        with self.mlflow_manager.start_run(run_name=run_name, nested=True):
            # Log parameters

            # Preprocess data
            approach = {
                "model": model_name,
                "hyperparams": hyperparams,
                "model_type": model_type,
            }

            # Run cross-validation
            agg_metrics: Dict[str, Any] = self.cv_pipeline.run_cv(
                approach, is_tuning=True
            )

            # Log aggregated metrics
            self.mlflow_manager.log_metadata(params=hyperparams, metrics=agg_metrics)
            # Return averaged metric for Optuna
            value = agg_metrics.get(self.metric_name)
            if value is None:
                raise ValueError(
                    f"don't have this metrics: \
                        {self.metric_name}, we only have {agg_metrics.keys()}"
                )
            return float(value)

    def tune(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run the Optuna hyperparameter optimization.

        Args:
            n_trials (int): Maximum number of trials.
            timeout (Optional[int]): Timeout in seconds. Defaults to None.
            **kwargs: Additional options, e.g., n_jobs, show_progress.

        Returns:
            Dict[str, Any]:
                Dictionary containing best parameters, value, and study object.
        """
        n_jobs = kwargs.get("n_jobs", 1)
        show_progress = kwargs.get("show_progress", True)

        study = optuna.create_study(
            direction=self.direction,
            study_name=f"{self.cfg.experiment.name}_tuning",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress,
        )

        print("\n" + "=" * 60)
        print("  OPTUNA TUNING COMPLETED")
        print("=" * 60)
        print(f"Best {self.metric_name}: {study.best_value:.6f}")
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key:20} = {value}")
        print(f"Total trials: {len(study.trials)}")
        print("=" * 60 + "\n")

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "study": study,
        }
