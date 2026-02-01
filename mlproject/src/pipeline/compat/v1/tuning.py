#!/usr/bin/env python
"""
Tuning pipeline orchestrating hyperparameter search and final model retraining.

Workflow overview:
    1. Run Optuna-based hyperparameter tuning using time-series CV.
    2. Extract the best hyperparameters from the tuning study.
    3. Retrain the model using the full dataset with the best parameters.
    4. Register the retrained model into MLflow Model Registry.
"""


from omegaconf import DictConfig

from mlproject.src.datamodule.splitters.base import BaseSplitter
from mlproject.src.datamodule.splitters.timeseries import TimeSeriesFoldSplitter
from mlproject.src.pipeline.compat.v1.base import BasePipeline
from mlproject.src.pipeline.compat.v1.training import TrainingPipeline
from mlproject.src.tuning.optuna import OptunaTuner
from mlproject.src.utils.config_class import ConfigLoader


class TuningPipeline(BasePipeline):
    """
    End-to-end tuning workflow combining:
        - Hyperparameter optimization
        - Best-parameter extraction
        - Final model retraining
        - Model registration in MLflow
    """

    def __init__(self, cfg_path: str = ""):
        """
        Initialize tuning pipeline.

        Args:
            cfg_path:
                Path to the experiment configuration YAML file.
        """
        self.cfg_path = cfg_path
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

        # Build CV splitter. Parameters may be overridden in YAML config.
        self.splitter: BaseSplitter
        eval_type = self.cfg.get("data", {}).get("type", "timeseries")
        if eval_type == "timeseries":
            self.splitter = TimeSeriesFoldSplitter(
                self.cfg,
                n_splits=self.cfg.get("tuning", {}).get("n_splits", 3),
            )
        else:
            self.splitter = BaseSplitter(
                self.cfg,
                n_splits=self.cfg.get("tuning", {}).get("n_splits", 3),
            )

    def preprocess(self):
        """
        Load and preprocess the dataset for tuning and retraining.

        Returns:
            Processed dataset produced by offline preprocessing workflow.
        """

    def run_exp(self, data):
        """
        Not used for tuning workflow.

        Raises:
            NotImplementedError:
                Always raised because tuning uses `run()`.
        """
        raise NotImplementedError("Use run() for tuning workflow.")

    def run(self, data=None):
        # START PARENT RUN HERE
        # Run name: Hparam_Tuning_Experiment
        with self.mlflow_manager.start_run(run_name="Hparam_Tuning_Experiment"):
            print(
                "[MLflow] Started Parent Run: \
                  Hparam_Tuning_Experiment"
            )

            # Step 2: Initialize the tuner
            tuner = OptunaTuner(
                cfg=self.cfg,
                splitter=self.splitter,
                mlflow_manager=self.mlflow_manager,  # Pass manager to tuner
                metric_name=self.cfg.get("tuning", {}).get(
                    "optimize_metric", "mae_mean"
                ),
                direction="minimize",
            )

            # Step 3: Run tuning (Trials will be nested children)
            n_trials = self.cfg.get("tuning", {}).get("n_trials", 20)

            # Pass aggregated metrics of best trial to parent run (optional but good)
            result = tuner.tune(n_trials=n_trials)
            best_params = result["best_params"]

            # Log best params to Parent Run for quick view
            self.mlflow_manager.log_metadata(params=best_params)

        # Step 4: Update experiment hyperparameters
        self.cfg.experiment.hyperparams.update(best_params)

        # Step 5: Retrain model using best parameters
        # This is OUTSIDE the tuning run, so it will create its own standalone run
        # (or you can nest it if you prefer, but usually Retrain is a separate run)
        print(f"\n{'=' * 60}")
        print("  RETRAINING WITH BEST HYPERPARAMETERS")
        print(f"{'=' * 60}\n")

        training_pipeline = TrainingPipeline(self.cfg_path)
        training_pipeline.cfg.experiment.hyperparams = self.cfg.experiment.hyperparams

        # This will log the FINAL MODEL (artifacts enabled by default in
        # TrainingPipeline)
        training_pipeline.run(data)

        return best_params
