from typing import Any, Dict, Optional

from omegaconf import DictConfig

from mlproject.src.eval.base import BaseEvaluator
from mlproject.src.eval.classification import ClassificationEvaluator
from mlproject.src.eval.clustering import ClusteringEvaluator
from mlproject.src.eval.regression import RegressionEvaluator
from mlproject.src.eval.timeseries import TimeSeriesEvaluator
from mlproject.src.pipeline.compat.v1.base import BasePipeline
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.preprocess.transform_manager import TransformManager
from mlproject.src.utils.config_class import ConfigLoader


class TrainingPipeline(BasePipeline):
    """
    End-to-end training pipeline supporting deep learning (DL) and
    traditional machine learning (ML) forecasting models.

    Responsibilities:
    - Preprocess dataset and save artifacts.
    - Initialize models and trainers.
    - Train models and evaluate on test sets.
    - Log metrics and models to MLflow when enabled.
    """

    def __init__(self, cfg_path: str = ""):
        """Initialize training pipeline and load configuration."""
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

        self.preprocessor = OfflinePreprocessor(is_train=True, cfg=self.cfg)
        self.evaluator: BaseEvaluator
        eval_type = self.cfg.get("evaluation", {}).get("type", "regression")
        if eval_type == "classification":
            self.evaluator = ClassificationEvaluator()
        elif eval_type == "regression":
            self.evaluator = RegressionEvaluator()
        elif eval_type == "timeseries":
            self.evaluator = TimeSeriesEvaluator()
        elif eval_type == "clustering":
            self.evaluator = ClusteringEvaluator()
        else:
            raise ValueError(f"don't support this type {eval_type}")

    def preprocess(self) -> Any:
        """
        Fit preprocessing pipeline and transform dataset.

        Returns:
            Preprocessed dataset object.
        """
        df = self.preprocessor.fit_and_transform()
        return df

    def _execute_training(self, trainer, dm, hyperparams: Dict[str, Any]):
        """
        Execute training and evaluation, independent of MLflow.

        Returns:
            Evaluation metrics dictionary.
        """

        wrapper = trainer.train(dm, hyperparams)
        if hasattr(dm, "get_test_windows"):
            x_test, y_test = dm.get_test_windows()  # TSDLDataModule
        elif hasattr(dm, "get_data"):
            _, _, _, _, x_test, y_test = dm.get_data()  # TSMLDataModule
        else:
            raise AttributeError("DataModule must support retrieving test data")
        preds = wrapper.predict(x_test)
        metrics = self.evaluator.evaluate(
            y_test, preds, x=x_test, model=wrapper.get_model()
        )
        return metrics

    def run_exp(self, data: Any) -> Dict[str, float]:
        """
        Execute a full training approach with optional MLflow logging.

        This method acts as a high-level orchestrator. It is responsible for:
        - Initializing model wrapper and data module
        - Creating the trainer
        - Delegating execution either to:
            * an MLflow-enabled training flow, or
            * a plain training flow without tracking

        Heavy logic (training, MLflow logging) is delegated to helper methods
        to keep this function simple and pylint-compliant.

        Args:
            approach (Dict[str, Any]):
                Dictionary describing the training approach.
                Expected keys include:
                - "model": model name
                - "hyperparams": optional hyperparameter dictionary
            data (Any):
                Raw dataset used to build the DataModule.

        Returns:
            Dict[str, float]:
                Dictionary of evaluation metrics produced by the training run.
        """
        df = data
        hyperparams: Dict[str, Any] = self.exp.get("hyperparams", {})
        dm, wrapper, trainer = self._get_components(df)
        dm.setup()
        if self.mlflow_manager:
            run_name: str = f"{self.experiment_name}_run"

            with self.mlflow_manager.start_run(run_name=run_name):
                transform_manager: Optional[
                    TransformManager
                ] = self.preprocessor.transform_manager

                # Log Preprocessor (với interface thống nhất)
                self.mlflow_manager.log_component(
                    obj=transform_manager,
                    name=f"{self.experiment_name}_preprocessor",
                    artifact_type="preprocess",
                )

                metrics: Dict[str, float] = self._execute_training(
                    trainer, dm, hyperparams
                )
                # Log Model
                self.mlflow_manager.log_component(
                    obj=wrapper,
                    name=f"{self.experiment_name}_model",
                    artifact_type="model",
                )

                self.mlflow_manager.log_metadata(params=hyperparams, metrics=metrics)

                return metrics

        return self._execute_training(trainer, dm, hyperparams)
