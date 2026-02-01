"""
Evaluation pipeline.

Responsibilities:
- Load prediction model from MLflow Model Registry
- Load companion preprocessing model (if available)
- Preprocess test dataset
- Run evaluation and log metrics to MLflow
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.datamodule.loader import resolve_datasets_from_cfg
from mlproject.src.eval.base import BaseEvaluator
from mlproject.src.eval.classification import ClassificationEvaluator
from mlproject.src.eval.clustering import ClusteringEvaluator
from mlproject.src.eval.regression import RegressionEvaluator
from mlproject.src.eval.timeseries import TimeSeriesEvaluator
from mlproject.src.pipeline.compat.v1.base import BasePipeline
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.utils.config_class import ConfigLoader
from mlproject.src.utils.func_utils import flatten_metrics_for_mlflow


class EvalPipeline(BasePipeline):
    """
    Evaluation pipeline for models stored in MLflow Model Registry.
    """

    def __init__(self, cfg_path: str = "", alias: str = "lastest") -> None:
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

        self.model: Any = None
        self.preprocessor_model: Optional[Any] = None

        self.preprocessor = OfflinePreprocessor(is_train=False, cfg=self.cfg)
        self.evaluator: BaseEvaluator = self._build_evaluator()

        if self.mlflow_manager.enabled:
            # Load artifacts đồng nhất
            self.preprocessor_model = self.mlflow_manager.load_component(
                name=f"{self.experiment_name}_preprocessor", alias=alias
            )
            self.model = self.mlflow_manager.load_component(
                name=f"{self.experiment_name}_model", alias=alias
            )

    def preprocess(self) -> pd.DataFrame:
        """
        Load raw data and apply preprocessing.

        Priority:
        1. MLflow PyFunc preprocessing model
        2. Local OfflinePreprocessor fallback

        Returns
        -------
        pd.DataFrame
            Preprocessed dataset.
        """
        df, _, _, df_raw = resolve_datasets_from_cfg(self.cfg)
        is_use_dataset = True
        if len(df) > 0:
            df_raw = df.copy()
            is_use_dataset = False

        fea_df: pd.DataFrame = self._transform_data(df_raw)
        df = self._attach_targets_if_needed(df_raw, fea_df)
        if is_use_dataset:
            df["dataset"] = "test"

        return df

    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformation.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data.

        Returns
        -------
        pd.DataFrame
            Transformed features.
        """

        if self.preprocessor_model is not None:
            print("[EvalPipeline] Using MLflow preprocessing model")
            return self.preprocessor_model.transform(df)
        else:
            print("[EvalPipeline] Using local preprocessing fallback")
            self.preprocessor.transform_manager.load(self.cfg)
            return self.preprocessor.transform(df)

    def _attach_targets_if_needed(
        self, df_raw: pd.DataFrame, fea_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Attach target columns back for tabular datasets.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw input data.
        fea_df : pd.DataFrame
            Transformed features.

        Returns
        -------
        pd.DataFrame
            Final dataset for evaluation.
        """
        data_cfg = self.cfg.get("data", {})
        data_type = str(data_cfg.get("type", "timeseries")).lower()

        if data_type == "timeseries":
            return fea_df

        target_cols = data_cfg.get("target_columns", [])
        df = fea_df.copy()
        for col in target_cols:
            df[col] = df_raw[col]
        return df

    def _build_evaluator(self) -> BaseEvaluator:
        """
        Build evaluator based on configuration.

        Returns
        -------
        BaseEvaluator
            Evaluator instance.
        """
        eval_type = self.cfg.get("evaluation", {}).get("type", "regression")

        if eval_type == "classification":
            return ClassificationEvaluator()
        elif eval_type == "regression":
            return RegressionEvaluator()

        elif eval_type == "clustering":
            return ClusteringEvaluator()
        elif eval_type == "timeseries":
            return TimeSeriesEvaluator()
        else:
            raise ValueError(f"Don't support this type {eval_type}")

    def run_exp(self, data: pd.DataFrame) -> dict:
        """
        Evaluate model on test dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Preprocessed dataset.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        dm = DataModuleFactory.build(self.cfg, data)
        dm.setup()

        if hasattr(dm, "get_test_windows"):
            x_test, y_test = dm.get_test_windows()
        else:
            _, _, _, _, x_test, y_test = dm.get_data()

        x_test = np.asarray(x_test, dtype=np.float32)

        run_name = f"eval_{self.experiment_name}_latest"

        with self.mlflow_manager.start_run(run_name=run_name):
            preds = self.model.predict(x_test)
            metrics = self.evaluator.evaluate(y_test, preds)
            safe_metrics = flatten_metrics_for_mlflow(metrics)
            self.mlflow_manager.log_metadata(metrics=safe_metrics)
        return metrics
