"""
Cross-validation orchestrator for time-series experiments.

This class handles orchestration only. Heavy computation is delegated
to FoldRunner to keep functions short and testable.
"""

# import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.datamodule.splitters.base import BaseSplitter
from mlproject.src.pipeline.compat.v1.base import BasePipeline
from mlproject.src.tuning.fold_runner import FoldRunner

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# os.environ["OMP_NUM_THREADS"] = "1"


class CrossValidationPipeline(BasePipeline):
    """Orchestrate cross-validation using raw data windows and fold runners."""

    def __init__(
        self,
        cfg: DictConfig,
        splitter: BaseSplitter,
    ) -> None:
        """Initialize the cross-validation pipeline.

        Args:
            cfg: Hydra/OmegaConf configuration object.
            splitter: TimeSeriesFoldSplitter instance defining CV folds.
            mlflow_manager: Object implementing the MLflowManager interface.
        """
        super().__init__(cfg)
        self.splitter = splitter
        self._fold_runner = FoldRunner(cfg, self.mlflow_manager)

    def preprocess(self) -> pd.DataFrame:
        """No-op preprocessing for CV pipeline.

        Returns:
            Always None for API compatibility.
        """
        return None

    def run_exp(self, data: Any) -> Any:
        """Not supported for cross-validation pipeline.

        Raises:
            NotImplementedError: Always raised for this class.
        """
        raise NotImplementedError("Use run_cv() for cross-validation")

    def run_cv(
        self,
        data: Any = None,
        is_tuning: bool = False,
    ) -> Dict[str, float]:
        """Execute cross-validation using raw input windows.

        Args:
            approach: Dictionary containing model name and hyperparameters.
            data: Raw data (ignored).
            is_tuning: Whether this run is part of hyperparameter tuning.

        Returns:
            Aggregated cross-validation metrics.
        """
        _ = data

        # Generate folds from splitter
        folds = self.splitter.generate_folds()
        model_name: str = self.exp["model"].lower()
        hyperparams: Dict[str, Any] = self.exp.get("hyperparams", {})
        model_type: str = self.exp["model_type"].lower()
        fold_metrics: List[Dict[str, float]] = []
        for i, df_fold in enumerate(folds):
            metrics = self._fold_runner.run_fold(
                df_fold=df_fold,
                model_type=model_type,
                model_name=model_name,
                hyperparams=hyperparams,
                is_tuning=is_tuning,
                fold_index=i,
            )
            fold_metrics.append(metrics)

        return self._aggregate_fold_metrics(fold_metrics)

    def _aggregate_fold_metrics(
        self,
        fold_metrics: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """Compute aggregated metrics over folds.

        Args:
            fold_metrics: List of metric dictionaries, one per fold.

        Returns:
            Dictionary mapping metric names to their mean and std values.
        """
        if not fold_metrics:
            return {}

        aggregated: Dict[str, float] = {}
        first = fold_metrics[0]

        for key in first:
            values = [m[key] for m in fold_metrics]
            aggregated[f"{key}_mean"] = float(sum(values) / len(values))
            aggregated[f"{key}_std"] = float(np.std(values))
        return aggregated
