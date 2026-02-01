"""
Abstract base pipeline defining a safe,
 high-level, typed workflow for ML/DL experiments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.trainer.factory import TrainerFactory


class BasePipeline(ABC):
    """
    Base class orchestrating preprocessing, component initialization, and experiment
    execution using Hydra/OmegaConf configuration.
    """

    def __init__(
        self,
        cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize pipeline and cache model metadata.
        """
        if cfg is None:
            self.cfg: DictConfig = OmegaConf.create({})
        elif isinstance(cfg, dict):
            self.cfg = OmegaConf.create(cfg)
        elif isinstance(cfg, DictConfig):
            self.cfg = cfg
        else:
            raise TypeError("cfg must be dict or DictConfig")

        self.mlflow_manager = MLflowManager(self.cfg)

        self.exp: Dict[str, Any] = OmegaConf.select(self.cfg, "experiment") or {}
        model = self.exp.get("model")
        mtype = self.exp.get("model_type")
        self.experiment_name = self.cfg.experiment.name

        self.model_name = str(model).lower() if model else "undefined"
        self.model_type = str(mtype).lower() if mtype else "undefined"

        print(f"[Pipeline] Init -> model='{self.model_name}', type='{self.model_type}'")

    def _get_components(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Any, Any, Any]:
        """
        Initialize core components using cached pipeline metadata.
        """
        print(f"[Pipeline] Approach keys: {list(self.exp.keys())}")

        if "model" not in self.exp or "model_type" not in self.exp:
            print("[Pipeline] Missing required keys")
            raise KeyError("approach must contain 'model' and 'model_type'")

        print(
            f"[Pipeline] Build -> model='{self.model_name}', type='{self.model_type}'"
        )

        wrapper = ModelFactory.create(self.model_name, self.cfg)
        print(f"[Pipeline] Wrapper: {type(wrapper).__name__}")

        datamodule = DataModuleFactory.build(self.cfg, df)
        datamodule.setup()
        print(f"[Pipeline] DataModule: {type(datamodule).__name__}")

        trainer = TrainerFactory.create(
            model_type=self.model_type,
            model_name=self.model_name,
            wrapper=wrapper,
            save_dir=self.cfg.training.artifacts_dir,
        )
        print(f"[Pipeline] Trainer: {type(trainer).__name__}")

        return datamodule, wrapper, trainer

    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        """Run preprocessing and return DataFrame."""

    @abstractmethod
    def run_exp(
        self,
        data: pd.DataFrame,
    ) -> Any:
        """Execute a single experiment approach."""

    def run(self, data: Optional[pd.DataFrame] = None) -> None:
        """
        Run pipeline end-to-end.
        """
        if data is None:
            print("[Pipeline] Preprocess()")
            data = self.preprocess()
            print(f"[Pipeline] Shape: {data.shape}")

        print(f"[Pipeline] Run -> '{self.exp.get('name', 'Unnamed')}'")
        print(f"[Pipeline] Use -> model='{self.model_name}', type='{self.model_type}'")

        self.run_exp(data)
