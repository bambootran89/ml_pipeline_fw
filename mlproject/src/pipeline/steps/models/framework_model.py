"""
Generic model step with flexible data wiring and external config support.

This module provides a unified model step that supports:
- Any model type with configurable input/output routing
- External experiment config override via YAML file
- Inline hyperparameter override
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.steps.core.base import BasePipelineStep
from mlproject.src.pipeline.steps.core.constants import DataTypes
from mlproject.src.pipeline.steps.core.factory import StepFactory
from mlproject.src.pipeline.steps.core.utils import ConfigAccessor, ConfigMerger
from mlproject.src.trainer.factory import TrainerFactory


def _ensure_df(x: Any) -> pd.DataFrame:
    """Convert pandas DataFrame to numpy array if needed."""
    if isinstance(x, np.ndarray):
        return pd.DataFrame(x)
    return x


class FrameworkModelStep(BasePipelineStep):
    """
    Step to train models using internal ModelFactory and TrainerFactory.

    This step is designed for models that strictly follow the project's
    Trainer/Wrapper pattern and supports external configuration files.

    Supports 3 ways to configure model (priority order):
    1. experiment_config: Load full config from external YAML file
    2. model_config: Inline config block in pipeline YAML
    3. model_name + hyperparams: Simple override (backward compatible)
    """

    DEFAULT_INPUTS = {"data": "preprocessed_data"}

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        # Method 1: External config file
        experiment_config: Optional[str] = None,
        # Method 2: Inline config block
        model_config: Optional[Dict[str, Any]] = None,
        # Method 3: Simple override (backward compatible)
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        # Common options
        output_as_feature: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize generic model step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : DictConfig
            Base experiment configuration.
        enabled : bool, default=True
            Whether step should execute.
        depends_on : Optional[List[str]], default=None
            IDs of prerequisite steps.
        experiment_config : Optional[str], default=None
            Path to external YAML file with full experiment config.
            This config will be merged with base config.
        model_config : Optional[Dict[str, Any]], default=None
            Inline config block with model_name, model_type, hyperparams.
        model_name : Optional[str], default=None
            Model name from registry. Falls back to cfg.experiment.model.
        model_type : Optional[str], default=None
            Model type (ml/dl). Falls back to cfg.experiment.model_type.
        hyperparams : Optional[Dict[str, Any]], default=None
            Override hyperparameters for this step.
        output_as_feature : bool, default=False
            If True, store predictions as features for downstream steps.
        **kwargs
            Additional parameters including wiring config.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)

        # Store config sources
        self.experiment_config_path = experiment_config
        self.model_config = model_config or {}
        self.simple_model_name = model_name
        self.simple_model_type = model_type
        self.simple_hyperparams = hyperparams or {}
        self.output_as_feature = output_as_feature
        self.log_artifact: bool = kwargs.get("log_artifact", False)
        self.artifact_type: str = kwargs.get("artifact_type", "component")
        # Build effective config
        self.effective_cfg = self._build_effective_config()

        # Extract final model settings
        self.model_name = self.effective_cfg.experiment.get("model", "").lower()
        self.model_type = self.effective_cfg.experiment.get("model_type", "ml").lower()

    def _build_effective_config(self) -> DictConfig:
        """
        Build effective config by merging sources in priority order.

        Priority (later overrides earlier):
        1. Base cfg (from pipeline)
        2. External experiment_config file
        3. Inline model_config block
        4. Simple model_name/model_type/hyperparams

        Returns
        -------
        DictConfig
            Merged effective configuration.
        """
        # Start with base config
        base_container = OmegaConf.to_container(self.cfg, resolve=True)
        if not isinstance(base_container, dict):
            base_container = {}
        effective = OmegaConf.create(base_container)

        # Method 1: Merge external config file
        if self.experiment_config_path:
            effective = self._merge_external_config(effective)

        # Method 2: Merge inline model_config
        if self.model_config:
            effective = self._merge_model_config(effective, self.model_config)

        # Method 3: Apply simple overrides
        effective = self._apply_simple_overrides(effective)
        return effective

    def _merge_external_config(self, base_cfg: DictConfig) -> DictConfig:
        """
        Merge external experiment config file.

        Parameters
        ----------
        base_cfg : DictConfig
            Base configuration.

        Returns
        -------
        DictConfig
            Merged configuration.

        Raises
        ------
        FileNotFoundError
            If experiment_config file not found.
        """
        assert isinstance(self.experiment_config_path, str)

        print(
            f"[{self.step_id}] Loading experiment config: {self.experiment_config_path}"
        )

        # Use ConfigMerger utility
        search_paths = ["mlproject", "."]
        return cast(
            DictConfig,
            ConfigMerger.merge_external_file(
                base_cfg, self.experiment_config_path, search_paths
            ),
        )

    def _merge_model_config(
        self, base_cfg: DictConfig, model_config: Dict[str, Any]
    ) -> DictConfig:
        """
        Merge inline model_config block.

        Parameters
        ----------
        base_cfg : DictConfig
            Base configuration.
        model_config : Dict[str, Any]
            Inline config with model_name, model_type, hyperparams, etc.

        Returns
        -------
        DictConfig
            Merged configuration.
        """
        # Use ConfigMerger utility
        return cast(
            DictConfig,
            ConfigMerger.merge_model_config(
                base_cfg,
                model_name=model_config.get("model_name"),
                model_type=model_config.get("model_type"),
                hyperparams=model_config.get("hyperparams"),
                data_config=model_config.get("data"),
            ),
        )

    def _apply_simple_overrides(self, base_cfg: DictConfig) -> DictConfig:
        """
        Apply simple model_name/model_type/hyperparams overrides.

        Parameters
        ----------
        base_cfg : DictConfig
            Base configuration.

        Returns
        -------
        DictConfig
            Configuration with simple overrides applied.
        """
        # Use ConfigMerger utility
        return cast(
            DictConfig,
            ConfigMerger.merge_model_config(
                base_cfg,
                model_name=self.simple_model_name,
                model_type=self.simple_model_type,
                hyperparams=self.simple_hyperparams
                if self.simple_hyperparams
                else None,
            ),
        )

    def _get_input_data(self, context: Dict[str, Any]) -> pd.DataFrame:
        """Assemble DataFrame from features and targets."""
        f = self.get_input(context, "features")
        tg = self.get_input(context, "targets", required=False)
        if f is None:
            raise ValueError(f"Step '{self.step_id}': features is None.")

        # Auto-restore column names if input is numpy array (e.g. from sklearn)
        data_cfg_eff = self.effective_cfg.get("data", {})
        feature_names = list(data_cfg_eff.get("features", []))
        target_names = list(data_cfg_eff.get("target_columns", []))

        if isinstance(f, np.ndarray):
            f = pd.DataFrame(f)
            if feature_names and len(feature_names) == f.shape[1]:
                f.columns = feature_names

        if tg is not None and isinstance(tg, np.ndarray):
            tg = pd.DataFrame(tg)
            if target_names and len(target_names) == tg.shape[1]:
                tg.columns = target_names

        config_accessor = ConfigAccessor(self.cfg)

        if config_accessor.is_timeseries():
            return _ensure_df(f).copy()

        if tg is not None:
            tg = _ensure_df(tg)
            f = _ensure_df(f)
            return pd.concat([f, tg], axis=1)

        return _ensure_df(f).copy()

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model training/prediction.

        Uses effective_cfg built from merged config sources.
        """
        self.validate_dependencies(context)

        print(
            f"[{self.step_id}] Executing model: {self.model_name} ({self.model_type})"
        )

        # Log config source
        if self.experiment_config_path:
            print(f"[{self.step_id}] Config source: {self.experiment_config_path}")
        elif self.model_config:
            print(f"[{self.step_id}] Config source: inline model_config")
        else:
            print(f"[{self.step_id}] Config source: simple override")

        # Get and prepare data
        # Get hyperparams from effective config
        hyperparams = dict(self.effective_cfg.experiment.get("hyperparams", {}))

        # Build model
        wrapper = ModelFactory.create(self.model_name, self.effective_cfg)

        # Build datamodule
        datamodule = self.get_input(context, "datamodule", required=False)
        if datamodule is None:
            print("datamodule is None, the we get from data")
            df = self._get_input_data(context)
            if df is None:
                raise ValueError(
                    f"data must be specified if datamodule is {datamodule}"
                )
            datamodule = DataModuleFactory.build(self.effective_cfg, df)
            datamodule.setup()

        # Build trainer
        trainer = TrainerFactory.create(
            model_type=self.model_type,
            model_name=self.model_name,
            wrapper=wrapper,
            save_dir=self.effective_cfg.training.get(
                "artifacts_dir", "artifacts/models"
            ),
        )

        # Train
        print(f"[{self.step_id}] Training with hyperparams: {hyperparams}")
        trained_wrapper = trainer.train(datamodule, hyperparams)
        if self.log_artifact:
            self.register_for_discovery(context, trained_wrapper)

        # Store outputs
        self.set_output(context, "model", trained_wrapper)
        self.set_output(context, "datamodule", datamodule)

        # Generate features if requested
        if self.output_as_feature:
            features = self._generate_features(trained_wrapper, datamodule)
            self.set_output(context, "features", features)
            print(f"[{self.step_id}] Generated features: {features.shape}")

        print(f"[{self.step_id}] Model step completed")
        return context

    def _generate_timeseries_features(
        self, wrapper: Any, datamodule: Any, df: pd.DataFrame
    ) -> np.ndarray:
        """Handle timeseries feature generation with windowing."""
        if not hasattr(datamodule, "_create_windows"):
            x_train, _, _, _, _, _ = datamodule.get_data()
            return wrapper.predict(x_train)

        input_chunk = getattr(datamodule, "input_chunk", 1)
        output_chunk = getattr(datamodule, "output_chunk", 1)
        # pylint: disable=protected-access
        x_all, _ = datamodule._create_windows(df, input_chunk, output_chunk)
        predictions = wrapper.predict(x_all)

        n = len(df)
        if len(predictions) < n:
            start_align = input_chunk - 1
            expected_len = len(predictions)

            if start_align + expected_len <= n:
                full_preds = np.full((n, 1), np.nan)
                if predictions.ndim > 1:
                    full_preds = np.full((n, predictions.shape[1]), np.nan)

                full_preds[
                    start_align : start_align + expected_len
                ] = predictions.reshape(expected_len, -1)

                # Fill NaNs generated by padding
                # bfill for start padding, ffill for end padding
                df_preds = pd.DataFrame(full_preds)
                df_preds = df_preds.bfill().ffill()
                # If still NaN (empty predictions?), fill 0
                df_preds = df_preds.fillna(0.0)
                predictions = df_preds.values

                print(f"[{self.step_id}] Padded {expected_len}->{n} (imp: bfill/ffill)")

        return predictions

    def _generate_features(
        self, wrapper: Any, datamodule: Any
    ) -> pd.DataFrame | np.ndarray:
        """Generate features from trained model."""
        if not hasattr(datamodule, "df") or not hasattr(datamodule, "features"):
            if hasattr(datamodule, "get_data"):
                x_train, _, _, _, _, _ = datamodule.get_data()
                predictions = wrapper.predict(x_train)
            elif hasattr(datamodule, "get_test_windows"):
                x_train, _ = datamodule.get_test_windows()
                predictions = wrapper.predict(x_train)
            else:
                raise AttributeError("DataModule does not support data extraction")

            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            return predictions

        df = datamodule.df
        data_type = getattr(datamodule, "data_type", DataTypes.TABULAR)
        if DataTypes.is_timeseries(data_type):
            predictions = self._generate_timeseries_features(wrapper, datamodule, df)
        else:
            x_all = df[datamodule.features].values
            predictions = wrapper.predict(x_all)

        if len(predictions) == len(df):
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            return pd.DataFrame(predictions, index=df.index)

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions


class ClusteringModelStep(FrameworkModelStep):
    """
    Specialized step for clustering models.

    Extends FrameworkModelStep with clustering-specific defaults.
    """

    def __init__(self, step_id: str, cfg: DictConfig, **kwargs: Any) -> None:
        kwargs.setdefault("output_as_feature", True)
        kwargs.setdefault("model_name", "kmean")
        kwargs.setdefault("model_type", "ml")
        super().__init__(step_id, cfg, **kwargs)


# Register step types
StepFactory.register("framework_model", FrameworkModelStep)
StepFactory.register("clustering", ClusteringModelStep)
