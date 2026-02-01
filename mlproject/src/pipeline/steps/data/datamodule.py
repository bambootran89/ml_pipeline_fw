"""Enhanced DataModule step with multi-source feature support."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.pipeline.steps.core.base import BasePipelineStep
from mlproject.src.pipeline.steps.core.constants import ContextKeys
from mlproject.src.pipeline.steps.core.factory import StepFactory
from mlproject.src.pipeline.steps.core.utils import ConfigAccessor, SampleAligner


class DataModuleStep(BasePipelineStep):
    """Build DataModule with optional multi-source feature composition.

    Context Inputs
    --------------
    features : pd.DataFrame
        Base features from preprocessing.
    targets : pd.DataFrame
        Target labels.
    additional_feature_keys : List[str], optional
        Additional feature sources to compose.

    Context Outputs
    ---------------
    datamodule : Any
        Constructed DataModule instance.
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        output_as_feature: bool = False,
        additional_feature_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DataModule step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : Any
            Configuration object.
        enabled : bool
            Whether step should execute.
        depends_on : Optional[List[str]]
            Prerequisite steps.
        output_as_feature : bool
            Whether to generate features from model predictions.
        additional_feature_keys : Optional[List[str]]
            Keys for additional feature sources to compose.
        **kwargs
            Additional parameters.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.output_as_feature = output_as_feature
        self.additional_feature_keys = additional_feature_keys or []

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DataModule construction with feature composition.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Updated context with DataModule.
        """
        self.validate_dependencies(context)

        # Get base features and targets
        df_features = self.get_input(context, "features")
        df_targets = self.get_input(context, "targets", required=False)

        # Track composed feature names for config injection
        composed_feature_names: List[str] = []

        # Compose with additional features if specified
        if self.additional_feature_keys:
            df_features, composed_feature_names = self._compose_features_with_names(
                context, df_features
            )

        # Build input DataFrame
        config_accessor = ConfigAccessor(self.cfg)

        if config_accessor.is_timeseries():
            input_df = df_features.copy()
        elif df_targets is not None:
            input_df = pd.concat([df_features, df_targets], axis=1)
        else:
            input_df = df_features.copy()

        # Inject composed feature names into config if we have additional features
        cfg_for_dm = self._inject_composed_features_to_config(
            composed_feature_names, context
        )

        # Build DataModule
        print(f"[{self.step_id}] Building DataModule with shape {input_df.shape}")
        dm = DataModuleFactory.build(cfg_for_dm, input_df)
        dm.setup()

        self.set_output(context, "datamodule", dm)

        # Optional: Generate features from model
        if self.output_as_feature:
            self._generate_model_features(context, dm)

        return context

    def _compose_features(
        self,
        context: Dict[str, Any],
        base_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compose base features with additional sources.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.
        base_features : pd.DataFrame
            Base feature DataFrame.

        Returns
        -------
        pd.DataFrame
            Composed features.
        """
        composed, _ = self._compose_features_with_names(context, base_features)
        return composed

    def _compose_features_with_names(
        self,
        context: Dict[str, Any],
        base_features: pd.DataFrame,
    ) -> tuple[pd.DataFrame, List[str]]:
        """Compose base features with additional sources and return feature names.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.
        base_features : pd.DataFrame
            Base feature DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, List[str]]
            Composed features DataFrame and list of all feature column names.
        """
        if not isinstance(base_features, pd.DataFrame):
            base_features = pd.DataFrame(base_features)

        composed = base_features.copy()
        len(composed)

        print(f"[{self.step_id}] Composing features:")
        print(f"  Base: {composed.shape}")

        for key in self.additional_feature_keys:
            additional = context.get(key)
            if additional is None:
                print(f"  Warning: Feature key '{key}' not found, skipping")
                continue

            # Convert to DataFrame
            if isinstance(additional, np.ndarray):
                additional = self._ndarray_to_df(additional, key)
            elif not isinstance(additional, pd.DataFrame):
                additional = pd.DataFrame(additional)

            # Align samples using SampleAligner utility
            _, additional_aligned = SampleAligner.align_samples(
                base_data=composed, additional_data=additional, method="auto"
            )
            # Convert back to DataFrame if needed
            if isinstance(additional, pd.DataFrame):
                additional = pd.DataFrame(
                    additional_aligned,
                    columns=additional.columns,
                    index=additional.index
                    if len(additional) == len(additional_aligned)
                    else None,
                )
            else:
                additional = pd.DataFrame(additional_aligned)

            # Force index alignment if lengths match
            # This handles cases where additional features lost their index (e.g. from
            # numpy)
            if len(additional) == len(composed):
                additional.index = composed.index

            # Prefix columns to avoid collisions
            if isinstance(additional, pd.DataFrame):
                additional.columns = [f"{key}_{c}" for c in additional.columns]

            # Concatenate
            composed = pd.concat([composed, additional], axis=1)
            print(f"  + {key}: {additional.shape} -> Total: {composed.shape}")

        # Return composed DataFrame and all feature column names
        composed_feature_names = list(composed.columns)
        return composed, composed_feature_names

    def _inject_composed_features_to_config(
        self,
        composed_feature_names: List[str],
        context: Dict[str, Any],
    ) -> DictConfig:
        """Inject composed feature names into config for DataModule.

        This ensures BaseDataModule uses all composed features (base + additional)
        without requiring changes to experiment yaml.

        Parameters
        ----------
        composed_feature_names : List[str]
            List of all feature column names after composition.
        context : Dict[str, Any]
            Pipeline context (for storing metadata).

        Returns
        -------
        DictConfig
            Modified config with injected feature names.
        """
        # Deep copy config to avoid mutating original
        if isinstance(self.cfg, DictConfig):
            cfg_copy = OmegaConf.to_container(self.cfg, resolve=True)
            cfg_copy = OmegaConf.create(cfg_copy)
        else:
            cfg_copy = OmegaConf.create(copy.deepcopy(dict(self.cfg)))

        # Only inject if we have composed features from additional_feature_keys
        if composed_feature_names and self.additional_feature_keys:
            original_features = OmegaConf.select(cfg_copy, "data.features", default=[])

            # Update data.features with composed feature names
            OmegaConf.update(cfg_copy, "data.features", composed_feature_names)

            # Also update n_features in hyperparams if present
            if (
                OmegaConf.select(cfg_copy, "experiment.hyperparams.n_features")
                is not None
            ):
                OmegaConf.update(
                    cfg_copy,
                    "experiment.hyperparams.n_features",
                    len(composed_feature_names),
                )

            print(
                f"[{self.step_id}] Injected composed features into config: "
                f"{len(original_features)} -> {len(composed_feature_names)} features"
            )

            # Store composed feature metadata in context for downstream steps
            # (serve, eval, tune can use this)
            context[ContextKeys.COMPOSED_FEATURE_NAMES] = composed_feature_names
            context[ContextKeys.ADDITIONAL_FEATURE_KEYS] = self.additional_feature_keys

        return cast(DictConfig, cfg_copy)

    def _ndarray_to_df(self, arr: np.ndarray, prefix: str) -> pd.DataFrame:
        """Convert numpy array to DataFrame.

        Parameters
        ----------
        arr : np.ndarray
            Input array.
        prefix : str
            Column name prefix.

        Returns
        -------
        pd.DataFrame
            Converted DataFrame.
        """
        if arr.ndim == 1:
            return pd.DataFrame({f"{prefix}_0": arr})

        if arr.ndim == 2:
            cols = [f"{prefix}_{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols)

        raise ValueError(
            f"Cannot convert {arr.ndim}D array to DataFrame. " "Supported: 1D, 2D"
        )

    def _generate_model_features(
        self,
        context: Dict[str, Any],
        datamodule: Any,
    ) -> None:
        """Generate features from model predictions.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.
        datamodule : Any
            DataModule instance.
        """
        model = context.get("model")
        if model is None:
            print(
                f"[{self.step_id}] Warning: output_as_feature=True "
                "but no model found"
            )
            return

        # Get data for prediction
        if hasattr(datamodule, "get_test_windows"):
            x_data, _ = datamodule.get_test_windows()
        else:
            x_data = datamodule.get_data()[-2]

        # Generate predictions
        preds = model.predict(x_data)
        features = np.asarray(preds, dtype=float)

        self.set_output(context, "features", features)
        print(f"[{self.step_id}] Generated features from model: " f"{features.shape}")


StepFactory.register("datamodule", DataModuleStep)
