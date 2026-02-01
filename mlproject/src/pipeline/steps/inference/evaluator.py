"""Enhanced evaluator step with multi-source feature support."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.eval.base import BaseEvaluator
from mlproject.src.eval.classification import ClassificationEvaluator
from mlproject.src.eval.clustering import ClusteringEvaluator
from mlproject.src.eval.regression import RegressionEvaluator
from mlproject.src.eval.timeseries import TimeSeriesEvaluator
from mlproject.src.pipeline.steps.core.base import PipelineStep
from mlproject.src.pipeline.steps.core.constants import ContextKeys
from mlproject.src.pipeline.steps.core.factory import StepFactory
from mlproject.src.pipeline.steps.core.utils import ConfigAccessor


def _ensure_numpy(x: Any) -> np.ndarray:
    """Convert to numpy array."""
    if isinstance(x, pd.DataFrame):
        return x.to_numpy()
    return np.asarray(x, dtype=float)


class EvaluatorStep(PipelineStep):
    """Evaluate model with multi-source feature composition support.

    This step composes features from multiple sources before evaluation,
    allowing models trained on composed features to be evaluated properly.

    Context Inputs (configurable via wiring)
    -----------------------------------------
    features : pd.DataFrame or array-like
        Primary feature source.
    additional_feature_keys : List[str], optional
        Additional feature source keys to compose.
    targets : pd.DataFrame or array-like, optional
        Target labels for evaluation.
    predictions : np.ndarray, optional
        Pre-computed predictions (MODE 1).
    model : Any, optional
        Model wrapper for inference (MODE 2).
    datamodule : Any, optional
        Pre-built DataModule (MODE 2).

    Context Outputs
    ---------------
    metrics : Dict[str, float]
        Evaluation metrics.
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        step_eval_type: str = "",
        additional_feature_keys: Optional[List[str]] = None,
        feature_align_method: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Initialize enhanced evaluator.

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
        step_eval_type : str
            Override evaluation type (clustering, classification, etc).
        additional_feature_keys : Optional[List[str]]
            Keys for additional feature sources.
        feature_align_method : str
            Method for aligning feature shapes.
        **kwargs
            Additional parameters.
        """
        super().__init__(
            step_id,
            cfg,
            enabled,
            depends_on,
            additional_feature_keys=additional_feature_keys,
            feature_align_method=feature_align_method,
            **kwargs,
        )

        self.step_eval_type = step_eval_type
        if "cluster" in self.step_id:
            self.step_eval_type = "clustering"

        self.evaluator = self._build_evaluator()

    def _build_evaluator(self) -> BaseEvaluator:
        """Build evaluator based on config.

        Returns
        -------
        BaseEvaluator
            Appropriate evaluator instance.
        """
        if self.step_eval_type:
            eval_type = self.step_eval_type
        else:
            eval_type = self.cfg.get("evaluation", {}).get("type", "regression")

        evaluator_map = {
            "classification": ClassificationEvaluator,
            "regression": RegressionEvaluator,
            "clustering": ClusteringEvaluator,
            "timeseries": TimeSeriesEvaluator,
        }

        evaluator_class = evaluator_map.get(eval_type)
        if evaluator_class is None:
            raise ValueError(f"Unsupported eval type: {eval_type}")

        return evaluator_class()

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute evaluation with feature composition.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Context with metrics added.
        """
        self.validate_dependencies(context)

        # MODE 1: Evaluate using pre-computed predictions
        preds = self.get_input(context, "predictions", required=False)
        targets = self.get_input(context, "targets", required=False)

        if preds is not None and targets is not None:
            metrics = self._eval_from_predictions(preds, targets)
            self.set_output(context, "metrics", metrics)
            self._log_metrics(metrics)
            return context

        # MODE 2: Compose features and run model inference
        model = self.get_input(context, "model", required=False)
        datamodule = self.get_input(context, "datamodule", required=False)

        if datamodule is None:
            datamodule = self._build_datamodule_with_composition(context)

        y_true, y_pred, x_test = self._eval_from_datamodule(model, datamodule)

        metrics = self.evaluator.evaluate(y_true, y_pred, x=x_test, model=model)

        self.set_output(context, "metrics", metrics)
        self._log_metrics(metrics)

        return context

    def _eval_from_predictions(
        self,
        preds: Any,
        targets: Any,
    ) -> Dict[str, float]:
        """Evaluate from pre-computed predictions.

        Parameters
        ----------
        preds : Any
            Model predictions.
        targets : Any
            True labels.

        Returns
        -------
        Dict[str, float]
            Evaluation metrics.
        """
        y_true = _ensure_numpy(targets)
        y_pred = _ensure_numpy(preds)

        return self.evaluator.evaluate(y_true, y_pred)

    def _build_datamodule_with_composition(
        self,
        context: Dict[str, Any],
    ) -> Any:
        """Build DataModule with composed features.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Any
            DataModule instance.
        """
        # Compose features
        features_df, metadata = self.get_composed_features(
            context, "features", required=True
        )

        # Get targets
        targets_df = self.get_input(context, "targets", required=False)

        # Build input DataFrame
        config_accessor = ConfigAccessor(self.cfg)

        # Get expected feature names from config
        expected_features = config_accessor.get_feature_columns()

        # Fix column names if they were lost (e.g., from numpy array)
        features_df = self._restore_column_names(
            features_df, expected_features, metadata
        )

        if config_accessor.is_timeseries():
            input_df = features_df.copy()
        elif targets_df is not None:
            input_df = pd.concat([features_df, targets_df], axis=1)
        else:
            input_df = features_df.copy()

        # Get composed feature names for config injection
        composed_feature_names = list(features_df.columns)

        # Debug: print shape and columns before building DataModule
        print(
            f"[{self.step_id}] DEBUG - Features shape: {input_df.shape}, "
            f"columns: {list(input_df.columns)[:10]}..."
        )

        # Inject composed feature names into config for DataModule
        cfg_for_dm = self._inject_composed_features_to_config(
            composed_feature_names, context
        )

        # Build DataModule
        print(
            f"[{self.step_id}] Building DataModule with composed features: "
            f"{input_df.shape}"
        )

        dm = DataModuleFactory.build(cfg_for_dm, input_df)
        dm.setup()

        return dm

    def _restore_column_names(
        self,
        df: pd.DataFrame,
        expected_features: List[str],
        metadata: Dict[str, tuple],
    ) -> pd.DataFrame:
        """Restore column names if they were lost.

        Args:
            df: DataFrame with potentially numeric column names.
            expected_features: Expected feature names from experiment config.
            metadata: Feature composition metadata with column indices.

        Returns:
            pd.DataFrame: DataFrame with proper column names.
        """
        current_columns = list(df.columns)
        if not self._should_restore(current_columns, expected_features):
            return df

        print(f"[{self.step_id}] Restoring column names...")

        # Get base feature indices from metadata
        base_start, base_end = metadata.get("base", (0, len(expected_features)))
        new_columns = self._generate_base_columns(
            current_columns, expected_features, base_end - base_start
        )

        # Handle additional features
        for source, (start, end) in metadata.items():
            if source != "base":
                extra_cols = self._generate_extra_columns(
                    current_columns, source, start, end
                )
                new_columns.extend(extra_cols)
                print(f"  {source}: {end - start} columns")

        if len(new_columns) == len(df.columns):
            df = df.copy()
            df.columns = new_columns
        else:
            print(
                f"  Warning: Mismatch (expected \
                {len(new_columns)}, got {len(df.columns)})"
            )

        return df

    def _should_restore(
        self, current_columns: List, expected_features: List[str]
    ) -> bool:
        """Check if columns need restoration."""
        if not current_columns:
            return False
        first_col = str(current_columns[0])
        return (
            first_col.isdigit()
            or first_col.startswith("base_")
            or (bool(expected_features) and first_col not in expected_features)
        )

    def _generate_base_columns(
        self, current_columns: List, expected: List[str], n_base: int
    ) -> List[str]:
        """Generate names for base features."""
        if expected and len(expected) == n_base:
            print(f"  Base: restored {len(expected)} feature names")
            return list(expected)

        names = []
        for i in range(n_base):
            name = current_columns[i] if i < len(current_columns) else f"feature_{i}"
            names.append(str(name))
        return names

    def _generate_extra_columns(
        self, current_columns: List, source: str, start: int, end: int
    ) -> List[str]:
        """Generate names for additional features based on metadata."""
        names = []
        for i in range(end - start):
            col_idx = start + i
            if col_idx < len(current_columns):
                existing = str(current_columns[col_idx])
                if existing.startswith(f"{source}_"):
                    names.append(existing)
                    continue
            names.append(f"{source}_{i}")
        return names

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

        original_features = OmegaConf.select(cfg_copy, "data.features", default=[])

        # Always inject composed feature names if they differ from original
        # This handles both:
        # 1. Cases with additional_feature_keys (composed features)
        # 2. Cases where column names were lost (e.g., from numpy arrays)
        should_inject = False

        if self.additional_feature_keys:
            # We have additional features, definitely inject
            should_inject = True
        elif composed_feature_names:
            # Check if column names are numeric (lost original names)
            # or different from expected features
            first_col = str(composed_feature_names[0]) if composed_feature_names else ""
            if first_col.isdigit() or (
                original_features and composed_feature_names != list(original_features)
            ):
                # Columns are numeric or mismatched - use original feature names
                # if they match the count
                if original_features and len(original_features) == len(
                    composed_feature_names
                ):
                    composed_feature_names = list(original_features)
                    should_inject = True

        if should_inject and composed_feature_names:
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
            context[ContextKeys.COMPOSED_FEATURE_NAMES] = composed_feature_names
            context[ContextKeys.ADDITIONAL_FEATURE_KEYS] = self.additional_feature_keys

        return cast(DictConfig, cfg_copy)

    def _eval_from_datamodule(
        self,
        model: Any,
        datamodule: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run prediction and get results.

        Parameters
        ----------
        model : Any
            Model wrapper.
        datamodule : Any
            DataModule instance.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            True labels, predictions, test features.
        """
        if hasattr(datamodule, "get_test_windows"):
            x_test, y_true = datamodule.get_test_windows()
        else:
            data = datamodule.get_data()
            x_test, y_true = data[-2], data[-1]

        y_pred = model.predict(x_test)

        return y_true, y_pred, x_test

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics.

        Parameters
        ----------
        metrics : Dict[str, float]
            Evaluation metrics.
        """
        print(f"[{self.step_id}] Evaluation complete")
        print(f"  Metrics: {metrics}")


StepFactory.register("evaluator", EvaluatorStep)
