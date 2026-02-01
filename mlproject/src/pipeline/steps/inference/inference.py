"""Enhanced inference step with multi-source feature support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mlproject.src.pipeline.steps.core.base import PipelineStep
from mlproject.src.pipeline.steps.core.factory import StepFactory
from mlproject.src.pipeline.steps.core.utils import ConfigAccessor, WindowBuilder


class InferenceStep(PipelineStep):
    """Generate predictions with multi-source feature composition.

    This step composes features from multiple sources before inference,
    enabling models trained on engineered features to make predictions.

    Context Inputs (configurable via wiring)
    -----------------------------------------
    features : pd.DataFrame or array-like
        Primary feature source.
    additional_feature_keys : List[str], optional
        Additional feature source keys to compose.
    model : Any
        Trained model for inference.

    Context Outputs
    ---------------
    predictions : np.ndarray
        Model predictions.
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        include_inputs: bool = False,
        additional_feature_keys: Optional[List[str]] = None,
        feature_align_method: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Initialize enhanced inference step.

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
        save_path : Optional[str]
            Path to save predictions.
        include_inputs : bool
            Whether to save input features with predictions.
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
        self.save_path = save_path
        self.include_inputs = include_inputs

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference with feature composition.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Context with predictions added.
        """
        self.validate_dependencies(context)

        # Compose features from multiple sources
        features_df, _ = self.get_composed_features(context, "features", required=True)

        # Get model
        model = self.get_input(context, "model")

        # Prepare input
        x = self._prepare_model_input(features_df)

        # Run inference
        print(f"[{self.step_id}] Running inference on shape {x.shape}")
        predictions = model.predict(x)

        # Store output - always store to predictions
        self.set_output(context, "predictions", predictions)

        # Also store to features if configured (for clustering/feature-generating steps)
        # This ensures clustering models that output_as_feature work correctly in
        # serve mode
        if "features" in self.output_keys:
            self.set_output(context, "features", predictions)
            print(
                f"[{self.step_id}] Also stored to features key: "
                f"{self.output_keys['features']}"
            )

        # Optional: Save predictions
        if self.save_path:
            self._save_predictions(
                predictions, features_df if self.include_inputs else None
            )

        print(
            f"[{self.step_id}] Inference complete. "
            f"Output shape: {predictions.shape}"
        )

        return context

    def _prepare_model_input(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare input array for model.

        Parameters
        ----------
        features_df : pd.DataFrame
            Composed features.

        Returns
        -------
        np.ndarray
            Model input array.
        """
        config_accessor = ConfigAccessor(self.cfg)

        if config_accessor.is_timeseries():
            return self._prepare_timeseries_input(features_df)

        # Tabular data: use as-is
        return features_df.values.astype(np.float32)

    def _prepare_timeseries_input(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Prepare timeseries windowed input.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        np.ndarray
            Windowed input array.
        """
        config_accessor = ConfigAccessor(self.cfg)
        entity_key = config_accessor.get_entity_key()
        window_config = config_accessor.get_window_config()

        win_in = window_config["input_chunk_length"]
        win_out = window_config["output_chunk_length"]
        stride = window_config.get("stride", win_out)

        # Use WindowBuilder utility
        return WindowBuilder.create_grouped_windows(
            df, entity_key, win_in, win_out, stride
        )

    def _save_predictions(
        self,
        predictions: np.ndarray,
        inputs: Optional[pd.DataFrame],
    ) -> None:
        """Save predictions to file.

        Parameters
        ----------
        predictions : np.ndarray
            Model predictions.
        inputs : Optional[pd.DataFrame]
            Input features if include_inputs=True.
        """
        df = pd.DataFrame(predictions, columns=["prediction"])

        if inputs is not None:
            df = pd.concat([inputs.reset_index(drop=True), df], axis=1)

        df.to_csv(self.save_path, index=False)
        print(f"[{self.step_id}] Saved predictions to {self.save_path}")


StepFactory.register("inference", InferenceStep)
