"""Flexible training pipeline using configuration-driven execution.

Enhanced to support runtime context pre-initialization for serving mode.
"""

from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig

from mlproject.src.pipeline.compat.v1.base import BasePipeline
from mlproject.src.pipeline.executor import PipelineExecutor
from mlproject.src.utils.config_class import ConfigLoader


class FlexiblePipeline(BasePipeline):
    """Configuration-driven flexible training pipeline.

    This pipeline replaces hardcoded logic with YAML-configured steps
    that can be composed, reordered, and conditionally enabled.

    Enhanced Features:
    ------------------
    - Support for runtime context pre-initialization (serving mode)
    - Backward compatible with existing usage
    """

    def __init__(self, cfg_path: str = "") -> None:
        """Initialize flexible pipeline.

        Parameters
        ----------
        cfg_path : str
            Path to configuration file.
        """
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)
        self.executor = PipelineExecutor(self.cfg)

    def preprocess(self) -> Optional[pd.DataFrame]:
        """Not used in flexible pipeline.

        Returns
        -------
        None
            Preprocessing is handled by pipeline steps.
        """
        return None

    def run_exp(
        self, data: Any = None, initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute configured pipeline with optional context pre-initialization.

        This method supports two modes of operation:

        1. Normal mode (training/eval):
           - Context starts empty
           - DataLoaderStep provides initial data

        2. Serving mode (test/inference):
           - Context pre-initialized with runtime data
           - DataLoaderStep can be skipped in pipeline config
           - Data injected from CLI/API before pipeline execution

        Parameters
        ----------
        data : Any, optional
            Unused. Kept for backward compatibility.
            Data loading is handled by pipeline steps or initial_context.
        initial_context : Dict[str, Any], optional
            Pre-initialized context for serving mode.
            If provided, these keys are available to all steps before execution.

            Common keys for serving mode:
            - df: Full input dataframe
            - train_df: Empty (not used in serving)
            - test_df: Same as df (will be preprocessed)
            - is_splited_input: False

        Returns
        -------
        Dict[str, Any]
            Final pipeline context containing all outputs from executed steps.

        Examples
        --------
        # Training mode (normal - backward compatible)
        >>> pipeline = FlexibleTrainingPipeline("train_config.yaml")
        >>> context = pipeline.run_exp()
        >>> metrics = context["evaluate_metrics"]

        # Serving mode with CSV data
        >>> pipeline = FlexibleTrainingPipeline("test_config.yaml")
        >>> df = pd.read_csv("test.csv")
        >>> initial_ctx = {
        ...     "df": df,
        ...     "train_df": pd.DataFrame(),
        ...     "test_df": df,
        ...     "is_splited_input": False
        ... }
        >>> context = pipeline.run_exp(initial_context=initial_ctx)
        >>> predictions = context["inference_predictions"]

        Notes
        -----
        - The `data` parameter is kept for backward compatibility but unused
        - When `initial_context` is None, behavior is identical to before
        - When `initial_context` is provided, executor starts with that context
        """
        # Execute pipeline with optional initial context
        context = self.executor.execute(initial_context=initial_context)
        return context
