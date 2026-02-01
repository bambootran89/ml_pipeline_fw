"""Data loading step for flexible pipeline với wiring support."""

from typing import Any, Dict, List, Optional

import pandas as pd

from mlproject.src.datamodule.loader import resolve_datasets_from_cfg
from mlproject.src.pipeline.steps.core.base import BasePipelineStep
from mlproject.src.pipeline.steps.core.constants import ColumnNames, ContextKeys
from mlproject.src.pipeline.steps.core.factory import StepFactory


class DataLoaderStep(BasePipelineStep):
    """Load raw data from configured source.

    This step loads data using the existing datamodule loader system
    and stores the result in context. Supports data wiring for
    custom output keys.

    Context Outputs (configurable via wiring)
    ------------------------------------------
    df : pd.DataFrame
        Full dataset.
    train_df : pd.DataFrame
        Training subset.
    val_df : pd.DataFrame
        Validation subset.
    test_df : pd.DataFrame
        Test subset.
    is_splited_input : bool
        Whether data was pre-split.
    feature_columns_size : int
        Size of features (for conditional branching).
    """

    DEFAULT_OUTPUTS = {
        "df": "df",
        "train_df": "train_df",
        "val_df": "val_df",
        "test_df": "test_df",
    }

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize data loading step."""
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Load data from configuration."""
        df, train_df, val_df, test_df = resolve_datasets_from_cfg(self.cfg)

        is_splited = False
        if len(df) == 0 and len(train_df) > 0:
            train_df[ColumnNames.DATASET] = ColumnNames.TRAIN
            val_df[ColumnNames.DATASET] = ColumnNames.VAL
            test_df[ColumnNames.DATASET] = ColumnNames.TEST
            df = pd.concat([train_df, val_df, test_df], axis=0)
            is_splited = True

        # Use wiring to set outputs
        self.set_output(context, "df", df)
        self.set_output(context, "train_df", train_df)
        self.set_output(context, "val_df", val_df)
        self.set_output(context, "test_df", test_df)
        context[ContextKeys.IS_SPLITED_INPUT] = is_splited

        print(f"[{self.step_id}] Loaded data: {df.shape}")
        print(f"[{self.step_id}] Train: {train_df.shape}")
        print(f"[{self.step_id}] Val: {val_df.shape}")
        print(f"[{self.step_id}] Test: {test_df.shape}")

        return context


# Register step type
StepFactory.register("data_loader", DataLoaderStep)
