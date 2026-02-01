from typing import List, Optional, Sequence

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold, StratifiedKFold

from mlproject.src.datamodule.loader import resolve_datasets_from_cfg


class BaseSplitter:
    """
    K-fold splitter for base datasets.

    The splitter partitions a DataFrame into `n_splits` folds and returns
    a list of DataFrames. Each fold contains the FULL set of columns,
    including target columns.

    Target columns are used ONLY for stratified splitting when enabled.
    """

    def __init__(self, cfg: DictConfig, n_splits: int) -> None:
        """
        Initialize the splitter from configuration and number of folds.

        Args:
            cfg: Dictionary-like configuration object.
            n_splits: Number of folds to generate.
        """
        self.cfg = cfg if cfg is not None else OmegaConf.create()
        self.df_cfg = cfg.get("data", OmegaConf.create())

        self.n_splits = int(n_splits)
        self.shuffle = bool(self.df_cfg.get("shuffle", True))
        self.random_state = int(self.df_cfg.get("random_state", 42))
        self.stratify = bool(self.df_cfg.get("stratify", False))

        self.target_columns = self._resolve_target_columns()

    def _load_data(
        self,
    ) -> pd.DataFrame:
        """
        Load the DataFrame from disk, parse datetime index, and sort.

        Returns:
            The loaded and indexed DataFrame.

        Raises:
            FileNotFoundError: If the CSV file path is invalid.
            ValueError: If index column is missing.
        """

        df, train_df, val_df, _ = resolve_datasets_from_cfg(self.cfg)
        if len(df) > 0:
            df = df.sort_index()
        else:
            df = pd.concat([train_df, val_df], axis=0).sort_index()
        return df

    def _resolve_target_columns(self) -> Sequence[str]:
        data_cfg = self.cfg.get("data", {})
        target_cols = data_cfg.get("target_columns", [])

        if isinstance(target_cols, str):
            return [target_cols]

        if isinstance(target_cols, (list, tuple)):
            return list(target_cols)

        return []

    def _resolve_stratify_y(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Resolve target series for stratified splitting only.
        """
        if not self.target_columns:
            return None

        valid_targets = [c for c in self.target_columns if c in df.columns]

        # StratifiedKFold requires exactly one target column
        if len(valid_targets) != 1:
            return None

        return df[valid_targets[0]]

    def generate_folds(self) -> List[pd.DataFrame]:
        """
        Split DataFrame into K folds.

        Returns
        -------
        List[pd.DataFrame]
            A list of DataFrames, each representing one fold.
            All original columns (including target) are preserved.
        """
        df = self._load_data()
        if self.stratify:
            y = self._resolve_stratify_y(df)
            if y is None:
                raise ValueError(
                    "Stratified splitting requires exactly one valid target "
                    "column defined in cfg.data.target_columns."
                )

            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
            split_iter = splitter.split(df, y)
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
            split_iter = splitter.split(df)

        folds: List[pd.DataFrame] = []

        for train_idx, test_idx in split_iter:
            # Train set: K-1 folds
            train_df = df.iloc[train_idx].reset_index(drop=True)
            train_df["dataset"] = "train"
            # Test set: 1 fold
            test_df = df.iloc[test_idx].reset_index(drop=True)
            test_df["dataset"] = "test"
            fold_df = pd.concat([train_df, test_df], axis=0)
            folds.append(fold_df)
        return folds
