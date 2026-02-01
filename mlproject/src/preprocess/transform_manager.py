"""
Transform Manager.

This module defines `TransformManager`, a utility class responsible for
managing stateful preprocessing transforms, including:

- Missing value imputation (mean / median / mode)
- Label encoding for categorical features
- Feature scaling (StandardScaler / MinMaxScaler)

All fitted states can be persisted and restored from disk to ensure
reproducibility across training and inference.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, TypedDict, Union, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

ScalerType = Union[StandardScaler, MinMaxScaler]


class FillnaStat(TypedDict):
    """Typed structure for fillna statistics of a single column."""

    method: str
    value: float


class TransformManager:
    """
    Manage all stateful preprocessing transforms.

    This class encapsulates fitting, applying, saving, and loading
    preprocessing steps with internal state.

    Attributes
    ----------
    artifacts_dir : str
        Directory used to persist transform artifacts.
    fillna_stats : Dict[str, LabelEncoder]
        Per-column missing value statistics.
    label_encoders : Dict[str, LabelEncoder]
        Label encoders for categorical columns.
    scaler : Optional[ScalerType]
        Fitted scaler instance.
    scaler_columns : Optional[List[str]]
        Columns used when fitting the scaler.
    """

    def __init__(self, cfg: DictConfig, artifacts_dir: str) -> None:
        """
        Initialize the TransformManager.

        Parameters
        ----------
        artifacts_dir : str
            Directory where transform artifacts are stored.
        """
        self.artifacts_dir: str = artifacts_dir
        self.is_load = False
        self.fillna_stats: Dict[str, FillnaStat] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[ScalerType] = None
        self.scaler_columns: Optional[List[str]] = None
        self.steps: List[Dict[str, Any]] = self.normalize_preprocessing_steps(cfg)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full data transformation pipeline in a deterministic order.

        This method serves as the single source of truth for all feature
        transformations and guarantees consistent preprocessing across
        training, validation, and inference stages.

        Transformation steps:
        1. Missing value handling
        2. Label encoding for categorical features
        3. Feature scaling

        Parameters
        ----------
        df : pd.DataFrame
            Input raw dataframe.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe ready for modeling.
        """
        df_out = df.copy()
        for step_cfg in self.steps:
            step_name = step_cfg.get("name", "")
            if step_name == "fill_missing":
                df_out = self.transform_fillna(
                    df_out,
                    columns=step_cfg.get("columns", list(df_out.columns)),
                    method=step_cfg.get("method", "mean"),
                )

            elif step_name == "label_encoding":
                df_out = self.transform_label_encoding(df_out)

            elif step_name == "normalize":
                df_out = self.transform_scaler(df_out)
            elif step_name in [
                "clip",
                "log",
                "abs",
                "round",
                "binary",
                "exponential",
                "udf",
            ]:
                self.stateless_transform(df_out, step_cfg)

            else:
                raise ValueError(f"Unknown transform step: {step_name}")

        return df_out

    def stateless_transform(
        self,
        df: pd.DataFrame,
        step_cfg: Dict[str, Any],
    ):
        """
        Apply a stateless preprocessing transformation.

        This method applies transformations that do not require fitting
        (e.g. clip, log, abs, round, binary, exponential, udf).
        The operation is selected based on the `name` field in `step_cfg`.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to be transformed.
        step_cfg : Dict[str, Any]
            Configuration of the preprocessing step. Must contain key `name`.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.

        Raises
        ------
        ValueError
            If the preprocessing step name is unknown.
        """
        step_name = step_cfg.get("name", "")
        if step_name == "clip":
            self.transform_clip(df, step_cfg)

        elif step_name == "log":
            self.transform_log(df, step_cfg)

        elif step_name == "abs":
            self.transform_abs(df, step_cfg)
        elif step_name == "round":
            self.transform_round(df, step_cfg)
        elif step_name == "binary":
            self.transform_binary(df, step_cfg)

        elif step_name == "exponential":
            self.transform_exponential(df, step_cfg)

        elif step_name == "udf":
            self.transform_udf(df, step_cfg)
        else:
            raise ValueError(f"Unknown transform step: {step_name}")

    def transform_exponential(
        self,
        df: pd.DataFrame,
        step_cfg: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Apply exponential transformation to numeric columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        step_cfg : Dict[str, Any]
            Configuration with `columns` and optional `scale`.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe.
        """
        columns: List[str] = step_cfg.get("columns", [])
        scale: float = float(step_cfg.get("scale", 1.0))

        if not isinstance(columns, list):
            raise TypeError("exponential transform requires `columns` as a list")

        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found for exponential transform")

            df[col] = np.exp(scale * df[col])

    def transform_binary(
        self,
        df: pd.DataFrame,
        step_cfg: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Convert numeric columns to binary values based on a threshold.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        step_cfg : Dict[str, Any]
            Configuration with `columns` and optional `threshold`.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe.
        """
        columns: List[str] = step_cfg.get("columns", [])
        threshold: float = float(step_cfg.get("threshold", 0.0))

        if not isinstance(columns, list):
            raise TypeError("binary transform requires `columns` as a list")

        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found for binary transform")

            df[col] = (df[col] > threshold).astype(int)

    def transform_abs(
        self,
        df: pd.DataFrame,
        step_cfg: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Apply absolute value transformation to specified columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        step_cfg : Dict[str, Any]
            Step configuration containing `columns`.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe.
        """
        columns: List[str] = step_cfg.get("columns", [])
        if not isinstance(columns, list):
            raise TypeError("abs transform requires `columns` as a list")

        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found for abs transform")
            df[col] = df[col].abs()

    def transform_round(
        self,
        df: pd.DataFrame,
        step_cfg: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Round numeric columns to a fixed number of decimals.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        step_cfg : Dict[str, Any]
            Step configuration containing `columns` and optional `decimals`.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe.
        """
        columns: List[str] = step_cfg.get("columns", [])
        decimals: int = int(step_cfg.get("decimals", 0))

        if not isinstance(columns, list):
            raise TypeError("round transform requires `columns` as a list")

        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found for round transform")
            df[col] = df[col].round(decimals)

    def transform_clip(
        self,
        df: pd.DataFrame,
        step_cfg: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Clip numerical values to a specified range.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        step_cfg : Dict[str, Any]
            Step configuration containing columns, min, and max.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """

        columns: List[str] = step_cfg.get("columns", [])
        min_val: Optional[float] = step_cfg.get("min")
        max_val: Optional[float] = step_cfg.get("max")

        if min_val is None and max_val is None:
            return

        for col in columns:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            df[col] = df[col].clip(lower=min_val, upper=max_val)

    def transform_log(
        self,
        df: pd.DataFrame,
        step_cfg: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Apply logarithmic transformation to numerical columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        step_cfg : Dict[str, Any]
            Step configuration containing columns and optional offset.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """

        columns: List[str] = step_cfg.get("columns", [])
        offset: float = float(step_cfg.get("offset", 0.0))

        for col in columns:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            values = df[col].astype(np.float64)

            if (values + offset <= 0).any():
                raise ValueError(
                    f"Log transform invalid for column '{col}': "
                    "non-positive values encountered."
                )

            df[col] = np.log(values + offset)

    def transform_udf(
        self,
        df: pd.DataFrame,
        step_cfg: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Apply a user-defined function to specified columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        step_cfg : Dict[str, Any]
            Step configuration containing columns and a callable function.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.

        Raises
        ------
        ValueError
            If UDF function is missing or not callable.
        """

        columns: List[str] = step_cfg.get("columns", [])
        func = step_cfg.get("func")

        if func is None or not callable(func):
            raise ValueError("UDF step requires a callable `func`.")

        for col in columns:
            if col not in df.columns:
                continue

            df[col] = df[col].apply(func)

    def fit_fillna(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "mean",
    ) -> None:
        """
        Fit missing-value statistics for the specified columns.

        The computed statistics are stored internally and later used by
        :meth:`transform_fillna` to impute missing values.

        Supported strategies:
            - "mean":   Column-wise mean
            - "median": Column-wise median
            - "mode":   Most frequent value

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame used to compute fillna statistics.
        columns : List[str]
            List of column names for which fillna statistics are computed.
        method : str, default="mean"
            Fillna strategy. Must be one of {"mean", "median", "mode"}.

        Raises
        ------
        ValueError
            If an unsupported fillna method is provided.
        """
        if method not in {"mean", "median", "mode", "ffill"}:
            raise ValueError(f"Invalid fillna method: {method}")
        if method == "ffill":
            for col in columns:
                df[col] = df[col].ffill()
                df[col] = df[col].bfill()
            return

        for col in columns:
            if col not in df.columns:
                continue

            if method == "mean":
                value: float = float(df[col].mean())
            elif method == "median":
                value = float(df[col].median())
            else:
                modes = df[col].mode()
                value = modes.iloc[0] if not modes.empty else 0.0
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    value = float(value)

            self.fillna_stats[col] = {
                "method": method,
                "value": value,
            }

    def transform_fillna(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "mean",
    ) -> pd.DataFrame:
        """
        Apply missing value imputation.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
                columns : List[str]
            List of column names for which fillna statistics are computed.
        method : str, default="mean"
            Fillna strategy. Must be one of {"mean", "median", "mode"}.
        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        df_out = df.copy()
        if method not in {"mean", "median", "mode", "ffill"}:
            raise ValueError(f"Invalid fillna method: {method}")
        if method == "ffill":
            for col in columns:
                df_out[col] = df_out[col].ffill()
                df_out[col] = df_out[col].bfill()
            return df_out

        for col, stats in self.fillna_stats.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(stats["value"])

        return df_out

    def fit_label_encoding(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> None:
        """
        Fit label encoders for categorical columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        columns : List[str]
            Categorical columns to encode.
        """
        for col in columns:
            if col not in df.columns:
                continue

            encoder = LabelEncoder()
            non_null_values = df[col].dropna()

            if not non_null_values.empty:
                encoder.fit(non_null_values.astype(str))
                self.label_encoders[col] = encoder

    @staticmethod
    def _encode_series_safe(
        values: pd.Series,
        encoder: LabelEncoder,
    ) -> pd.Series:
        """
        Encode a pandas Series using a fitted LabelEncoder.

        Any unseen label is mapped to the first known class.

        Parameters
        ----------
        values : pd.Series
            Input categorical values (no NaNs).
        encoder : LabelEncoder
            Fitted label encoder.

        Returns
        -------
        pd.Series
            Encoded integer values.
        """
        known_classes = set(encoder.classes_)
        default_class = encoder.classes_[0]

        safe_values = values.where(values.isin(known_classes), default_class)
        return pd.Series(
            encoder.transform(safe_values),
            index=values.index,
        )

    def transform_label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply label encoding to categorical columns.

        Unseen labels are mapped to the first known class.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        df_out = df.copy()

        for col, encoder in self.label_encoders.items():
            if col not in df_out.columns:
                continue

            mask = df_out[col].notna()
            if not mask.any():
                continue

            values = df_out.loc[mask, col].astype(str)

            df_out.loc[mask, col] = self._encode_series_safe(
                values=values,
                encoder=encoder,
            )

        return df_out

    def fit_scaler(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "zscore",
    ) -> None:
        """
        Fit a feature scaler.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        columns : List[str]
            Columns to scale.
        method : str, default="zscore"
            One of {"zscore", "minmax"}.

        Raises
        ------
        ValueError
            If an unsupported scaling method is provided.
        """
        if method == "zscore":
            scaler: ScalerType = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Invalid scaling method: {method}")

        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return

        numeric_df = df[valid_cols].astype(np.float64)
        scaler.fit(numeric_df.values)

        self.scaler = scaler
        self.scaler_columns = valid_cols

    def transform_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature scaling.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        if self.scaler is None or self.scaler_columns is None:
            return df

        df_out = df.copy()
        valid_cols = [c for c in self.scaler_columns if c in df_out.columns]

        if not valid_cols:
            return df_out

        for col in valid_cols:
            if not pd.api.types.is_float_dtype(df[col]):
                df_out[col] = df_out[col].astype(np.float64)

        scaled_values = self.scaler.transform(
            df_out[valid_cols].to_numpy(dtype=np.float64)
        )
        df_out.loc[:, valid_cols] = scaled_values

        return df_out

    def save(self) -> None:
        """
        Persist all transform states to disk.
        """
        os.makedirs(self.artifacts_dir, exist_ok=True)

        with open(os.path.join(self.artifacts_dir, "fillna_stats.pkl"), "wb") as f:
            pickle.dump(self.fillna_stats, f)

        with open(os.path.join(self.artifacts_dir, "label_encoders.pkl"), "wb") as f:
            pickle.dump(self.label_encoders, f)

        with open(os.path.join(self.artifacts_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "columns": self.scaler_columns,
                },
                f,
            )

    def normalize_preprocessing_steps(
        self,
        cfg: DictConfig,
    ) -> List[Dict[str, Any]]:
        """
        Normalize preprocessing steps from OmegaConf config.

        Converts `preprocessing.steps` into a validated list of plain
        Python dictionaries for safe downstream usage.
        """
        raw_steps = cfg.get("preprocessing", {}).get("steps", [])

        if not isinstance(raw_steps, (list, ListConfig)):
            raise TypeError(
                "preprocessing.steps must be a list or OmegaConf ListConfig"
            )

        steps: List[Dict[str, Any]] = []

        for idx, step in enumerate(raw_steps):
            if isinstance(step, DictConfig):
                step_dict_raw = OmegaConf.to_container(
                    step,
                    resolve=True,
                )
            elif isinstance(step, dict):
                step_dict_raw = step
            else:
                raise TypeError(
                    "Each preprocessing step must be dict or DictConfig, "
                    f"got {type(step)} at index {idx}"
                )

            if not isinstance(step_dict_raw, dict):
                raise TypeError(f"Step at index {idx} could not be converted to dict")

            if "name" not in step_dict_raw:
                raise ValueError(
                    f"Preprocessing step at index {idx} must contain key 'name'"
                )

            # mypy-safe cast after validation
            step_dict = cast(Dict[str, Any], step_dict_raw)
            steps.append(step_dict)

        return steps

    def load(self, cfg) -> None:
        """
        Load all transform states from disk.
        """

        if self.is_load:
            return

        self.steps = self.normalize_preprocessing_steps(cfg)
        path = os.path.join(self.artifacts_dir, "fillna_stats.pkl")
        print(f"Loading artifacts {path}")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.fillna_stats = pickle.load(f)

        path = os.path.join(self.artifacts_dir, "label_encoders.pkl")
        print(f"Loading artifacts {path}")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.label_encoders = pickle.load(f)

        path = os.path.join(self.artifacts_dir, "scaler.pkl")
        print(f"Loading artifacts {path}")
        if os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
                self.scaler = obj["scaler"]
                self.scaler_columns = obj["columns"]
        self.is_load = True

    def get_params(self) -> Dict[str, Any]:
        """
        Retrieve transform parameters for experiment tracking.

        Returns
        -------
        Dict[str, Any]
            Dictionary of transform parameters.
        """
        params: Dict[str, Any] = {}

        for col, stats in self.fillna_stats.items():
            params[f"fillna.{col}.method"] = stats["method"]
            params[f"fillna.{col}.value"] = stats["value"]

        for col, encoder in self.label_encoders.items():
            params[f"label_encode.{col}.n_classes"] = len(encoder.classes_)
            params[f"label_encode.{col}.classes"] = ",".join(
                map(str, encoder.classes_[:5])
            )

        if self.scaler is not None:
            params["scaler.type"] = type(self.scaler).__name__
            params["scaler.n_features"] = len(self.scaler_columns or [])

        return params
