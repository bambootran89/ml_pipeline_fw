import textwrap
from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from mlproject.src.datamodule.dataset import NumpyWindowDataset


class BaseDataModule:
    """Base class for ML/DL time-series DataModules.

    This module handles:
        - Sliding-window generation for supervised time-series.
        - Train/validation/test temporal splitting.
        - DataLoader instantiation.

    Attributes:
        df (pd.DataFrame): Input time-series dataframe.
        cfg (dict): Configuration dictionary.
        target_column (str): Name of the target variable.
        input_chunk (int): Length of each input window.
        output_chunk (int): Length of each output prediction horizon.
        x_train, y_train, x_val, y_val, x_test, y_test (np.ndarray):
            Arrays generated after preprocessing.

    Raises:
        ValueError: If the dataframe is empty or windowing yields no samples.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: dict,
        input_chunk: int,
        output_chunk: int,
    ) -> None:
        """Initialize the data module.

        Args:
            df (pd.DataFrame): Input dataframe containing features and target.
            cfg (dict): Configuration dictionary.
            input_chunk (int): Length of input sequence.
            output_chunk (int): Length of output sequence.

        Raises:
            ValueError: If the dataframe is empty.
        """
        if df is None or len(df) == 0:
            raise ValueError("Input dataframe is empty.")

        self.df = df
        self.cfg = cfg
        self.input_chunk = input_chunk
        self.output_chunk = output_chunk
        data_cfg = self.cfg.get("data", {})
        self.target_columns = []
        for col in data_cfg.get("target_columns", []):
            if col in self.df.columns:
                self.target_columns.append(col)
        self.features = data_cfg.get("features", [])
        data_type = data_cfg.get(
            "type",
            "tabular",
        )
        self.data_type = data_type
        self.x_train: np.ndarray
        self.y_train: np.ndarray
        self.x_val: np.ndarray
        self.y_val: np.ndarray
        self.x_test: np.ndarray
        self.y_test: np.ndarray
        self._prepare_data()
        self.print_info()

    def print_info(self) -> None:
        """
        Prints full DataModule information with conditional sequence display.

        The sequence information (input/output chunks) is hidden for tabular data.
        Features are printed in full without truncation, wrapped to 88 characters.
        """
        line_width = 80
        border = "=" * line_width
        print(border)
        print(f"DATAMODULE SUMMARY - Type: {self.data_type.upper()}")
        print(border)

        # Basic metadata
        print(f"Total Samples:  {len(self.df)}")

        # Conditional display: Hide sequence info if data type is tabular
        if self.data_type.lower() != "tabular":
            print(
                f"Sequence:       Input={self.input_chunk}, Output={self.output_chunk}"
            )

        print(f"Target Columns: {', '.join(self.target_columns)}")

        # Full Ordered Features (Wrapped for line length)
        feat_header = "Features:       "
        all_features_str = ", ".join(self.features)

        # textwrap.fill handles multi-line alignment within 88 characters
        wrapped_output = textwrap.fill(
            all_features_str,
            width=88,
            initial_indent=feat_header,
            subsequent_indent=" " * len(feat_header),
        )
        print(wrapped_output)
        print(f"Total Count:    {len(self.features)} columns")

        # Data Split Shapes
        print("-" * line_width)
        self._print_shapes_summary()
        print(border)

    def _print_shapes_summary(self) -> None:
        """
        Helper to print shapes of initialized data splits.
        """
        splits = {
            "Train": (getattr(self, "x_train", None), getattr(self, "y_train", None)),
            "Val": (getattr(self, "x_val", None), getattr(self, "y_val", None)),
            "Test": (getattr(self, "x_test", None), getattr(self, "y_test", None)),
        }

        for name, (x_data, y_data) in splits.items():
            # Validate data presence and size
            if x_data is not None and x_data.size > 0:
                x_shape = str(x_data.shape)
                y_shape = str(y_data.shape) if y_data is not None else "N/A"
            else:
                x_shape = "N/A"
                y_shape = "N/A"
            print(f"{name:15} Split: X={x_shape}, Y={y_shape}")

    def setup(self) -> None:
        """Optional hook to extend setup logic."""

    def _prepare_data(self) -> None:
        """Generate windows and perform temporal splits.

        Split ratios taken from:

        cfg["preprocessing"]["split"] = {"train": x, "val": y, "test": z}

        Raises:
            ValueError: If no windows can be created.
        """
        if "dataset" not in self.df.columns:
            print("Data Module uses split param to split dataframn (df)")
            split_cfg = self.cfg.get("preprocessing", {}).get(
                "split",
                {"train": 0.6, "val": 0.2, "test": 0.2},
            )
            if self.data_type == "timeseries":
                x, y = self._create_windows(
                    self.df.sort_index(),
                    input_chunk=self.input_chunk,
                    output_chunk=self.output_chunk,
                )
            else:
                df_shuffled = self.df.sample(frac=1.0, random_state=42).reset_index(
                    drop=True
                )
                y = df_shuffled[self.target_columns].values
                x = df_shuffled.drop(columns=self.target_columns).values

            if len(x) == 0:
                raise ValueError("Windowing produced zero samples. Check input length.")

            r_train = split_cfg.get("train", 0.7)
            r_test = split_cfg.get("test", 0.2)
            r_val = split_cfg.get("val", 0.1)
            assert r_test > 0
            assert (r_test + r_train + r_val) <= 1
            n = len(x)
            n_train = int(n * r_train)
            n_val = int(n * split_cfg["val"])

            self.x_train = x[:n_train]
            self.y_train = y[:n_train]

            self.x_val = x[n_train : n_train + n_val]
            self.y_val = y[n_train : n_train + n_val]

            self.x_test = x[n_train + n_val :]
            self.y_test = y[n_train + n_val :]

            if r_val == 0:
                self.x_val = self.x_test
                self.y_val = self.y_test
        else:
            print("Data Module uses dataset columns to split dataframe(df)")
            cols = list(set(self.features + self.target_columns))
            train_df = self.df[self.df["dataset"] == "train"][cols].sort_index()
            test_df = self.df[self.df["dataset"] == "test"][cols].sort_index()
            val_df = self.df[self.df["dataset"] == "val"][cols].sort_index()
            if len(val_df) == 0:
                val_df = test_df.copy()

            if self.data_type == "timeseries":
                self.x_train, self.y_train = self._create_windows(
                    train_df,
                    input_chunk=self.input_chunk,
                    output_chunk=self.output_chunk,
                )
                self.x_val, self.y_val = self._create_windows(
                    val_df,
                    input_chunk=self.input_chunk,
                    output_chunk=self.output_chunk,
                )
                self.x_test, self.y_test = self._create_windows(
                    test_df,
                    input_chunk=self.input_chunk,
                    output_chunk=self.output_chunk,
                )
            else:
                self.x_train, self.y_train = (
                    train_df[self.features].values,
                    train_df[self.target_columns].values,
                )
                self.x_val, self.y_val = (
                    val_df[self.features].values,
                    val_df[self.target_columns].values,
                )
                self.x_test, self.y_test = (
                    test_df[self.features].values,
                    test_df[self.target_columns].values,
                )

    def _create_windows(
        self,
        df: pd.DataFrame,
        input_chunk: int,
        output_chunk: int,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert the dataframe into sliding windows.

        Args:
            df (pd.DataFrame): dataframe
            input_chunk (int): Input sequence length.
            output_chunk (int): Output horizon length.
            stride (int, optional): Window stride. Defaults to 1.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                x_windows: Array of shape (N, input_chunk, num_features)
                y_windows: Array of shape (N, output_chunk)

        Notes:
            A window is created only if both:
            - Input window is fully inside dataframe.
            - Output horizon fits within dataframe.
        """
        n = len(df)

        x_windows: list[np.ndarray] = []
        y_windows: list[np.ndarray] = []
        for end_idx in range(input_chunk, n - output_chunk + 1, stride):
            start_idx = end_idx - input_chunk
            x_win = df.iloc[start_idx:end_idx][self.features].values
            y_win = df.iloc[end_idx : end_idx + output_chunk][
                self.target_columns
            ].values
            x_windows.append(x_win)
            y_windows.append(y_win)

        if not x_windows:
            x_shape = (1, self.input_chunk, len(self.features))
            y_shape = (1, self.output_chunk)
            return np.empty(x_shape), np.empty(y_shape)

        return np.stack(x_windows), np.stack(y_windows)

    def get_data(self) -> Tuple[np.ndarray, ...]:
        """Get all train/val/test arrays.

        Returns:
            Tuple[np.ndarray, ...]: (x_train, y_train, x_val, y_val, x_test, y_test)

        Raises:
            AssertionError: If arrays are missing.
        """
        assert isinstance(self.x_train, np.ndarray)
        assert isinstance(self.y_train, np.ndarray)
        assert isinstance(self.x_val, np.ndarray)
        assert isinstance(self.y_val, np.ndarray)
        assert isinstance(self.x_test, np.ndarray)
        assert isinstance(self.y_test, np.ndarray)

        return (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
        )

    def get_loaders(
        self, batch_size: int = 16
    ) -> Tuple[DataLoader, DataLoader, None, None]:
        """Create PyTorch DataLoaders for train/validation sets.

        Args:
            batch_size (int, optional): Loader batch size. Defaults to 16.

        Returns:
            Tuple[DataLoader, DataLoader, None, None]:
                (train_loader, val_loader, None, None)
        """
        train_loader = DataLoader(
            NumpyWindowDataset(self.x_train, self.y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            NumpyWindowDataset(self.x_val, self.y_val),
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, val_loader, None, None

    def summary(self) -> Tuple[int, int, int]:
        """Return summary of dataset split sizes.

        Returns:
            Tuple[int, int, int]: (num_train, num_val, num_test)

        Raises:
            AssertionError: If internal arrays are missing.
        """
        assert isinstance(self.x_train, np.ndarray)
        assert isinstance(self.x_val, np.ndarray)
        assert isinstance(self.x_test, np.ndarray)

        return len(self.x_train), len(self.x_val), len(self.x_test)
