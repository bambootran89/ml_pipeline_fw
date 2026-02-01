"""
Test suite for DataModules using a config-driven approach.

Covers:
- Factory pattern with config registry
- DL DataModule (PyTorch DataLoaders)
- ML DataModule (numpy arrays)
- Edge cases & validation
- Config validation

Run:
    pytest tests/test_datamodules.py -v -s
"""


import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from mlproject.src.datamodule.base import BaseDataModule
from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.datamodule.ts_sequence_dm import TSDLDataModule


class DataGenerator:
    """Helper class to generate synthetic time series data."""

    @staticmethod
    def create_dataframe(n_periods: int = 200, freq: str = "H") -> pd.DataFrame:
        """
        Generate synthetic time series DataFrame.

        Args:
            n_periods: Number of time periods
            freq: Frequency string (e.g., 'H' for hourly)

        Returns:
            pd.DataFrame with datetime index and 3 features
        """
        idx = pd.date_range("2020-01-01", periods=n_periods, freq=freq)
        df = pd.DataFrame(
            {
                "HUFL": np.sin(np.arange(len(idx)) / 24)
                + np.random.randn(len(idx)) * 0.1,
                "MUFL": np.cos(np.arange(len(idx)) / 24)
                + np.random.randn(len(idx)) * 0.1,
                "mobility_inflow": np.random.rand(len(idx)) * 10,
            },
            index=idx,
        )
        df.index.name = "date"
        return df


class ConfigBuilder:
    """Helper class to build test configs."""

    @staticmethod
    def create_base_config() -> dict:
        """Create base config dict."""
        return {
            "preprocessing": {"split": {"train": 0.6, "val": 0.2, "test": 0.2}},
            "training": {"num_workers": 0},
            "data": {
                "features": ["HUFL", "MUFL", "mobility_inflow"],
                "target_columns": ["HUFL"],
                "type": "timeseries",
            },
        }

    @staticmethod
    def create_dl_config(
        input_chunk: int = 5, output_chunk: int = 2, batch_size: int = 16
    ) -> DictConfig:
        """Create DictConfig for DL models."""
        base = ConfigBuilder.create_base_config()
        base["experiment"] = {
            "model": "nlinear",
            "hyperparams": {
                "input_chunk_length": input_chunk,
                "output_chunk_length": output_chunk,
                "batch_size": batch_size,
            },
        }
        base["model_registry"] = {
            "nlinear": {
                "module": "mlproject.src.models.nlinear_wrapper",
                "class": "NLinearWrapper",
                "datamodule_type": "dl",
            }
        }
        return OmegaConf.create(base)

    @staticmethod
    def create_ml_config(input_chunk: int = 5, output_chunk: int = 2) -> DictConfig:
        """Create DictConfig for ML models."""
        base = ConfigBuilder.create_base_config()
        base["experiment"] = {
            "model": "xgboost",
            "hyperparams": {
                "input_chunk_length": input_chunk,
                "output_chunk_length": output_chunk,
            },
        }
        base["model_registry"] = {
            "xgboost": {
                "module": "mlproject.src.models.xgboost_wrapper",
                "class": "XGBWrapper",
                "datamodule_type": "ml",
            }
        }
        return OmegaConf.create(base)


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """Fixture providing synthetic DataFrame."""
    return DataGenerator.create_dataframe(n_periods=200, freq="H")


@pytest.fixture
def dl_config() -> DictConfig:
    """Fixture providing DL config."""
    return ConfigBuilder.create_dl_config(input_chunk=5, output_chunk=2, batch_size=16)


@pytest.fixture
def ml_config() -> DictConfig:
    """Fixture providing ML config."""
    return ConfigBuilder.create_ml_config(input_chunk=5, output_chunk=2)


class TestDataModuleFactory:
    """Test suite for DataModuleFactory (config-driven)."""

    def test_factory_creates_dl_datamodule(
        self, synthetic_data: pd.DataFrame, dl_config: DictConfig
    ):
        """Test factory creates TSDLDataModule for DL models."""
        dm = DataModuleFactory.build(dl_config, synthetic_data)
        assert isinstance(dm, TSDLDataModule)
        assert dm.input_chunk == 5
        assert dm.output_chunk == 2

    def test_factory_creates_ml_datamodule(
        self, synthetic_data: pd.DataFrame, ml_config: DictConfig
    ):
        """Test factory creates TSBaseDataModule for ML models."""
        dm = DataModuleFactory.build(ml_config, synthetic_data)
        assert isinstance(dm, BaseDataModule)
        assert dm.input_chunk == 5
        assert dm.output_chunk == 2

    def test_factory_invalid_datamodule_type(self, synthetic_data: pd.DataFrame):
        """Test factory raises error for invalid datamodule_type."""
        cfg = ConfigBuilder.create_dl_config()
        cfg.model_registry.nlinear.datamodule_type = "invalid"
        with pytest.raises(ValueError, match="Invalid datamodule_type"):
            DataModuleFactory.build(cfg, synthetic_data)

    def test_factory_missing_model_in_registry(self, synthetic_data: pd.DataFrame):
        """Test factory raises error when model not in registry."""
        cfg = ConfigBuilder.create_dl_config()
        cfg.experiment.model = "unknown_model"
        with pytest.raises(ValueError, match="not found in model_registry"):
            DataModuleFactory.build(cfg, synthetic_data)


class TestTSDLDataModule:
    """Test suite for TSDLDataModule (PyTorch DL)."""

    def test_setup_creates_dataloaders(
        self, synthetic_data: pd.DataFrame, dl_config: DictConfig
    ):
        """Test setup() creates train/val DataLoaders."""
        dm = TSDLDataModule(
            df=synthetic_data,
            cfg=dl_config,
            input_chunk=5,
            output_chunk=2,
        )
        dm.setup()
        train_loader, val_loader, input_chunk, output_chunk = dm.get_loaders()
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert input_chunk == 5
        assert output_chunk == 2

    def test_batch_shapes(self, synthetic_data: pd.DataFrame, dl_config: DictConfig):
        """Test batch shapes from DataLoader."""
        dm = TSDLDataModule(
            df=synthetic_data,
            cfg=dl_config,
            input_chunk=5,
            output_chunk=2,
        )
        dm.setup()
        train_loader, _, _, _ = dm.get_loaders()
        x_batch, y_batch = next(iter(train_loader))
        assert x_batch.ndim == 3
        assert x_batch.shape[1] == 5
        assert x_batch.shape[2] == 3
        assert y_batch.ndim == 3
        assert y_batch.shape[1] == 2

    def test_test_windows(self, synthetic_data: pd.DataFrame, dl_config: DictConfig):
        """Test get_test_windows() returns correct shapes."""
        dm = TSDLDataModule(
            df=synthetic_data,
            cfg=dl_config,
            input_chunk=5,
            output_chunk=2,
        )
        x_test, y_test = dm.get_test_windows()
        assert x_test.shape[1] == 5
        assert x_test.shape[2] == 3
        assert y_test.shape[1] == 2

    def test_summary(self, synthetic_data: pd.DataFrame, dl_config: DictConfig):
        """Test summary() returns dataset sizes and approximate split ratios."""
        dm = TSDLDataModule(
            df=synthetic_data,
            cfg=dl_config,
            input_chunk=5,
            output_chunk=2,
        )
        n_train, n_val, n_test = dm.summary()
        total = n_train + n_val + n_test
        assert n_train > 0 and n_val > 0 and n_test > 0
        assert abs(n_train / total - 0.6) < 0.1
        assert abs(n_val / total - 0.2) < 0.1
        assert abs(n_test / total - 0.2) < 0.1


class TestTSBaseDataModule:
    """Test suite for TSBaseDataModule (traditional ML)."""

    def test_get_data_returns_numpy_arrays(
        self, synthetic_data: pd.DataFrame, ml_config: DictConfig
    ):
        """Test get_data() returns numpy arrays."""
        dm = BaseDataModule(
            df=synthetic_data,
            cfg=OmegaConf.to_container(ml_config, resolve=True),
            input_chunk=5,
            output_chunk=2,
        )
        X_train, y_train, X_val, y_val, X_test, y_test = dm.get_data()
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_test, np.ndarray)

    def test_array_shapes(self, synthetic_data: pd.DataFrame, ml_config: DictConfig):
        """Test array shapes match expected dimensions."""
        dm = BaseDataModule(
            df=synthetic_data,
            cfg=OmegaConf.to_container(ml_config, resolve=True),
            input_chunk=5,
            output_chunk=2,
        )
        X_train, y_train, X_val, y_val, X_test, y_test = dm.get_data()
        assert X_test.shape[1] == 5
        assert X_test.shape[2] == 3
        assert y_test.shape[1] == 2

    def test_summary(self, synthetic_data: pd.DataFrame, ml_config: DictConfig):
        """Test summary() returns dataset sizes."""
        dm = BaseDataModule(
            df=synthetic_data,
            cfg=OmegaConf.to_container(ml_config, resolve=True),
            input_chunk=5,
            output_chunk=2,
        )
        n_train, n_val, n_test = dm.summary()
        assert n_train > 0 and n_val > 0 and n_test > 0
