from typing import Any, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from mlproject.src.datamodule.base import BaseDataModule
from mlproject.src.datamodule.dataset import NumpyWindowDataset


class TSDLDataModule(BaseDataModule):
    """Deep Learning DataModule for PyTorch models.

    Extends TSBaseDataModule by adding DataLoader creation.
    """

    def __init__(
        self,
        df,
        cfg,
        input_chunk: int,
        output_chunk: int,
    ) -> None:
        """Initialize the deep learning DataModule.

        Args:
            df (pd.DataFrame): Dataset.
            cfg (dict): Configuration dictionary.
            input_chunk (int): Length of input sequence.
            output_chunk (int): Length of output sequence.
        """
        super().__init__(df, cfg, input_chunk, output_chunk)
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

    def setup(self) -> None:
        """Create train and validation DataLoaders.

        Reads batch_size and num_workers from config.
        """
        assert isinstance(self.x_train, np.ndarray)
        assert isinstance(self.y_train, np.ndarray)

        if isinstance(self.cfg, dict):
            batch_size = (
                self.cfg.get("experiment", {})
                .get("hyperparams", {})
                .get("batch_size", 16)
            )
            num_workers = (
                self.cfg.get("experiment", {})
                .get("hyperparams", {})
                .get("num_workers", 0)
            )
        else:
            batch_size = self.cfg.experiment.hyperparams.get("batch_size", 16)
            num_workers = self.cfg.experiment.hyperparams.get("num_workers", 0)

        self.train_loader = DataLoader(
            NumpyWindowDataset(self.x_train, self.y_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        assert isinstance(self.x_val, np.ndarray)
        assert isinstance(self.y_val, np.ndarray)

        self.val_loader = DataLoader(
            NumpyWindowDataset(self.x_val, self.y_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def get_loaders(self, batch_size: int = 16) -> Tuple[Any, ...]:
        """Return loaders with same method signature as parent class.

        Args:
            batch_size (int, optional):
                Ignored. Only kept to match superclass signature.

        Returns:
            tuple:
                (train_loader, val_loader, input_chunk, output_chunk)
        """
        assert self.train_loader is not None
        assert self.val_loader is not None

        return (
            self.train_loader,
            self.val_loader,
            self.input_chunk,
            self.output_chunk,
        )

    def get_test_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get input/output windows for the test set.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (x_test, y_test)
        """
        return self.x_test, self.y_test
