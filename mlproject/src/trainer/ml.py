from typing import Any, Dict, Optional

from mlproject.src.trainer.base import BaseTrainer


class MLTrainer(BaseTrainer):
    """Trainer for non-deep-learning (classic ML) models."""

    def train(
        self,
        datamodule: Any,
        hyperparams: Optional[Dict] = None,
    ):
        """
        Train the model wrapper on provided datasets.

        Args:
            train_data: Training dataset
            val_data: Optional validation dataset
            hyperparams: Optional hyperparameters
        """
        x_train, y_train, x_val, y_val, _, _ = datamodule.get_data()
        fit_kwargs = {}
        if x_val is not None and y_val is not None:
            fit_kwargs["x_val"] = x_val
            fit_kwargs["y_val"] = y_val

        self.wrapper.fit(x_train, y_train, **fit_kwargs)
        self.save()
        return self.wrapper
