from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from mlproject.src.trainer.base import BaseTrainer


def infer_dims_from_batch(xb: Any, yb: Any) -> tuple[int, int]:
    """Infer input and output dimensions from a batch."""
    if hasattr(xb, "shape"):
        input_dim = xb.shape[1] * xb.shape[2] if len(xb.shape) == 3 else xb.shape[1]
        output_dim = yb.shape[1] if len(yb.shape) > 1 else 1
    else:
        input_dim = xb[0].shape[0]
        output_dim = yb.shape[1] if len(yb.shape) > 1 else 1
    return input_dim, output_dim


class DeepLearningTrainer(BaseTrainer):
    """Trainer for deep learning models (PyTorch wrappers)."""

    def __init__(
        self,
        wrapper: Any,
        device: str = "cpu",
        save_dir: str = "mlproject/artifacts/models",
        model_type: str = "dl",
    ) -> None:
        super().__init__(wrapper, save_dir, model_type)
        self.device: torch.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

    def setup_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Setup Adam optimizer for the model."""
        if self.wrapper.model is None:
            raise RuntimeError("Model is not built yet. Call build() first.")
        return torch.optim.Adam(self.wrapper.model.parameters(), lr=lr)

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
    ) -> float:
        """Run one epoch over the training data and return mean loss."""
        losses: list[float] = []
        for xb, yb in train_loader:
            x_batch = xb.numpy() if hasattr(xb, "numpy") else xb
            y_batch = yb.numpy() if hasattr(yb, "numpy") else yb
            loss = self.wrapper.train_step(
                (x_batch, y_batch), optimizer, loss_fn, self.device
            )
            losses.append(loss)
        return float(np.mean(losses))

    def validate(self, val_loader: DataLoader) -> float:
        """Compute validation MSE over the validation set."""
        if self.wrapper.model is None:
            raise RuntimeError("Model is not built yet. Cannot validate.")
        losses: list[float] = []
        for xb, yb in val_loader:
            x_batch = xb.numpy() if hasattr(xb, "numpy") else xb
            y_true = yb.numpy() if hasattr(yb, "numpy") else yb
            preds = self.wrapper.predict(x_batch)
            losses.append(float(np.mean((preds - y_true) ** 2)))
        return float(np.mean(losses))

    def train(
        self,
        datamodule: Any,
        hyperparams: Dict[str, Any],
    ) -> Any:
        """Full training loop for n_epochs with validation after each epoch."""
        train_loader, val_loader, _, _ = datamodule.get_loaders()
        # Build model if needed
        if self.wrapper.model is None:
            self.wrapper.build(model_type=self.model_type)

        # Move model to device
        if self.wrapper.model is not None:
            self.wrapper.model.to(self.device)
        else:
            raise RuntimeError("Model build failed. Cannot move to device.")

        optimizer = self.setup_optimizer(float(hyperparams.get("lr", 1e-3)))
        n_epochs = int(hyperparams.get("n_epochs", 1))

        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_one_epoch(
                train_loader, optimizer, torch.nn.MSELoss()
            )
            val_loss = self.validate(val_loader)
            print(
                f"Epoch {epoch}/{n_epochs} \
                    - train_loss={train_loss:.6f} val_mse={val_loss:.6f}"
            )

        self.save()
        return self.wrapper
