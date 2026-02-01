from typing import Any, Tuple

import torch
from torch import nn

from mlproject.src.models.base import DLModelWrapperBase


class TFTFallback(nn.Module):
    """
    Minimal LSTM-based model used as a fallback for
    Temporal Fusion Transformer (TFT) behavior.

    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        hidden_size (int, optional): LSTM hidden layer size. Default is 64.
        num_layers (int, optional): Number of LSTM layers. Default is 1.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM and linear head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim)
                              or (batch, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, output_dim).
        """
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)


class TFTWrapper(DLModelWrapperBase):
    """
    Wrapper for training and prediction using a fallback TFT-like model.

    Args:
        cfg (Optional[Dict[str, Any]]): Optional configuration dict
        (keys: 'hidden_size', 'num_layers').
    """

    def build(self, model_type: str) -> None:
        """
        Build the TFTFallback model with specified input and output dimensions.

        Args:
            model_type (str): type of model
        """
        print(f"building {model_type}")
        hidden = self.cfg.get("hidden_size", 64)
        layers = self.cfg.get("num_layers", 1)
        input_dim = self.cfg.get("n_features", 4)
        output_dim = self.cfg.get("output_chunk_length", 6) * self.n_targets

        self.model = TFTFallback(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden,
            num_layers=layers,
        )

        self.model_type = model_type

    @staticmethod
    def _ensure_seq_dim(x: torch.Tensor) -> torch.Tensor:
        """
        Ensure input tensor has a sequence dimension (add if missing).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with sequence dimension.
        """
        if x.ndim == 2:
            return x.unsqueeze(1)
        return x

    def _ensure_float(self, x):
        """
        Convert input to torch.float32 tensor if needed.

        Args:
            x (Any): Input array or tensor.

        Returns:
            torch.Tensor: Float32 tensor.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.to(torch.float32)
        return x

    def train_step(
        self,
        batch: Tuple[Any, Any],
        optimizer: torch.optim.Optimizer,
        loss_fn,
        device: torch.device,
    ) -> float:
        """
        Perform one training step (forward, loss computation, backward, optimizer step).

        Args:
            batch (Tuple): Tuple (x, y) of training data.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            loss_fn: Loss function.
            device (torch.device): Device to run the training on.

        Returns:
            float: Training loss for the batch.
        """
        assert self.model is not None  # <-- fix mypy

        self.model.train()

        x, y = batch
        y = y.reshape(y.shape[0], -1)
        x = self._ensure_float(x).to(device)
        y = self._ensure_float(y).to(device)

        x = self._ensure_seq_dim(x)

        preds = self.model(x)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return float(loss.item())

    def predict(self, x: Any, **kwargs: Any):
        """
        Make predictions with the trained TFTFallback model.

        Args:
            x (Any): Input data array of shape (batch, seq_len, input_dim)
                    or (batch, input_dim).

        Returns:
            np.ndarray: Predicted outputs.

        Raises:
            RuntimeError: If model is not built or trained yet.
            ValueError: If input features do not match model input size.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not built/trained yet — call build() or train_step() first."
            )

        assert self.model is not None  # fix mypy

        t = self._ensure_float(x)
        t = self._ensure_seq_dim(t)

        if t.shape[-1] != self.model.rnn.input_size:
            raise ValueError(
                f"[ERROR] predict() feature mismatch — model expects "
                f"{self.model.rnn.input_size}, but input has {t.shape[-1]}."
            )

        self.model.eval()
        with torch.no_grad():
            out = self.model(t)

            out = out.reshape(out.size(0), -1, self.n_targets)
            return out.cpu().numpy()
