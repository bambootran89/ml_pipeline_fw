import torch
from torch import nn

from mlproject.src.models.base import DLModelWrapperBase


class FallbackNLinear(nn.Module):
    """
    Simple feedforward neural network with one hidden layer and ReLU activation.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the NLinear fallback model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_dim)
                              or (batch, seq_len, input_dim).

        Returns:
            torch.Tensor: Output predictions of shape (batch, output_dim).
        """
        if x.dim() == 3:  # (batch, seq, feat)
            b, s, f = x.shape
            x = x.reshape(b, s * f)
        return self.net(x)


class NLinearWrapper(DLModelWrapperBase):
    """
    Wrapper for FallbackNLinear.
    Provides build(), train_step(), and predict() following the DLModelWrapperBase API.
    """

    def build(self, model_type: str) -> None:
        """
        Build the internal FallbackNLinear model.

        Args:
            model_type (str):type of model
        """
        print(f"building {model_type}")
        hidden = self.cfg.get("hidden", 128)
        n_features = self.cfg.get("n_features", 4)

        input_dim = self.cfg.get("input_chunk_length", 24) * n_features
        output_dim = self.cfg.get("output_chunk_length", 6) * self.n_targets
        self.model = FallbackNLinear(input_dim, output_dim, hidden=hidden)
        self.model_type = model_type

    def train_step(self, batch, optimizer, loss_fn, device):
        """
        Perform a training step.

        Args:
            batch (tuple): Tuple (x, y) as numpy arrays.
            optimizer: Optimizer instance.
            loss_fn: Loss function.
            device: Torch device.

        Returns:
            float: Loss value.
        """
        self.model.train()

        x, y = batch
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        y_t = y_t.reshape(y_t.size(0), -1)

        preds = self.model(x_t)
        loss = loss_fn(preds, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return float(loss.item())

    def predict(self, x, **kwargs):
        """
        Predict outputs for the given input.

        NOTE:
            The signature matches the BaseModelWrapper:
                predict(self, x: Any, **kwargs: Any)

        Args:
            x (Any): Input numpy array or tensor.

        Returns:
            numpy.ndarray: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model has not been built or trained.")

        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            out = self.model(t)
            out = out.reshape(out.size(0), -1, self.n_targets)
            return out.cpu().numpy()
