import os
from abc import ABC, abstractmethod
from typing import Any, Optional, cast

import joblib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator

from mlproject.src.utils.func_utils import flatten_timeseries


class BaseModelWrapper(ABC):
    """
    Generic base wrapper for any ML or DL model.

    Responsibilities:
        - Build / initialize model
        - Fit / train
        - Predict
        - Save / Load model and metadata
        - Ensure model is initialized
    """

    def __init__(self, cfg: Optional[dict] = None):
        if cfg is None:
            self.cfg = DictConfig({})
        elif isinstance(cfg, DictConfig):
            self.cfg = cfg
        elif isinstance(cfg, dict):
            self.cfg = OmegaConf.create(cfg)
        else:
            raise TypeError("cfg must be dict or DictConfig")
        self.model_type: str = ""
        self.model: Optional[Any] = None
        self.n_targets = self.cfg.get("n_targets", 1)

    def get_model(
        self,
    ):
        """
        return model
        """
        return self.model

    @abstractmethod
    def build(self, model_type: str):
        """Build or initialize model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x, **kwargs):
        """Predict using the model."""
        raise NotImplementedError

    def ensure_built(self):
        """Ensure model is initialized before use."""
        if self.model is None:
            raise RuntimeError(
                "Model is not built yet. Call build(input_dim, output_dim)."
            )

    @abstractmethod
    def save(self, save_dir: str):
        """Save model and metadata. To be implemented in subclass."""
        raise NotImplementedError

    @abstractmethod
    def load(self, save_dir: str):
        """Load model and metadata. To be implemented in subclass."""
        raise NotImplementedError


class DLModelWrapperBase(BaseModelWrapper):
    """
    Base class for all deep-learning model wrappers.
    """

    @abstractmethod
    def build(self, model_type: str):
        """Build model architecture."""
        raise NotImplementedError

    def ensure_built(self):
        """Ensure DL model is initialized."""
        if self.model is None:
            raise RuntimeError(
                "Model is not built yet. Call build(input_dim, output_dim)."
            )

    def save(self, save_dir: str):
        """Save model weights + metadata with torch.save."""
        self.ensure_built()
        os.makedirs(save_dir, exist_ok=True)
        assert self.model is not None

        state = {
            "model_state": self.model.state_dict(),
            "model_type": self.model_type,
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }
        torch.save(state, os.path.join(save_dir, "model.pt"))

    def load(self, save_dir: str):
        """Load DL model from disk (weights + metadata)."""
        path = os.path.join(save_dir, "model.pt")
        if not os.path.exists(path):
            raise RuntimeError(f"Model file not found: {path}")

        state = torch.load(path, map_location="cpu")
        self.model_type = state["model_type"]
        assert isinstance(self.model_type, str) and len(self.model_type) > 0
        self.build(self.model_type)
        assert self.model is not None
        self.model.load_state_dict(state["model_state"])
        print(f"[Model Loaded] {path}")
        return self


class MLModelWrapper(BaseModelWrapper):
    """
    Generic ML wrapper for sklearn-style estimators.
    """

    def __init__(
        self,
        cfg: Optional[dict] = None,
        estimator_class: Any = BaseEstimator,
        **estimator_kwargs,
    ):
        """
        Args:
            cfg: configuration dictionary
            estimator_class: sklearn-like estimator
            estimator_kwargs: parameters for estimator init
        """
        super().__init__(cfg)
        self.estimator_class = estimator_class
        self.estimator_kwargs = estimator_kwargs

    def build(self, model_type: str):
        """Initialize estimator."""
        self.model_type = model_type
        self.model = self.estimator_class(**self.estimator_kwargs)

    def fit(
        self,
        x,
        y,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Train model with sklearn-style estimator."""
        if self.model is None:
            self.build(model_type=self.model_type)

        self.ensure_built()

        x_reshaped = flatten_timeseries(x)
        y_reshaped = flatten_timeseries(y)

        model = cast(BaseEstimator, self.model)
        model.fit(x_reshaped, y_reshaped, sample_weight=sample_weight, **kwargs)

    def predict(self, x, **kwargs):
        """Predict with sklearn estimator."""
        self.ensure_built()
        x_reshaped = flatten_timeseries(x)

        out = self.model.predict(x_reshaped, **kwargs)
        if self.n_targets == 1:
            return out
        else:
            return out.reshape(len(out), -1, self.n_targets)

    def save(self, save_dir: str):
        """Save estimator + metadata with joblib."""
        self.ensure_built()
        os.makedirs(save_dir, exist_ok=True)

        state = {
            "model": self.model,
            "model_type": self.model_type,
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }

        joblib.dump(state, os.path.join(save_dir, "model.pkl"))
        print(f"[ML Model Saved] {os.path.join(save_dir, 'model.pkl')}")

    def load(self, save_dir: str):
        """Load estimator + metadata via joblib."""
        path = os.path.join(save_dir, "model.pkl")
        if not os.path.exists(path):
            raise RuntimeError(f"Model file not found: {path}")

        state = joblib.load(path)

        self.model_type = state.get("model_type")

        loaded_cfg = OmegaConf.create(state.get("cfg", {}))
        if not isinstance(self.cfg, DictConfig):
            self.cfg = DictConfig({})
        self.cfg = cast(DictConfig, OmegaConf.merge(self.cfg, loaded_cfg))

        self.model = state["model"]
        self.ensure_built()

        print(f"[ML Model Loaded] {path}")
        return self
