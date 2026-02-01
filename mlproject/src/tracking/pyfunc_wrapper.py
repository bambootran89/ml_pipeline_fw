"""
Universal MLflow PyFunc wrapper module enabling PyFunc packaging for arbitrary
model or preprocessing artifacts.
"""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
from mlflow.pyfunc import PythonModel


class ArtifactPyFuncWrapper(PythonModel):
    """
    Generic MLflow PyFunc wrapper to package arbitrary Python artifacts.

    This wrapper delegates inference to a specified method (e.g., `predict`
    for models or `transform` for preprocessors) and allows retrieval of the
    original unwrapped artifact.
    """

    def __init__(self, artifact: Any, predict_method: str = "predict") -> None:
        """
        Initialize the PyFunc wrapper.

        Args:
            artifact: The underlying Python object to wrap.
            predict_method: Name of the method to invoke during inference.
        """
        self.artifact: Any = artifact
        self.predict_method: str = predict_method

    def load_context(
        self, context: mlflow.pyfunc.PythonModelContext  # type: ignore[name-defined]
    ) -> None:
        """
        Load MLflow context. Reserved for future extension.

        Args:
            context: MLflow model context supplied at model load time.
        """

    def predict_stream(self, *args: Any, **kwargs: Any) -> Any:
        """No-op override to satisfy pylint abstract method requirement."""

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,  # type: ignore[name-defined]
        model_input: Any,
        params: Any = None,
    ) -> Any:
        """
        Delegate inference to the wrapped artifact's specified method.

        Args:
            context: MLflow model context supplied at inference time.
            model_input: Input data passed to the artifact method.

        Returns:
            Output produced by the delegated artifact method.
        """
        _ = context
        _ = params
        method = getattr(self.artifact, self.predict_method, None)
        if method is None:
            raise AttributeError(
                f"Artifact method '{self.predict_method}' not found on wrapped object."
            )
        return method(model_input)

    def get_raw_artifact(self) -> Any:
        """
        Retrieve the original unwrapped artifact.

        Returns:
            The underlying Python artifact originally supplied to the wrapper.
        """
        return self.artifact
