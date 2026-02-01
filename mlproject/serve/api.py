"""API module for serving time series forecasts with Feast integration.

Provides endpoints:
- GET /health - Health check with Feast status
- GET / - Root status message
- POST /predict - Traditional data-in-payload prediction
- POST /predict/feast - Feast-native single/multi-entity prediction
- POST /predict/feast/batch - Batch prediction for multiple entities
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Union

import psutil
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mlproject.serve.models_service import ModelsService
from mlproject.serve.schemas import (
    BatchPredictResponse,
    FeastBatchPredictRequest,
    FeastPredictRequest,
    PredictRequest,
    PredictResponse,
)
from mlproject.src.utils.func_utils import get_env_path

ARTIFACTS_DIR: str = get_env_path(
    "ARTIFACTS_DIR",
    "mlproject/artifacts/models",
).as_posix()

CONFIG_PATH: str = get_env_path(
    "CONFIG_PATH",
    "mlproject/configs/experiments/etth1_feast.yaml",
).as_posix()

# Initialize FastAPI app
app = FastAPI(
    title="TS Forecasting API with Feast",
    version="2.0.0",
    description="Time series forecasting with Feast feature store",
)

# Initialize service once at startup
# All logic (model, preprocessing, Feast) is in ModelsService
service = ModelsService(cfg_path=CONFIG_PATH)


def _check_mlflow_connection() -> bool:
    """Ping MLflow server."""
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not mlflow_uri:
            return False
        response = requests.get(f"{mlflow_uri}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def _check_feast_connection() -> bool:
    """Check Feast facade availability."""
    try:
        return service.feast_facade is not None
    except Exception:
        return False


class HealthResponse(BaseModel):
    """
    Response schema for the health check endpoint.

    Attributes
    ----------
    status : str
        Overall service status ("healthy", "degraded", "unhealthy").
    timestamp : datetime
        UTC timestamp when health check was generated.
    model_loaded : bool
        Whether ML model is loaded and ready.
    memory_usage_mb : float
        Current memory usage in megabytes.
    dependencies : dict
        Health status of external dependencies.
    """

    status: str
    timestamp: datetime
    model_loaded: bool
    memory_usage_mb: float
    dependencies: dict


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    K8s liveness/readiness probe endpoint.

    Returns 200 if:
    - Model loaded
    - Memory < 90% threshold
    - MLflow connection alive
    - Feast facade available
    """
    checks = {
        "model_loaded": service.model is not None,
        "mlflow_reachable": _check_mlflow_connection(),
        "feast_available": _check_feast_connection(),
        "disk_space_ok": psutil.disk_usage("/").percent < 90,
    }

    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    all_healthy = all(checks.values()) and memory_mb < 2000

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.utcnow(),
        model_loaded=checks["model_loaded"],
        memory_usage_mb=round(memory_mb, 2),
        dependencies=checks,
    )


@app.get("/")
def root() -> Dict[str, Any]:
    """
    Root endpoint with status message.

    Returns
    -------
    Dict[str, any]
        Simple status message and available endpoints.
    """
    return {
        "message": "TimeSeries Forecasting API with Feast is running",
        "version": "2.0.0",
        "endpoints": [
            "GET /health",
            "GET /",
            "POST /predict",
            "POST /predict/feast",
            "POST /predict/feast/batch",
        ],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> Dict[str, List[float]]:
    """
    Traditional prediction with data in payload.

    Maintains backward compatibility with existing clients.
    All logic delegated to ModelsService.predict().

    Parameters
    ----------
    request : PredictRequest
        Request containing feature data.

    Returns
    -------
    Dict[str, List[float]]
        Dictionary with 'prediction' key.

    Raises
    ------
    HTTPException
        400 for invalid input, 500 for server errors.
    """
    try:
        # Delegate to service
        result = service.predict(request)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}",
        ) from e


@app.post("/predict/feast", response_model=PredictResponse)
def predict_feast(
    request: FeastPredictRequest,
) -> Dict[str, List[float]]:
    """
    Feast-native prediction for single or few entities.

    The API automatically fetches features from Feast.
    All logic delegated to ModelsService.predict_feast().

    Parameters
    ----------
    request : FeastPredictRequest
        Request with time_point and optional entities.

    Returns
    -------
    Dict[str, List[float]]
        Dictionary with 'prediction' key.

    Raises
    ------
    HTTPException
        400 for invalid input, 500 for server errors.

    Examples
    --------
    Single entity:
    >>> POST /predict/feast
    >>> {
    ...     "time_point": "2024-01-01T12:00:00",
    ...     "entities": [1]
    ... }

    Multiple entities (returns first entity):
    >>> POST /predict/feast
    >>> {
    ...     "time_point": "now",
    ...     "entities": [1, 2, 3]
    ... }
    """
    try:
        # Delegate to service
        result = service.predict_feast(request)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feast prediction failed: {str(e)}",
        ) from e


@app.post("/predict/feast/batch", response_model=BatchPredictResponse)
def predict_feast_batch(
    request: FeastBatchPredictRequest,
) -> Dict[str, Dict[Union[int, str], List[float]]]:
    """
    Batch prediction for multiple entities from Feast.

    Optimized endpoint that fetches all entities in a single
    Feast query, then predicts for each entity.
    All logic delegated to ModelsService.predict_feast_batch().

    Parameters
    ----------
    request : FeastBatchPredictRequest
        Request with time_point and entities list.

    Returns
    -------
    Dict[str, Dict[Union[int, str], List[float]]]
        Dictionary with 'predictions' key containing results
        grouped by entity ID.

    Raises
    ------
    HTTPException
        400 for invalid input, 500 for server errors.

    Examples
    --------
    >>> POST /predict/feast/batch
    >>> {
    ...     "time_point": "now",
    ...     "entities": [1, 2, 3, 4, 5],
    ...     "entity_key": "location_id"
    ... }
    >>>
    >>> # Response:
    >>> {
    ...     "predictions": {
    ...         "1": [25.5, 26.0, 24.8],
    ...         "2": [22.3, 23.1, 22.7],
    ...         "3": [28.9, 29.5, 28.3],
    ...         "4": [24.1, 24.6, 24.0],
    ...         "5": [26.7, 27.2, 26.5]
    ...     }
    ... }
    """
    try:
        # Delegate to service
        result = service.predict_feast_batch(request)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        ) from e


if __name__ == "__main__":
    uvicorn.run(
        "mlproject.serve.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
