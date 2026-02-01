"""
Ray Serve deployment with Feast integration for model inference.

This module provides:
1. Traditional data-in-payload prediction
2. Feast-native single/multi-entity prediction
3. Batch prediction for multiple entities
"""

from __future__ import annotations

import asyncio
import os
import signal
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import ray
from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig, OmegaConf
from ray import serve

from mlproject.serve.schemas import (
    BatchPredictResponse,
    FeastBatchPredictRequest,
    FeastPredictRequest,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from mlproject.src.features.facade import FeatureStoreFacade
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.preprocess.transform_manager import TransformManager
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader
from mlproject.src.utils.func_utils import get_env_path

ARTIFACTS_DIR: str = get_env_path(
    "ARTIFACTS_DIR",
    "mlproject/artifacts/models",
).as_posix()

CONFIG_PATH: str = get_env_path(
    "CONFIG_PATH",
    "mlproject/configs/experiments/etth1_feast.yaml",
).as_posix()

app = FastAPI(
    title="mlproject Ray Serve Inference API with Feast",
    version="2.0.0",
    description="Model inference with Feast feature store integration",
)


@serve.deployment(
    health_check_period_s=10,
    health_check_timeout_s=30,
    num_replicas=int(os.getenv("RAY_FEAST_REPLICAS", "2")),
    ray_actor_options={"num_cpus": 0.5},
)
class FeastService:
    """
    Ray Serve deployment for Feast feature retrieval.

    Handles fetching features from Feast for single or multiple
    entities at specified time points.

    Attributes
    ----------
    cfg : DictConfig
        Experiment configuration.
    facade : FeatureStoreFacade
        Facade for Feast operations.
    ready : bool
        Service readiness status.
    """

    cfg: DictConfig
    facade: FeatureStoreFacade
    ready: bool

    def __init__(self) -> None:
        """Initialize FeastService with configuration and facade."""
        print("[FeastService] Initializing...")
        self.cfg = ConfigLoader.load(CONFIG_PATH)
        self.facade = FeatureStoreFacade(self.cfg, mode="online")
        self.ready = True
        print("[FeastService] Ready")

    def check_health(self) -> None:
        """Health check by Ray."""
        if not self.ready:
            raise RuntimeError("FeastService not ready")

    async def fetch_features(
        self,
        time_point: str = "now",
        entities: Optional[List[Union[int, str]]] = None,
        entity_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch features from Feast for entities at time_point.

        Parameters
        ----------
        time_point : str, default="now"
            Reference time. Can be "now" or ISO datetime string.
        entities : Optional[List[Union[int, str]]], default=None
            Entity IDs to query. If None, uses config default.
        entity_key : Optional[str], default=None
            Entity key name. If None, uses config default.

        Returns
        -------
        pd.DataFrame
            Features for requested entities and time point.

        Raises
        ------
        RuntimeError
            If Feast query fails.
        """
        try:
            # Fetch features (run in executor to not block event loop)
            _ = entity_key
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                None,
                self.facade.load_features,
                time_point,
                entities,
                # entity_key
            )

            print(
                f"[FeastService] Fetched {len(df)} rows for "
                f"{len(entities) if entities else 1} entities"
            )
            return df

        except Exception as e:
            print(f"[FeastService] Error fetching features: {e}")
            raise RuntimeError(f"Feast query failed: {e}") from e

    def is_available(self) -> bool:
        """Return whether Feast service is available."""
        return self.ready


@serve.deployment(
    health_check_period_s=10,
    health_check_timeout_s=30,
    num_replicas=int(os.getenv("RAY_PREPROCESS_REPLICAS", "2")),
    ray_actor_options={"num_cpus": 0.5},
)
class PreprocessingService:
    """
    Ray Serve deployment for input preprocessing.

    Attributes
    ----------
    cfg : DictConfig
        Experiment configuration.
    mlflow_manager : MLflowManager
        MLflow manager instance.
    preprocessor : Optional[Any]
        Either a PyFunc preprocessor or TransformManager.
    model_handle : Any
        Handle to ModelService for warmup.
    ready : bool
        Indicates whether preprocessing artifacts are loaded.
    """

    cfg: DictConfig
    mlflow_manager: MLflowManager
    preprocessor: Optional[Any]
    model_handle: Any
    ready: bool

    def __init__(self, model_handle: Any) -> None:
        """
        Initialize PreprocessingService and MLflow manager.

        Parameters
        ----------
        model_handle : Any
            Ray Serve handle to ModelService for warmup.
        """
        self.cfg = ConfigLoader.load(CONFIG_PATH)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.preprocessor = None
        self.model_handle = model_handle
        self.ready = False

        # Trigger async warmup
        asyncio.create_task(self._warmup())

    async def _warmup(self) -> None:
        """
        Lazily initialize preprocessing artifacts.

        Priority:
        1. MLflow companion preprocessor (if available)
        2. Local TransformManager fallback
        """
        if self.ready:
            return

        if self.mlflow_manager.enabled:
            experiment_name: str = self.cfg.experiment["name"]
            self.preprocessor = self.mlflow_manager.load_component(
                name=f"{experiment_name}_preprocessor", alias="production"
            )

        if self.preprocessor is None:
            # Fallback: Load local TransformManager
            self.preprocessor = TransformManager(
                self.cfg,
                artifacts_dir=self.cfg.training.artifacts_dir,
            )
            self.preprocessor.load(cfg=self.cfg)
            print("[PreprocessingService] Loaded local TransformManager")

        self.ready = True
        print("[PreprocessingService] Ready")

    def check_health(self) -> None:
        """Health check by Ray - allow warmup phase."""
        if not self.ready:
            print("[PreprocessingService] Still warming up...")

    async def preprocess(
        self, data: Union[Dict[str, List[Any]], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Transform raw input into feature DataFrame.

        Parameters
        ----------
        data : Union[Dict[str, List[Any]], pd.DataFrame]
            Raw input from request or Feast.

        Returns
        -------
        pd.DataFrame
            Transformed feature DataFrame.

        Raises
        ------
        RuntimeError
            If service not ready or transformation fails.
        """
        if not self.ready:
            raise RuntimeError("PreprocessingService temporarily unavailable")

        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Set index if date column exists
        if "date" in df.columns:
            df = df.set_index("date")

        # Transform
        loop = asyncio.get_running_loop()
        assert self.preprocessor is not None
        return await loop.run_in_executor(None, self.preprocessor.transform, df)


@serve.deployment(health_check_period_s=10, health_check_timeout_s=30)
class ModelService:
    """
    Ray Serve deployment for model inference.

    (Same as original - no changes needed)
    """

    cfg: DictConfig
    mlflow_manager: MLflowManager
    model: Optional[Any]
    model_loaded: bool
    run_id: Optional[str]
    input_chunk_length: int

    def __init__(self) -> None:
        """Initialize ModelService, load MLflow manager and model."""
        print("[ModelService] Initializing...")
        self.cfg = ConfigLoader.load(CONFIG_PATH)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.model = None
        self.model_loaded = False
        self.run_id = None
        self.input_chunk_length = self._load_input_chunk_length()
        self._load_model()
        print(
            f"[ModelService] Initialization Complete. " f"Loaded: {self.model_loaded}"
        )

    def check_health(self) -> None:
        """Health check by Ray."""
        if not self.model_loaded:
            raise RuntimeError("Model artifacts not loaded yet.")

    def _load_input_chunk_length(self) -> int:
        """Load input_chunk_length from experiment config."""
        if hasattr(self.cfg, "experiment") and hasattr(
            self.cfg.experiment, "hyperparams"
        ):
            return int(self.cfg.experiment.hyperparams.get("input_chunk_length", 24))
        return 24

    def _load_model(self) -> None:
        """Load model from MLflow or fallback to local artifacts."""
        if self.mlflow_manager.enabled:
            experiment_name: str = self.cfg.experiment["name"]
            self.model = self.mlflow_manager.load_component(
                name=f"{experiment_name}_model", alias="production"
            )
            if self.model is not None:
                self.model_loaded = True
            return

        self._load_local_model()

    def _load_local_model(self) -> None:
        """Load model from local artifacts directory."""
        approach = self.cfg.approaches[0]
        name = approach["model"].lower()
        hyperparams = approach.get("hyperparams", {})
        if isinstance(hyperparams, DictConfig):
            hyperparams = OmegaConf.to_container(hyperparams, resolve=True)
        self.model = ModelFactory.load(name, hyperparams, ARTIFACTS_DIR)
        self.model_loaded = True

    def prepare_input(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare model input from preprocessed DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if len(df) < self.input_chunk_length:
            raise ValueError(
                f"Need at least {self.input_chunk_length} rows, " f"got {len(df)}"
            )
        return df.values[-self.input_chunk_length :][np.newaxis, :]

    def predict_prepared(self, x_input: np.ndarray) -> List[float]:
        """Run prediction on prepared input array."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if not hasattr(self.model, "predict"):
            raise RuntimeError("Model has no predict method")

        preds = self.model.predict(x_input.astype(np.float32))
        return np.asarray(preds).flatten().tolist()

    def is_loaded(self) -> bool:
        """Return whether the model is loaded."""
        return self.model_loaded

    def get_run_id(self) -> Optional[str]:
        """Return MLflow run ID."""
        return self.run_id

    def get_model(self) -> Optional[Any]:
        """Return underlying model instance."""
        return self.model


@serve.deployment(num_replicas=2)
class PostprocessingService:
    """Deployment actor responsible for sanitizing model outputs.

    Supports both list-based predictions and dictionary-based batch outputs where
    each value is a list of predictions. Ensures numeric stability by replacing NaN
    and infinite values with safe defaults.
    """

    def _sanitize_array(self, preds: Any) -> List[float]:
        """Sanitize a single array of predictions.

        Args:
            preds: Raw model predictions convertible to a NumPy array.

        Returns:
            A flattened list of finite float values with NaN and infinite values
            replaced by 0.0.
        """
        arr = np.asarray(preds, dtype=np.float32).flatten()
        clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return clean.tolist()

    async def format_output(
        self, data: Union[List[float], Dict[Any, List[float]]]
    ) -> Union[List[float], Dict[Any, List[float]]]:
        """Format and sanitize model output.

        If the input is a dictionary, each value (list of predictions) is sanitized.
        If the input is a list, it is sanitized directly.

        Args:
            data: Either a list of predictions or a dictionary mapping entity IDs to
                lists of predictions.

        Returns:
            Sanitized predictions in the same structure as the input.
        """
        if isinstance(data, dict):
            return {key: self._sanitize_array(value) for key, value in data.items()}
        return self._sanitize_array(data)


@serve.deployment
@serve.ingress(app)
class ForecastAPI:
    """
    Ray Serve HTTP API with Feast integration.

    Provides three prediction modes:
    1. /predict - Traditional data-in-payload
    2. /predict/feast - Feast-native single/multi-entity
    3. /predict/feast/batch - Optimized batch prediction

    Attributes
    ----------
    preprocess_handle : Any
        Ray actor handle for PreprocessingService.
    model_handle : Any
        Ray actor handle for ModelService.
    feast_handle : Any
        Ray actor handle for FeastService.
    """

    preprocess_handle: Any
    model_handle: Any
    feast_handle: Any

    def __init__(
        self,
        preprocess_handle: Any,
        model_handle: Any,
        feast_handle: Any,
        postprocess_handle: Any,
    ) -> None:
        """
        Initialize ForecastAPI with service handles.

        Parameters
        ----------
        preprocess_handle : Any
            Preprocessing Ray actor handle.
        model_handle : Any
            Model Ray actor handle.
        feast_handle : Any
            Feast Ray actor handle.
        """
        self.preprocess_handle = preprocess_handle
        self.model_handle = model_handle
        self.feast_handle = feast_handle
        self.postprocess_handle = postprocess_handle
        self.cfg = ConfigLoader.load(CONFIG_PATH)
        self.default_entity_key = self.cfg.data.get("entity_key", "location_id")
        self.input_chunk_length = int(
            self.cfg.experiment.hyperparams.get("input_chunk_length", 24)
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(self, req: PredictRequest) -> Dict[str, List[float]]:
        """
        Traditional prediction with data in payload.

        This endpoint maintains backward compatibility with existing
        clients that provide feature data directly in the request.

        Parameters
        ----------
        req : PredictRequest
            Request containing feature data.

        Returns
        -------
        Dict[str, List[float]]
            Dictionary with 'prediction' key containing results.

        Examples
        --------
        >>> POST /predict
        >>> {
        ...     "data": {
        ...         "temp": [25.5, 26.0, ...],
        ...         "humidity": [60.0, 61.5, ...]
        ...     }
        ... }
        """
        try:
            df: pd.DataFrame = await self.preprocess_handle.preprocess.remote(req.data)
            x_input = await self.model_handle.prepare_input.remote(df)
            preds = await self.model_handle.predict_prepared.remote(x_input)
            return {"prediction": preds}

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {exc}"
            ) from exc

    @app.post("/predict/feast", response_model=PredictResponse)
    async def predict_feast(self, req: FeastPredictRequest) -> PredictResponse:
        """
        Feast-native prediction for single or few entities.

        The API automatically fetches features from Feast based on
        time_point and entities parameters.

        Parameters
        ----------
        req : FeastPredictRequest
            Request with time_point and optional entities.

        Returns
        -------
        Dict[str, List[float]]
            Dictionary with 'prediction' key containing results.

        Examples
        --------
        Single entity:
        >>> POST /predict/feast
        >>> {
        ...     "time_point": "2024-01-01T12:00:00",
        ...     "entities": [1]
        ... }

        Multiple entities (returns first entity's prediction):
        >>> POST /predict/feast
        >>> {
        ...     "time_point": "now",
        ...     "entities": [1, 2, 3]
        ... }
        """
        try:
            entity_key = req.entity_key or self.default_entity_key
            # 1. Fetch features from Feast
            df: pd.DataFrame = await self.feast_handle.fetch_features.remote(
                time_point=req.time_point,
                entities=req.entities,
                entity_key=entity_key,
            )

            # 2. For multi-entity, take first entity only

            if req.entities and len(req.entities) > 1:
                if entity_key in df.columns:
                    first_entity = req.entities[0]
                    df = df[df[entity_key] == first_entity]

            # 3. Preprocess
            df_transformed: pd.DataFrame = (
                await self.preprocess_handle.preprocess.remote(df)
            )
            if entity_key in df_transformed.columns:
                del df_transformed[entity_key]

            # 4. Predict
            x_input = await self.model_handle.prepare_input.remote(df_transformed)
            preds = await self.model_handle.predict_prepared.remote(x_input)
            # Sanitize JSON-unsafe values
            clean_preds = await self.postprocess_handle.format_output.remote(preds)
            return PredictResponse(prediction=clean_preds)

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Feast prediction failed: {exc}",
            ) from exc

    @app.post("/predict/feast/batch", response_model=BatchPredictResponse)
    async def predict_feast_batch(
        self, req: FeastBatchPredictRequest
    ) -> BatchPredictResponse:
        """
        Batch prediction for multiple entities from Feast.

        Optimized endpoint that fetches all entities in a single
        Feast query, then predicts for each entity in parallel.

        Parameters
        ----------
        req : FeastBatchPredictRequest
            Request with time_point and entities list.

        Returns
        -------
        Dict[str, Dict[Union[int, str], List[float]]]
            Dictionary with 'predictions' key containing results
            grouped by entity ID.

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
            # 1. Fetch features for ALL entities (single Feast query)
            df_all: pd.DataFrame = await self.feast_handle.fetch_features.remote(
                time_point=req.time_point,
                entities=req.entities,
                entity_key=req.entity_key,
            )

            # 2. Run inference chain for each valid entity
            entity_key = req.entity_key or self.default_entity_key
            tasks: List[asyncio.Task[List[float]]] = []
            valid_entity_ids: List[Union[int, str]] = []
            grouped = df_all.groupby(entity_key)

            for entity_id in req.entities:
                # Filter rows belonging to this entity
                if entity_id in grouped.groups:
                    entity_df = grouped.get_group(entity_id).copy()

                    tasks.append(
                        asyncio.create_task(
                            self._run_inference_chain(entity_df, entity_key)
                        )
                    )
                    valid_entity_ids.append(entity_id)
                else:
                    print(
                        f"[WARNING] Entity {entity_id} \
                            not found in Feast result, skipping."
                    )

            if not tasks:
                return BatchPredictResponse(predictions={})

            # 3. Collect predictions in parallel
            batch_preds = await asyncio.gather(*tasks, return_exceptions=True)

            # 4. Build response dictionary (entity_id → predictions)
            predictions: Dict[Union[str, int], List[float]] = {
                str(eid): preds
                for eid, preds in zip(valid_entity_ids, batch_preds)
                if isinstance(preds, list)
            }

            # 5. Normalize JSON-unsafe values (NaN, Inf)
            sanitized_dict = await self.postprocess_handle.format_output.remote(
                predictions
            )

            return BatchPredictResponse(predictions=sanitized_dict)

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"[ERROR] Batch prediction failed: {exc}",
            ) from exc

    async def _run_inference_chain(self, df: pd.DataFrame, key: str) -> List[float]:
        """Run feature preprocessing and model inference for one entity."""
        try:
            # 1. Preprocess features
            df_transformed: pd.DataFrame = (
                await self.preprocess_handle.preprocess.remote(df)
            )

            # 2. Drop entity key column if present
            if key in df_transformed.columns:
                df_transformed = df_transformed.drop(columns=[key])

            if len(df_transformed) < self.input_chunk_length:
                raise ValueError(
                    f"Insufficient data: need \
                        {self.input_chunk_length}, got {len(df_transformed)}"
                )
            # 3. Prepare model input
            x_input: np.ndarray = await self.model_handle.prepare_input.remote(
                df_transformed
            )

            # 4. Run model inference
            preds: List[float] = await self.model_handle.predict_prepared.remote(
                x_input
            )

            return preds

        except ValueError as exc:
            print(f"[InferenceChain] Value error: {exc}")
            raise

        except Exception as exc:
            print(f"[InferenceChain] Unexpected error: {exc}")
            raise

    @app.get("/health", response_model=HealthResponse)
    async def health(self) -> Dict[str, bool]:
        """
        Health check endpoint.

        Returns
        -------
        Dict[str, bool]
            Dictionary with status indicators.
        """
        try:
            model_loaded = await self.model_handle.is_loaded.remote()
            feast_available = await self.feast_handle.is_available.remote()
        except Exception:
            model_loaded = False
            feast_available = False

        return {
            "status": True,
            "model_loaded": model_loaded,
            "feast_available": feast_available,
        }


def main() -> None:
    """
    Start Ray Serve application with Feast integration.

    Initializes Ray, starts Serve, and deploys all services.
    """
    # Start Ray with dashboard
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
    )
    serve.start(detached=True)

    # Bind services
    # pylint: disable=no-member
    model = ModelService.bind()  # type: ignore[attr-defined]
    # pylint: disable=no-member
    preprocess = PreprocessingService.bind(model)  # type: ignore[attr-defined]
    # pylint: disable=no-member
    feast = FeastService.bind()  # type: ignore[attr-defined]
    # pylint: disable=no-member
    postprocess = PostprocessingService.bind()  # type: ignore[attr-defined]
    # Deploy API with all 3 handles
    # pylint: disable=no-member
    serve.run(
        ForecastAPI.bind(  # type: ignore[attr-defined]
            preprocess, model, feast, postprocess
        ),
        route_prefix="/",
    )

    print("[Ray Serve] API ready at http://localhost:8000")
    print("[Ray Serve] Endpoints:")
    print("  - POST /predict (traditional)")
    print("  - POST /predict/feast (Feast single/multi)")
    print("  - POST /predict/feast/batch (Feast batch)")
    print("  - GET /health")
    print("[Ray Serve] Dashboard at http://localhost:8265")
    print("[Ray Serve] Press Ctrl+C to stop")

    # Keep script running
    try:
        signal.pause()
    except KeyboardInterrupt:
        print("\n[Ray Serve] Shutting down...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
