import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ray import serve

from mlproject.src.features.facade import FeatureStoreFacade
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader

# Fix for potential OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


app = FastAPI(title="ML Pipeline API (Ray Serve)")


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]


class FeastPredictRequest(BaseModel):
    entities: List[Union[int, str]] = Field(..., description="List of entity IDs")
    entity_key: Optional[str] = Field(None, description="Key to join entities")
    time_point: str = Field(default="now", description="Time point for retrieval")


class PredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(
        ...,
        description="Input data as dict of columns to values"
    )


class BatchPredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(
        ...,
        description="Input data with multiple rows"
    )
    return_probabilities: bool = Field(default=False)


class MultiPredictResponse(BaseModel):
    predictions: Dict[str, Union[List[List[float]], List[float]]]
    metadata: Optional[Dict[str, Any]] = None


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 0.5}
)
class PreprocessService:
    def __init__(self, config_path: str) -> None:
        print("[PreprocessService] Initializing...")
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.preprocessor = None
        self._load_preprocessor()

    def _load_preprocessor(self) -> None:
        if not self.mlflow_manager.enabled:
            return
        experiment_name = self.cfg.experiment.get("name", "parallel_ensemble_serve")
        print(f"[PreprocessService] Loading preprocessor: "
              f"preprocess (alias: latest)...")
        component = self.mlflow_manager.load_component(
            name=f"{experiment_name}_"
                 f"preprocess",
            alias="latest",
        )
        if component is not None:
            self.preprocessor = component

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is not None:
            return self.preprocessor.transform(data)
        return data

    def check_health(self) -> str:
        if self.preprocessor is not None or not self.mlflow_manager.enabled:
            return "healthy"
        return "initializing"


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1}
)
class ModelService:
    INPUT_CHUNK_LENGTH = 24
    OUTPUT_CHUNK_LENGTH = 6

    def __init__(self, config_path: str) -> None:
        print("[ModelService] Initializing...")
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.models = {}
        self.ready = False
        self.feature_store = None
        self._check_feast()
        self._load_models()

    def _check_feast(self) -> None:
        if "feast://titanic_repo".startswith("feast://"):
            print(f"[ModelService] Initializing Feast Facade...")
            try:
                self.feature_store = FeatureStoreFacade(self.cfg, mode="online")
            except Exception as e:
                print(f"[WARNING] Feast initialization failed: {e}")
                self.feature_store = None

    def get_online_dataset(
        self, entities: List[Union[int, str]], time_point: str = "now"
    ) -> pd.DataFrame:
        if self.feature_store is None:
            raise RuntimeError("Feast feature store not initialized")

        # Retrieve using Facade
        data = self.feature_store.load_features(
            time_point=time_point, entity_ids=entities
        )
        return pd.DataFrame(data)

    def _load_models(self) -> None:
        if not self.mlflow_manager.enabled:
            return
        experiment_name = self.cfg.experiment.get("name", "parallel_ensemble_serve")
        print(f"[ModelService] Loading model: fitted_kmeans_branch "
              f"(alias: latest)...")
        component = self.mlflow_manager.load_component(
            name=f"{experiment_name}_kmeans_branch",
            alias="latest",
        )
        if component is not None:
            self.models["fitted_kmeans_branch"] = component
        print(f"[ModelService] Loading model: fitted_catboost_branch "
              f"(alias: latest)...")
        component = self.mlflow_manager.load_component(
            name=f"{experiment_name}_catboost_branch",
            alias="latest",
        )
        if component is not None:
            self.models["fitted_catboost_branch"] = component
        print(f"[ModelService] Loading model: fitted_xgboost_branch "
              f"(alias: latest)...")
        component = self.mlflow_manager.load_component(
            name=f"{experiment_name}_xgboost_branch",
            alias="latest",
        )
        if component is not None:
            self.models["fitted_xgboost_branch"] = component
        self.ready = True
        print("[ModelService] Ready")

    def check_health(self) -> None:
        if not self.ready:
            raise RuntimeError("ModelService not ready")

    def _prepare_input_tabular(self, features: Any) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            return features.values
        return np.atleast_2d(np.array(features))

    def _prepare_input_timeseries(
        self, features: Any, model_type: str
    ) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            x_input = features.values
        else:
            x_input = np.array(features)
        if model_type == "ml":
            return x_input[:self.INPUT_CHUNK_LENGTH, :].reshape(1, -1)
        return x_input[:self.INPUT_CHUNK_LENGTH, :][np.newaxis, :]

    def predict_tabular_batch(
        self, context: Dict[str, Any], return_probabilities: bool = False
    ) -> Dict[str, Any]:
        self.check_health()
        results = {}
        metadata = {"n_samples": 0, "model_type": "tabular"}
        model = self.models.get("fitted_xgboost_branch")
        if model is not None:
            features = context.get("preprocessed_data")
            if features is not None:
                x_input = self._prepare_input_tabular(features)
                metadata["n_samples"] = len(x_input)
                preds = model.predict(x_input)
                results["xgboost_branch_predictions"] = preds.tolist()
                if return_probabilities and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(x_input)
                        key_prob = "xgboost_branch_predictions_probabilities"
                        results[key_prob] = proba.tolist()
                    except Exception:
                        pass
        model = self.models.get("fitted_catboost_branch")
        if model is not None:
            features = context.get("preprocessed_data")
            if features is not None:
                x_input = self._prepare_input_tabular(features)
                metadata["n_samples"] = len(x_input)
                preds = model.predict(x_input)
                results["catboost_branch_predictions"] = preds.tolist()
                if return_probabilities and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(x_input)
                        key_prob = "catboost_branch_predictions_probabilities"
                        results[key_prob] = proba.tolist()
                    except Exception:
                        pass
        model = self.models.get("fitted_kmeans_branch")
        if model is not None:
            features = context.get("preprocessed_data")
            if features is not None:
                x_input = self._prepare_input_tabular(features)
                metadata["n_samples"] = len(x_input)
                preds = model.predict(x_input)
                results["kmeans_branch_predictions"] = preds.tolist()
                if return_probabilities and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(x_input)
                        key_prob = "kmeans_branch_predictions_probabilities"
                        results[key_prob] = proba.tolist()
                    except Exception:
                        pass
        return {"predictions": results, "metadata": metadata}

    def predict_timeseries_multistep(
        self, context: Dict[str, Any], steps_ahead: int
    ) -> Dict[str, Any]:
        self.check_health()
        results = {}
        n_blocks = (
            steps_ahead + self.OUTPUT_CHUNK_LENGTH - 1
        ) // self.OUTPUT_CHUNK_LENGTH
        metadata = {
            "output_chunk_length": self.OUTPUT_CHUNK_LENGTH,
            "n_blocks": n_blocks,
            "model_type": "timeseries"
        }
        model = self.models.get("fitted_xgboost_branch")
        if model is not None:
            features = context.get("preprocessed_data")
            if features is not None:
                all_predictions = []
                if isinstance(features, pd.DataFrame):
                    current_input = features.copy()
                else:
                    current_input = features
                for block_idx in range(n_blocks):
                    if len(current_input) < self.INPUT_CHUNK_LENGTH:
                        break
                    x_input = self._prepare_input_timeseries(
                        current_input, "ml"
                    )
                    block_preds = model.predict(x_input)
                    all_predictions.append(block_preds[0])
                    if block_idx < n_blocks - 1:
                        if hasattr(current_input, "iloc"):
                            shift = min(
                                self.OUTPUT_CHUNK_LENGTH,
                                len(block_preds)
                            )
                            if isinstance(current_input, pd.DataFrame):
                                current_input = current_input.iloc[shift:]
                all_predictions = np.array(all_predictions)
                if all_predictions.ndim == 1:
                    preds_2d = all_predictions
                else:
                    preds_2d = np.concatenate(all_predictions, axis=0)
                results["xgboost_branch_predictions"] = preds_2d.tolist()
        model = self.models.get("fitted_catboost_branch")
        if model is not None:
            features = context.get("preprocessed_data")
            if features is not None:
                all_predictions = []
                if isinstance(features, pd.DataFrame):
                    current_input = features.copy()
                else:
                    current_input = features
                for block_idx in range(n_blocks):
                    if len(current_input) < self.INPUT_CHUNK_LENGTH:
                        break
                    x_input = self._prepare_input_timeseries(
                        current_input, "ml"
                    )
                    block_preds = model.predict(x_input)
                    all_predictions.append(block_preds[0])
                    if block_idx < n_blocks - 1:
                        if hasattr(current_input, "iloc"):
                            shift = min(
                                self.OUTPUT_CHUNK_LENGTH,
                                len(block_preds)
                            )
                            if isinstance(current_input, pd.DataFrame):
                                current_input = current_input.iloc[shift:]
                all_predictions = np.array(all_predictions)
                if all_predictions.ndim == 1:
                    preds_2d = all_predictions
                else:
                    preds_2d = np.concatenate(all_predictions, axis=0)
                results["catboost_branch_predictions"] = preds_2d.tolist()
        model = self.models.get("fitted_kmeans_branch")
        if model is not None:
            features = context.get("preprocessed_data")
            if features is not None:
                all_predictions = []
                if isinstance(features, pd.DataFrame):
                    current_input = features.copy()
                else:
                    current_input = features
                for block_idx in range(n_blocks):
                    if len(current_input) < self.INPUT_CHUNK_LENGTH:
                        break
                    x_input = self._prepare_input_timeseries(
                        current_input, "ml"
                    )
                    block_preds = model.predict(x_input)
                    all_predictions.append(block_preds[0])
                    if block_idx < n_blocks - 1:
                        if hasattr(current_input, "iloc"):
                            shift = min(
                                self.OUTPUT_CHUNK_LENGTH,
                                len(block_preds)
                            )
                            if isinstance(current_input, pd.DataFrame):
                                current_input = current_input.iloc[shift:]
                all_predictions = np.array(all_predictions)
                if all_predictions.ndim == 1:
                    preds_2d = all_predictions
                else:
                    preds_2d = np.concatenate(all_predictions, axis=0)
                results["kmeans_branch_predictions"] = preds_2d.tolist()
        return {"predictions": results, "metadata": metadata}

    async def run_inference_pipeline(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Run appropriate inference based on data type
        if "tabular" == "timeseries":
            return self.predict_timeseries_multistep(
                context, self.OUTPUT_CHUNK_LENGTH
            )["predictions"]
        return self.predict_tabular_batch(context)["predictions"]


@serve.deployment
@serve.ingress(app)
class ServeAPI:
    DATA_TYPE = "tabular"
    INPUT_CHUNK_LENGTH = 24

    def __init__(self, preprocess_handle: Any, model_handle: Any) -> None:
        self.preprocess_handle = preprocess_handle
        self.model_handle = model_handle
        self.cfg = ConfigLoader.load("mlproject/configs/experiments/feast_tabular.yaml")

    @app.post("/predict", response_model=MultiPredictResponse)
    async def predict(self, request: PredictRequest) -> MultiPredictResponse:
        try:
            df = pd.DataFrame(request.data)
            preprocessed_data = (
                await self.preprocess_handle.preprocess.remote(df)
            )
            context = {"preprocessed_data": preprocessed_data}
            predictions = (
                await self.model_handle.run_inference_pipeline.remote(
                    context
                )
            )
            return MultiPredictResponse(predictions=predictions)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/predict/batch", response_model=MultiPredictResponse)
    async def predict_batch(
        self, request: BatchPredictRequest
    ) -> MultiPredictResponse:
        try:
            df = pd.DataFrame(request.data)
            preprocessed_data = (
                await self.preprocess_handle.preprocess.remote(df)
            )
            context = {"preprocessed_data": preprocessed_data}
            result = await self.model_handle.predict_tabular_batch.remote(
                context, request.return_probabilities
            )
            return MultiPredictResponse(
                predictions=result["predictions"],
                metadata=result["metadata"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/predict/feast", response_model=MultiPredictResponse)
    async def predict_feast(self, request: FeastPredictRequest) -> MultiPredictResponse:
        try:
            df = await self.model_handle.get_online_dataset.remote(
                request.entities, request.time_point
            )
            preprocessed_data = (
                await self.preprocess_handle.preprocess.remote(df)
            )
            context = {"preprocessed_data": preprocessed_data}
            predictions = (
                await self.model_handle.run_inference_pipeline.remote(
                    context
                )
            )
            return MultiPredictResponse(predictions=predictions)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/predict/feast/batch", response_model=MultiPredictResponse)
    async def predict_feast_batch(
        self, request: FeastPredictRequest
    ) -> MultiPredictResponse:
        return await self.predict_feast(request)

    @app.get("/health", response_model=HealthResponse)
    async def health(self) -> HealthResponse:
        try:
            prep_status = await self.preprocess_handle.check_health.remote()
            return HealthResponse(
                status="healthy",
                components={
                    "preprocess": prep_status,
                    "model": "ready"
                }
            )
        except Exception as e:
            return HealthResponse(
                status="unhealthy",
                components={"error": str(e)}
            )


def app_builder(args: Dict[str, str]) -> Any:
    config_path = args.get("config", "mlproject/configs/experiments/feast_tabular.yaml")

    preprocess_deployment = PreprocessService.bind(config_path)
    model_deployment = ModelService.bind(config_path)

    return ServeAPI.bind(preprocess_deployment, model_deployment)


if __name__ == "__main__":
    serve.start(http_options={"host": "0.0.0.0", "port": 8082})
    serve.run(app_builder({}))
