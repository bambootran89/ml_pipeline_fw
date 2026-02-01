# Auto-generated FastAPI serve for parallel_ensemble_serve
# Generated from serve configuration.
# Supports: Tabular batch prediction

import os
import platform

if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "1"

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlproject.src.features.facade import FeatureStoreFacade
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader

app = FastAPI(
    title="parallel_ensemble_serve API",
    version="1.0.0",
    description="Auto-generated serve API for tabular data",
)


# Request/Response Schemas

class FeastPredictRequest(BaseModel):
    entities: List[Union[int, str]] = Field(..., description="List of entity IDs")
    entity_key: Optional[str] = Field(None, description="Key to join entities")
    time_point: str = Field(default="now", description="Time point for retrieval")


class PredictRequest(BaseModel):
    data: Dict[str, List[Any]
               ] = Field(..., description="Input data as dict of columns to values")


class BatchPredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(..., description="Input data with multiple rows")
    return_probabilities: bool = Field(
        default=False,
        description="Return prediction probabilities if available"
    )


class MultiStepPredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(..., description="Input timeseries data")
    steps_ahead: int = Field(
        default=6,
        description="Number of steps to predict ahead",
        ge=1
    )


class PredictResponse(BaseModel):
    predictions: Dict[str, List[float]]
    metadata: Optional[Dict[str, Any]] = None


class MultiPredictResponse(BaseModel):
    predictions: Dict[str, Union[List[List[float]], List[float]]]
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_type: str = "tabular"
    features: List[str] = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']


# Service Implementation

class ServeService:
    DATA_TYPE = "tabular"
    INPUT_CHUNK_LENGTH = 24
    OUTPUT_CHUNK_LENGTH = 6
    FEATURES = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

    def __init__(self, config_path: str) -> None:
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.preprocessor = None
        self.models = {}
        self.feature_store = None

        if self.mlflow_manager.enabled:
            experiment_name = self.cfg.experiment.get("name", "parallel_ensemble_serve")

            print(f"[ModelService] Initializing Feast Facade...")
            try:
                self.feature_store = FeatureStoreFacade(self.cfg, mode="online")
            except Exception as e:
                print(f"[WARNING] Feast initialization failed: {e}")
                self.feature_store = None

            print(f"[ModelService] Loading preprocessor: preprocess "
                  f"(alias: latest)...")
            component = self.mlflow_manager.load_component(
                name=f"{experiment_name}_preprocess",
                alias="latest",
            )
            if component is not None:
                self.preprocessor = component

            print(
                f"[ModelService] Loading model: fitted_xgboost_branch from xgboost_branch "
                f"(alias: latest)...")
            component = self.mlflow_manager.load_component(
                name=f"{experiment_name}_xgboost_branch",
                alias="latest",
            )
            if component is not None:
                self.models["fitted_xgboost_branch"] = component

            print(
                f"[ModelService] Loading model: fitted_kmeans_branch from kmeans_branch "
                f"(alias: latest)...")
            component = self.mlflow_manager.load_component(
                name=f"{experiment_name}_kmeans_branch",
                alias="latest",
            )
            if component is not None:
                self.models["fitted_kmeans_branch"] = component

            print(
                f"[ModelService] Loading model: fitted_catboost_branch from catboost_branch "
                f"(alias: latest)...")
            component = self.mlflow_manager.load_component(
                name=f"{experiment_name}_catboost_branch",
                alias="latest",
            )
            if component is not None:
                self.models["fitted_catboost_branch"] = component

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            return data
        return self.preprocessor.transform(data)

    def _prepare_input_tabular(self, features: Any) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            return features.values
        return np.atleast_2d(np.array(features))

    def get_online_dataset(self,
                           entities: List[Union[int, str]],
                           time_point: str = "now") -> pd.DataFrame:
        if self.feature_store is None:
            raise RuntimeError("Feast not initialized")

        # Use Facade to load features (handles windowing and prefixes)
        print(f"[ModelService] Fetching features for entities: {entities}")
        df = self.feature_store.load_features(
            time_point=time_point,
            entity_ids=entities
        )
        return df

    def _prepare_input_timeseries(self, features: Any, model_type: str) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            x_input = features.values
        else:
            x_input = np.array(features)

        if model_type == "ml":
            x_input = x_input[:self.INPUT_CHUNK_LENGTH, :]
            x_input = x_input.reshape(1, -1)
        else:
            x_input = x_input[:self.INPUT_CHUNK_LENGTH, :]
            x_input = x_input[np.newaxis, :]
        return x_input

    def predict_tabular_batch(
        self,
        context: Dict[str, Any],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
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
                        key = "xgboost_branch_predictions_probabilities"
                        results[key] = proba.tolist()
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
                        key = "catboost_branch_predictions_probabilities"
                        results[key] = proba.tolist()
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
                        key = "kmeans_branch_predictions_probabilities"
                        results[key] = proba.tolist()
                    except Exception:
                        pass

        return {"predictions": results, "metadata": metadata}

    def predict_timeseries_multistep(
        self,
        context: Dict[str, Any],
        steps_ahead: int
    ) -> Dict[str, Any]:
        results = {}
        n_blocks = (steps_ahead + self.OUTPUT_CHUNK_LENGTH - 1)
        n_blocks = n_blocks // self.OUTPUT_CHUNK_LENGTH
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
                        current_input,
                        "ml"
                    )
                    block_preds = model.predict(x_input)
                    all_predictions.append(block_preds[0])
                    if block_idx < n_blocks - 1 and hasattr(current_input, "iloc"):
                        shift = min(self.OUTPUT_CHUNK_LENGTH, len(block_preds))
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
                        current_input,
                        "ml"
                    )
                    block_preds = model.predict(x_input)
                    all_predictions.append(block_preds[0])
                    if block_idx < n_blocks - 1 and hasattr(current_input, "iloc"):
                        shift = min(self.OUTPUT_CHUNK_LENGTH, len(block_preds))
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
                        current_input,
                        "ml"
                    )
                    block_preds = model.predict(x_input)
                    all_predictions.append(block_preds[0])
                    if block_idx < n_blocks - 1 and hasattr(current_input, "iloc"):
                        shift = min(self.OUTPUT_CHUNK_LENGTH, len(block_preds))
                        if isinstance(current_input, pd.DataFrame):
                            current_input = current_input.iloc[shift:]
                all_predictions = np.array(all_predictions)
                if all_predictions.ndim == 1:
                    preds_2d = all_predictions
                else:
                    preds_2d = np.concatenate(all_predictions, axis=0)
                results["kmeans_branch_predictions"] = preds_2d.tolist()

        return {"predictions": results, "metadata": metadata}

    def run_inference_pipeline(self, context: Dict[str, Any]) -> Dict[str, List[float]]:
        if self.DATA_TYPE == "tabular":
            result = self.predict_tabular_batch(context)
        else:
            result = self.predict_timeseries_multistep(
                context,
                steps_ahead=self.OUTPUT_CHUNK_LENGTH
            )
        return result.get("predictions", {})


service = ServeService("mlproject/configs/experiments/feast_tabular.yaml")


# API Endpoints

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    model_loaded = len(service.models) > 0
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        data_type=service.DATA_TYPE,
        features=service.FEATURES,
    )


@app.post("/predict", response_model=MultiPredictResponse)
def predict(request: PredictRequest) -> MultiPredictResponse:
    try:
        df = pd.DataFrame(request.data)
        preprocessed_data = service.preprocess(df)
        context = {"preprocessed_data": preprocessed_data}
        predictions = service.run_inference_pipeline(context)
        return MultiPredictResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/feast", response_model=MultiPredictResponse)
def predict_feast(request: FeastPredictRequest) -> MultiPredictResponse:
    try:
        df = service.get_online_dataset(request.entities)
        preprocessed_data = service.preprocess(df)
        context = {"preprocessed_data": preprocessed_data}
        predictions = service.run_inference_pipeline(context)
        return MultiPredictResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/feast/batch", response_model=MultiPredictResponse)
def predict_feast_batch(request: FeastPredictRequest) -> MultiPredictResponse:
    return predict_feast(request)


@app.post("/predict/batch", response_model=MultiPredictResponse)
def predict_batch(request: BatchPredictRequest) -> MultiPredictResponse:
    try:
        df = pd.DataFrame(request.data)
        preprocessed_data = service.preprocess(df)
        context = {"preprocessed_data": preprocessed_data}
        result = service.predict_tabular_batch(
            context,
            return_probabilities=request.return_probabilities
        )
        return MultiPredictResponse(
            predictions=result["predictions"],
            metadata=result["metadata"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
