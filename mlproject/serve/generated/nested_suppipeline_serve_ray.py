import os
import platform
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ray import serve

# Fix for potential OpenMP conflict on macOS
if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "1"

from mlproject.src.features.facade import FeatureStoreFacade
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader

app = FastAPI(title="ML Pipeline API (Ray Serve)")


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]


class FeastPredictRequest(BaseModel):
    entities: List[Union[int, str]] = Field(
        ..., description="List of entity IDs"
    )
    entity_key: Optional[str] = Field(
        None, description="Key to join entities"
    )
    time_point: str = Field(
        default="now", description="Time point for retrieval"
    )


class PredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(
        ..., description="Input data as dict of columns to values"
    )


class BatchPredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(
        ..., description="Input data with multiple rows"
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
        experiment_name = self.cfg.experiment.get(
            "name", "nested_suppipeline_serve"
        )
        print(
            f"[PreprocessService] Loading preprocessor: "
            f"preprocess (alias: production)..."
        )
        component = self.mlflow_manager.load_component(
            name=f"{experiment_name}_"
                 f"preprocess",
            alias="production",
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
    ADDITIONAL_FEATURE_KEYS = []

    def __init__(self, config_path: str) -> None:
        print("[ModelService] Initializing...")
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.models = {}
        self.feature_generators = {}
        self.ready = False
        self.feature_store = None
        self._check_feast()
        self._load_models()

    def _check_feast(self) -> None:
        if "mlproject/data/ETTh1.csv".startswith("feast://"):
            print(f"[ModelService] Initializing Feast Facade...")
            try:
                self.feature_store = FeatureStoreFacade(
                    self.cfg, mode="online"
                )
            except Exception as e:
                print(f"[WARNING] Feast initialization failed: {e}")
                self.feature_store = None

    def get_online_dataset(
        self, entities: List[Union[int, str]], time_point: str = "now"
    ) -> pd.DataFrame:
        if self.feature_store is None:
            raise RuntimeError("Feast feature store not initialized")

        data = self.feature_store.load_features(
            time_point=time_point, entity_ids=entities
        )
        return pd.DataFrame(data)

    def _load_models(self) -> None:
        if not self.mlflow_manager.enabled:
            return
        experiment_name = self.cfg.experiment.get(
            "name", "nested_suppipeline_serve"
        )
        print(
            f"[ModelService] Loading model: fitted_train_model "
            f"(alias: production)..."
        )
        component = self.mlflow_manager.load_component(
            name=f"{experiment_name}_train_model",
            alias="production",
        )
        if component is not None:
            self.models["fitted_train_model"] = component

        # Load feature generator: pca
        print(
            f"[ModelService] Loading feature generator: "
            f"pca (alias: production)..."
        )
        component = self.mlflow_manager.load_component(
            name=f"{experiment_name}_pca",
            alias="production",
        )
        if component is not None:
            self.feature_generators["pca_data"] = {
                "model": component,
                "method": "transform",
                "type": "dynamic_adapter",
            }
        # Load feature generator: cluster_type_1
        print(
            f"[ModelService] Loading feature generator: "
            f"cluster_type_1 (alias: production)..."
        )
        component = self.mlflow_manager.load_component(
            name=f"{experiment_name}_cluster_type_1",
            alias="production",
        )
        if component is not None:
            self.feature_generators["cluster_1_features"] = {
                "model": component,
                "method": "predict",
                "type": "inference",
            }
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

        model = self.models.get("fitted_train_model")
        if model is not None:

            # train_model_inference: merge
            base = context.get("preprocessed_data")
            adds = []
            for k in ["cluster_1_features", "pca_data"]:
                if k in context:
                    v = context[k]
                    if isinstance(v, pd.DataFrame):
                        adds.append(v.values)
                    elif isinstance(v, np.ndarray):
                        adds.append(v)
            if isinstance(base, pd.DataFrame):
                x = base.values
            else:
                x = np.array(base) if base is not None else None
            if x is not None and adds:
                x = np.concatenate([x] + adds, axis=-1)

            if x is not None:
                inp = self._prepare_input_tabular(x)
                metadata["n_samples"] = len(inp)
                p = model.predict(inp)
                results["train_model_predictions"] = p.tolist()
                context["train_model_predictions"] = p
                if return_probabilities and hasattr(model, "predict_proba"):
                    try:
                        pb = model.predict_proba(inp)
                        results["train_model_predictions_probabilities"] = (
                            pb.tolist()
                        )
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

        model = self.models.get("fitted_train_model")
        if model is not None:
            feat = context.get("preprocessed_data")
            if feat is not None:
                preds = []
                cur = feat.copy() if isinstance(feat, pd.DataFrame) else feat
                for bi in range(n_blocks):
                    if len(cur) < self.INPUT_CHUNK_LENGTH:
                        break

                    base = (
                        cur.values if isinstance(cur, pd.DataFrame)
                        else np.array(cur)
                    )
                    adds = []
                    for k in ["cluster_1_features", "pca_data"]:
                        if k in context:
                            v = context[k]
                            if isinstance(v, pd.DataFrame):
                                adds.append(v.values[:len(base)])
                            elif isinstance(v, np.ndarray):
                                adds.append(v[:len(base)])
                    merged = (
                        np.concatenate([base] + adds, axis=-1) if adds else base
                    )

                    inp = self._prepare_input_timeseries(
                        merged, "ml"
                    )
                    bp = model.predict(inp)
                    preds.append(bp[0])
                    if bi < n_blocks - 1 and hasattr(cur, "iloc"):
                        sh = min(self.OUTPUT_CHUNK_LENGTH, len(bp))
                        if isinstance(cur, pd.DataFrame):
                            cur = cur.iloc[sh:]
                preds = np.array(preds)
                out = (
                    preds if preds.ndim == 1
                    else np.concatenate(preds, axis=0)
                )
                results["train_model_predictions"] = out.tolist()
                context["train_model_predictions"] = out

        return {"predictions": results, "metadata": metadata}

    def generate_additional_features(
        self, base_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate additional features using feature generators."""
        additional_features = {}
        if not self.feature_generators:
            return additional_features

        print(
            f"[ModelService] Generating additional features from "
            f"{len(self.feature_generators)} generators..."
        )

        x_input = (
            base_features.values
            if isinstance(base_features, pd.DataFrame)
            else base_features
        )

        for output_key, fg_info in self.feature_generators.items():
            model = fg_info["model"]
            method = fg_info["method"]
            fg_type = fg_info["type"]

            try:
                inference_fn = getattr(model, method, None)
                if inference_fn is None:
                    inference_fn = (
                        getattr(model, "transform", None)
                        or getattr(model, "predict", None)
                    )

                if inference_fn is None:
                    print(f"  Warning: {output_key} has no inference method")
                    continue

                if (fg_type != "dynamic_adapter") and self.DATA_TYPE != "tabular":
                    ts_x_input = self._prepare_input_timeseries(x_input, "ml")
                    result = inference_fn(ts_x_input)
                else:
                    result = inference_fn(x_input)
                additional_features[output_key] = result
                result_shape = (
                    result.shape
                    if hasattr(result, "shape")
                    else len(result)
                )
                print(f"  + {output_key} ({fg_type}): {result_shape}")

            except Exception as e:
                print(f"  Warning: Failed to generate {output_key}: {e}")
                continue

        return additional_features

    def compose_features(
        self,
        base_features: pd.DataFrame,
        additional_features: Dict[str, Any]
    ) -> pd.DataFrame:
        """Compose base features with additional generated features."""
        if not additional_features:
            return base_features

        composed = (
            base_features.copy()
            if isinstance(base_features, pd.DataFrame)
            else pd.DataFrame(base_features)
        )
        n_samples = len(composed)

        print(f"[ModelService] Composing features: base {composed.shape}")

        for key, features in additional_features.items():
            if isinstance(features, np.ndarray):
                if features.ndim == 1:
                    feat_df = pd.DataFrame({f"{key}_0": features})
                else:
                    cols = [
                        f"{key}_{i}" for i in range(features.shape[1])
                    ]
                    feat_df = pd.DataFrame(features, columns=cols)
            elif isinstance(features, pd.DataFrame):
                feat_df = features.copy()
                feat_df.columns = [f"{key}_{c}" for c in feat_df.columns]
            else:
                feat_df = pd.DataFrame({f"{key}_0": features})

            if len(feat_df) != n_samples:
                if len(feat_df) == 1:
                    feat_df = pd.concat(
                        [feat_df] * n_samples, ignore_index=True
                    )
                elif len(feat_df) > n_samples:
                    feat_df = feat_df.iloc[:n_samples]
                else:
                    n_pad = n_samples - len(feat_df)
                    pad_df = pd.concat(
                        [feat_df.iloc[[0]]] * n_pad,
                        ignore_index=True
                    )
                    feat_df = pd.concat([pad_df, feat_df], ignore_index=True)

            feat_df.index = composed.index
            composed = pd.concat([composed, feat_df], axis=1)
            print(f"  + {key}: {feat_df.shape} -> Total: {composed.shape}")

        return composed

    def run_full_pipeline(
        self, preprocessed_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run full inference pipeline including feature generation."""
        additional_features = self.generate_additional_features(
            preprocessed_data
        )
        composed = self.compose_features(preprocessed_data, additional_features)
        context = {"preprocessed_data": composed}

        if "timeseries" == "timeseries":
            return self.predict_timeseries_multistep(
                context, self.OUTPUT_CHUNK_LENGTH
            )
        return self.predict_tabular_batch(context)

    async def run_inference_pipeline(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        preprocessed = context.get("preprocessed_data")
        if preprocessed is not None:
            result = self.run_full_pipeline(preprocessed)
            return result.get("predictions", {})

        if "timeseries" == "timeseries":
            return self.predict_timeseries_multistep(
                context, self.OUTPUT_CHUNK_LENGTH
            )["predictions"]
        return self.predict_tabular_batch(context)["predictions"]


@serve.deployment
@serve.ingress(app)
class ServeAPI:
    DATA_TYPE = "timeseries"
    INPUT_CHUNK_LENGTH = 24

    def __init__(self, preprocess_handle: Any, model_handle: Any) -> None:
        self.preprocess_handle = preprocess_handle
        self.model_handle = model_handle
        self.cfg = ConfigLoader.load("mlproject/configs/experiments/etth3.yaml")

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

    @app.post("/predict/multistep", response_model=MultiPredictResponse)
    async def predict_multistep(
        self,
        request: BatchPredictRequest,
        steps_ahead: int = 6
    ) -> MultiPredictResponse:
        try:
            df = pd.DataFrame(request.data)
            preprocessed_data = (
                await self.preprocess_handle.preprocess.remote(df)
            )
            context = {"preprocessed_data": preprocessed_data}
            result = await self.model_handle.predict_timeseries_multistep.remote(
                context, steps_ahead
            )
            return MultiPredictResponse(
                predictions=result["predictions"],
                metadata=result["metadata"]
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/predict/feast", response_model=MultiPredictResponse)
    async def predict_feast(
        self, request: FeastPredictRequest
    ) -> MultiPredictResponse:
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
    config_path = args.get("config", "mlproject/configs/experiments/etth3.yaml")

    preprocess_deployment = PreprocessService.bind(config_path)
    model_deployment = ModelService.bind(config_path)

    return ServeAPI.bind(preprocess_deployment, model_deployment)


if __name__ == "__main__":
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(app_builder({}))
