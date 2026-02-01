"""Ray Serve API generation mixin."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .extractors import ApiGeneratorExtractorsMixin
from .types import GenerationContext


class ApiGeneratorRayServeMixin(ApiGeneratorExtractorsMixin):
    """Mixin for generating Ray Serve API code."""

    def _generate_ray_serve_code(
        self,
        pipeline_name: str,
        load_map: Dict[str, str],
        preprocessor: Optional[Dict[str, Any]],
        inference_steps: List[Dict[str, Any]],
        experiment_config_path: str,
        data_config: Optional[Dict[str, Any]] = None,
        alias: str = "production",
    ) -> str:
        ctx = self._create_ray_context(
            pipeline_name,
            load_map,
            preprocessor,
            inference_steps,
            experiment_config_path,
            data_config,
            alias,
        )
        return self._build_ray_code(ctx)

    def _create_ray_context(
        self,
        pipeline_name: str,
        load_map: Dict[str, str],
        preprocessor: Optional[Dict[str, Any]],
        inference_steps: List[Dict[str, Any]],
        experiment_config_path: str,
        data_config: Optional[Dict[str, Any]],
        alias: str = "production",
    ) -> GenerationContext:
        # pylint: disable=C0415
        from .types import DataConfig

        return GenerationContext(
            pipeline_name=pipeline_name,
            load_map=load_map,
            preprocessor=preprocessor,
            inference_steps=inference_steps,
            experiment_config_path=experiment_config_path,
            data_config=DataConfig.from_dict(data_config),
            model_keys=[inf["model_key"] for inf in inference_steps],
            alias=alias,
        )

    def _get_feature_generator_keys(self, ctx: GenerationContext) -> List[str]:
        """Get list of feature generator model keys."""
        return [generator.model_key for generator in ctx.data_config.feature_generators]

    def _generate_feature_generators_config(self, ctx: GenerationContext) -> str:
        """Generate feature generators config as Python dict."""
        if not ctx.data_config.feature_generators:
            return "{}"

        items = []
        for generator in ctx.data_config.feature_generators:
            items.append(
                f'        "{generator.output_key}": {{'
                f'"model_key": "{generator.model_key}", '
                f'"artifact_name": "{generator.artifact_name}", '
                f'"inference_method": "{generator.inference_method}", '
                f'"step_type": "{generator.step_type}"}}'
            )

        return "{\n" + ",\n".join(items) + "\n    }"

    def _build_ray_code(self, ctx: GenerationContext) -> str:
        """Build full Ray Serve code string."""
        sections = [
            self._generate_ray_imports(),
            self._generate_pydantic_models(ctx),
            self._generate_preprocess_service(ctx),
            self._generate_model_service(ctx),
            self._generate_serve_api(ctx),
            self._generate_ray_entrypoint(ctx),
        ]
        return "\n".join(sections)

    def _generate_ray_imports(self) -> str:
        """Generate Ray Serve imports."""
        return """import os
import sys
from typing import Any, Dict, List, Optional, Union
import signal
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ray import serve
from ray.serve.handle import DeploymentHandle
import platform
# Fix for potential OpenMP conflict on macOS
if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "1"

from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader
from mlproject.src.features.facade import FeatureStoreFacade
from mlproject.src.generator.constants import API_DEFAULTS, CONTEXT_KEYS

app = FastAPI(title="ML Pipeline API (Ray Serve)")
"""

    def _generate_pydantic_models(self, ctx: GenerationContext) -> str:
        """Generate Ray Serve Pydantic models."""
        # pylint: disable=W0613
        return """
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

class MultiStepPredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(..., description="Input timeseries data")
    steps_ahead: int = Field(
        default=-1,
        description="Number of steps to predict ahead",
        ge=1
    )

class BatchPredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(
        ..., description="Input data with multiple rows"
    )
    return_probabilities: bool = Field(default=False)

class MultiPredictResponse(BaseModel):
    predictions: Dict[str, Union[List[List[float]], List[float]]]
    metadata: Optional[Dict[str, Any]] = None
"""

    def _generate_ray_entrypoint(self, ctx: GenerationContext) -> str:
        """Generate Ray Serve entrypoint."""
        return f"""
def app_builder(args: Dict[str, str]) -> Any:
    config_path = args.get("config", "{ctx.experiment_config_path}")

    preprocess_deployment = PreprocessService.bind(config_path)
    model_deployment = ModelService.bind(config_path)

    return ServeAPI.bind(preprocess_deployment, model_deployment)

if __name__ == "__main__":
    serve.run(app_builder({{}}))
    signal.pause()
"""

    def _generate_model_service(self, ctx: GenerationContext) -> str:
        """Generate ModelService methods."""
        model_loads = "\n".join(
            [
                f"""        print(
            f"[ModelService] Loading model: {key} "
            f"(alias: {ctx.alias})..."
        )
        component = self.mlflow_manager.load_component(
            name=f"{{experiment_name}}_{ctx.load_map.get(key, 'model')}",
            alias="{ctx.alias}",
        )
        if component is not None:
            self.models["{key}"] = component"""
                for key in set(ctx.model_keys)
            ]
        )

        feature_generator_loads = ""
        for generator in ctx.data_config.feature_generators:
            feature_generator_loads += f"""
        # Load feature generator: {generator.step_id}
        print(
            f"[ModelService] Loading feature generator: "
            f"{generator.artifact_name} (alias: {ctx.alias})..."
        )
        component = self.mlflow_manager.load_component(
            name=f"{{experiment_name}}_{generator.artifact_name}",
            alias="{ctx.alias}",
        )
        if component is not None:
            self.feature_generators["{generator.output_key}"] = {{
                "model": component,
                "method": "{generator.inference_method}",
                "type": "{generator.step_type}",
            }}"""

        tabular_inference_blocks = []
        for inference_step in ctx.inference_steps:
            additional_keys = inference_step.get("additional_feature_keys", [])
            features_key = inference_step["features_key"]

            if additional_keys:
                keys_str = ", ".join([f'"{key}"' for key in additional_keys])
                prep = f"""
            # {inference_step['id']}: merge additional features
            base = context.get("{features_key}")
            additional_features = []
            for key in [{keys_str}]:
                if key in context:
                    value = context[key]
                    if isinstance(value, pd.DataFrame):
                        additional_features.append(value.values)
                    elif isinstance(value, np.ndarray):
                        additional_features.append(value)
            if isinstance(base, pd.DataFrame):
                x_input = base.values
            else:
                x_input = np.array(base) if base is not None else None
            if x_input is not None and additional_features:
                x_input = np.concatenate([x_input] + additional_features, axis=-1)
"""
            else:
                prep = f"""
            base = context.get("{features_key}")
            x_input = base.values if isinstance(base, pd.DataFrame) else base
"""

            tabular_inference_blocks.append(
                f"""
        model = self.models.get("{inference_step['model_key']}")
        if model is not None:
{prep}
            if x_input is not None:
                prepared_input = self._prepare_input_tabular(x_input)
                metadata["n_samples"] = len(prepared_input)
                predictions = model.predict(prepared_input)
                results["{inference_step['output_key']}"] = predictions.tolist()
                context["{inference_step['output_key']}"] = predictions
                if return_probabilities and hasattr(model, "predict_proba"):
                    try:
                        probabilities = model.predict_proba(prepared_input)
                        results["{inference_step['output_key']}_probabilities"] = (
                            probabilities.tolist()
                        )
                    except Exception:
                        pass
"""
            )

        tabular_inference = "\n".join(tabular_inference_blocks)

        ts_inference_blocks = []
        for inference_step in ctx.inference_steps:
            additional_keys = inference_step.get("additional_feature_keys", [])

            if additional_keys:
                keys_str = ", ".join([f'"{key}"' for key in additional_keys])
                merge = f"""
                    base = (
                        current.values if isinstance(current, pd.DataFrame)
                        else np.array(current)
                    )
                    additional_features = []
                    for key in [{keys_str}]:
                        if key in context:
                            value = context[key]
                            if isinstance(value, pd.DataFrame):
                                additional_features.append(value.values[:len(base)])
                            elif isinstance(value, np.ndarray):
                                additional_features.append(value[:len(base)])
                    merged = (
                        np.concatenate([base] + additional_features, axis=-1)
                        if additional_features else base
                    )
"""
            else:
                merge = """
                    merged = current
"""

            ts_inference_blocks.append(
                f"""
        model = self.models.get("{inference_step['model_key']}")
        if model is not None:
            features = context.get("{inference_step['features_key']}")
            if features is not None:
                predictions = []
                current = (
                    features.copy() if isinstance(features, pd.DataFrame)
                    else features
                )
                for block_index in range(n_blocks):
                    if len(current) < self.INPUT_CHUNK_LENGTH:
                        break
{merge}
                    prepared_input = self._prepare_input_timeseries(
                        merged, "{inference_step['model_type']}"
                    )
                    block_predictions = model.predict(prepared_input)
                    predictions.append(block_predictions[0])
                    if block_index < n_blocks - 1 and hasattr(current, "iloc"):
                        shift = min(self.OUTPUT_CHUNK_LENGTH, len(block_predictions))
                        if isinstance(current, pd.DataFrame):
                            current = current.iloc[shift:]
                predictions = np.array(predictions)
                output = (
                    predictions if predictions.ndim == 1
                    else np.concatenate(predictions, axis=0)
                )
                results["{inference_step['output_key']}"] = output.tolist()
                context["{inference_step['output_key']}"] = output
"""
            )

        ts_inference = "\n".join(ts_inference_blocks)

        return f"""
@serve.deployment(
    num_replicas=API_DEFAULTS.RAY_MODEL_REPLICAS,
    ray_actor_options={{"num_cpus": API_DEFAULTS.RAY_MODEL_CPUS}}
)
class ModelService:
    DATA_TYPE = "{ctx.data_config.data_type}"
    INPUT_CHUNK_LENGTH = {ctx.data_config.input_chunk_length}
    OUTPUT_CHUNK_LENGTH = {ctx.data_config.output_chunk_length}
    FEATURES = {repr(ctx.data_config.features)}
    ADDITIONAL_FEATURE_KEYS = {repr(
        ctx.data_config.additional_feature_keys
    )}
    FEATURE_GENERATORS = {self._generate_feature_generators_config(ctx)}

    def __init__(self, config_path: str) -> None:
        print("[ModelService] Initializing...")
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.models = {{}}
        self.feature_generators = {{}}
        self.ready = False
        self.feature_store = None
        self._check_feast()
        self._load_models()

    def _check_feast(self) -> None:
        if "{ctx.data_config.path}".startswith("feast://"):
            print(f"[ModelService] Initializing Feast Facade...")
            try:
                self.feature_store = FeatureStoreFacade(
                    self.cfg, mode="online"
                )
            except Exception as e:
                print(f"[WARNING] Feast initialization failed: {{e}}")
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
            "name", "{ctx.pipeline_name}"
        )
{model_loads}
{feature_generator_loads}
        self.ready = True
        print("[ModelService] Ready")

    def check_health(self) -> str:
        if not self.ready:
            raise RuntimeError("ModelService not ready")
        return "healthy"

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
        results = {{}}
        metadata = {{"n_samples": 0, "model_type": "tabular"}}
{tabular_inference}
        return {{"predictions": results, "metadata": metadata}}

    def predict_timeseries_multistep(
        self, context: Dict[str, Any], steps_ahead: int
    ) -> Dict[str, Any]:
        self.check_health()
        results = {{}}
        n_blocks = (
            steps_ahead + self.OUTPUT_CHUNK_LENGTH - 1
        ) // self.OUTPUT_CHUNK_LENGTH
        metadata = {{
            "output_chunk_length": self.OUTPUT_CHUNK_LENGTH,
            "n_blocks": n_blocks,
            "model_type": "timeseries"
        }}
{ts_inference}
        return {{"predictions": results, "metadata": metadata}}

    def generate_additional_features(
        self, base_features: pd.DataFrame
    ) -> Dict[str, Any]:
        \"\"\"Generate additional features using feature generators.\"\"\"
        additional_features = {{}}
        if not self.feature_generators:
            return additional_features

        print(
            f"[ModelService] Generating additional features from "
            f"{{len(self.feature_generators)}} generators..."
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
                    print(f"  Warning: {{output_key}} has no inference method")
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
                print(f"  + {{output_key}} ({{fg_type}}): {{result_shape}}")

            except Exception as e:
                print(f"  Warning: Failed to generate {{output_key}}: {{e}}")
                continue

        return additional_features

    def compose_features(
        self,
        base_features: pd.DataFrame,
        additional_features: Dict[str, Any]
    ) -> pd.DataFrame:
        \"\"\"Compose base features with additional generated features.\"\"\"
        if not additional_features:
            return base_features

        composed = (
            base_features.copy()
            if isinstance(base_features, pd.DataFrame)
            else pd.DataFrame(base_features)
        )
        n_samples = len(composed)

        print(f"[ModelService] Composing features: base {{composed.shape}}")

        for key, features in additional_features.items():
            if isinstance(features, np.ndarray):
                if features.ndim == 1:
                    feat_df = pd.DataFrame({{f"{{key}}_0": features}})
                else:
                    cols = [
                        f"{{key}}_{{i}}" for i in range(features.shape[1])
                    ]
                    feat_df = pd.DataFrame(features, columns=cols)
            elif isinstance(features, pd.DataFrame):
                feat_df = features.copy()
                feat_df.columns = [f"{{key}}_{{c}}" for c in feat_df.columns]
            else:
                feat_df = pd.DataFrame({{f"{{key}}_0": features}})

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
            print(f"  + {{key}}: {{feat_df.shape}} -> Total: {{composed.shape}}")

        return composed

    async def run_full_pipeline(
        self,
        preprocessed: pd.DataFrame,
        steps_ahead: int = -1,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        \"\"\"Run full inference pipeline including feature generation.\"\"\"
        additional_features = self.generate_additional_features(
            preprocessed
        )
        composed = self.compose_features(preprocessed, additional_features)
        context = {{CONTEXT_KEYS.PREPROCESSED_DATA: composed}}

        if self.DATA_TYPE == "tabular":
            result = self.predict_tabular_batch(
                context, return_probabilities=return_probabilities
            )
        else:
            if steps_ahead == -1:
                steps_ahead = self.OUTPUT_CHUNK_LENGTH
            result = self.predict_timeseries_multistep(
                context,
                steps_ahead=steps_ahead
            )

        return result
"""

    def _generate_preprocess_service(self, ctx: GenerationContext) -> str:
        """Generate PreprocessService."""
        preprocessor_artifact = self._get_preprocessor_artifact_name(
            ctx.preprocessor, ctx.load_map
        )
        prep_load = ""
        if preprocessor_artifact:
            prep_load = (
                f"""        print(
            f"[PreprocessService] Loading preprocessor: "
            f"{preprocessor_artifact} (alias: {ctx.alias})..."
        )\n"""
                f"""        component = self.mlflow_manager.load_component(\n"""
                f"""            name=f"{{experiment_name}}_"\n"""
                f"""                 f"{preprocessor_artifact}",\n"""
                f"""            alias="{ctx.alias}",\n"""
                f"""        )\n"""
                f"""        if component is not None:\n"""
                f"""            self.preprocessor = component"""
            )

        return f"""@serve.deployment(
    num_replicas=API_DEFAULTS.RAY_PREPROCESS_REPLICAS,
    ray_actor_options={{"num_cpus": API_DEFAULTS.RAY_PREPROCESS_CPUS}}
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
            "name", "{ctx.pipeline_name}"
        )
{prep_load}

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is not None:
            return self.preprocessor.transform(data)
        return data

    def check_health(self) -> str:
        if self.preprocessor is not None or not self.mlflow_manager.enabled:
            return "healthy"
        return "initializing"
"""

    def _generate_serve_api(self, ctx: GenerationContext) -> str:
        """Generate ServeAPI."""
        if ctx.data_config.data_type == "tabular":
            specific_endpoint = """
    @app.post("/predict/batch", response_model=MultiPredictResponse)
    async def predict_batch(
        self, request: BatchPredictRequest
    ) -> MultiPredictResponse:
        try:
            df = pd.DataFrame(request.data)
            preprocessed_data = (
                await self.preprocess_handle.preprocess.remote(df)
            )
            result = await self.model_handle.run_full_pipeline.remote(
                preprocessed_data, return_probabilities=request.return_probabilities
            )
            return MultiPredictResponse(
                predictions=result.get("predictions", {}),
                metadata=result.get("metadata", {})
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
"""
        else:
            specific_endpoint = """
    @app.post("/predict/multistep", response_model=MultiPredictResponse)
    async def predict_multistep(
        self,
        request: MultiStepPredictRequest,
    ) -> MultiPredictResponse:
        try:
            df = pd.DataFrame(request.data)
            preprocessed_data = (
                await self.preprocess_handle.preprocess.remote(df)
            )
            result = await self.model_handle.run_full_pipeline.remote(
                preprocessed_data, request.steps_ahead
            )
            return MultiPredictResponse(
                predictions=result.get("predictions", {}),
                metadata=result.get("metadata", {})
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
"""

        return f"""@serve.deployment
@serve.ingress(app)
class ServeAPI:
    DATA_TYPE = "{ctx.data_config.data_type}"
    INPUT_CHUNK_LENGTH = {ctx.data_config.input_chunk_length}

    def __init__(self, preprocess_handle: Any, model_handle: Any) -> None:
        self.preprocess_handle = preprocess_handle
        self.model_handle = model_handle
        self.cfg = ConfigLoader.load("{ctx.experiment_config_path}")

    @app.post("/predict", response_model=MultiPredictResponse)
    async def predict(self, request: PredictRequest) -> MultiPredictResponse:
        try:
            df = pd.DataFrame(request.data)
            preprocessed_data = (
                await self.preprocess_handle.preprocess.remote(df)
            )
            result = (
                await self.model_handle.run_full_pipeline.remote(
                    preprocessed_data
                )
            )
            return MultiPredictResponse(
                predictions=result.get("predictions", {{}}),
                metadata=result.get("metadata", {{}})
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
{specific_endpoint}
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
            result = (
                await self.model_handle.run_full_pipeline.remote(
                    preprocessed_data
                )
            )
            return MultiPredictResponse(
                predictions=result.get("predictions", {{}}),
                metadata=result.get("metadata", {{}})
            )
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
            model_status = await self.model_handle.check_health.remote()
            return HealthResponse(
                status="healthy",
                components={{
                    "preprocess": prep_status,
                    "model": model_status
                }}
            )
        except Exception as e:
            return HealthResponse(
                status="unhealthy",
                components={{"error": str(e)}}
            )
"""
