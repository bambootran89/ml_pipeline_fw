from __future__ import annotations

from typing import Any, Dict, List, Optional

from .extractors import ApiGeneratorExtractorsMixin
from .types import DataConfig, GenerationContext


class ApiGeneratorFastAPIMixin(ApiGeneratorExtractorsMixin):
    """FastAPI code generation mixin."""

    def _generate_fastapi_code(
        self,
        pipeline_name: str,
        load_map: Dict[str, str],
        preprocessor: Optional[Dict[str, Any]],
        inference_steps: List[Dict[str, Any]],
        experiment_config_path: str,
        data_config: Optional[Dict[str, Any]] = None,
        alias: str = "production",
    ) -> str:
        """Generate FastAPI code."""
        ctx = self._create_generation_context(
            pipeline_name,
            load_map,
            preprocessor,
            inference_steps,
            experiment_config_path,
            data_config,
            alias,
        )
        return self._build_fastapi_code(ctx)

    def _create_generation_context(
        self,
        pipeline_name: str,
        load_map: Dict[str, str],
        preprocessor: Optional[Dict[str, Any]],
        inference_steps: List[Dict[str, Any]],
        experiment_config_path: str,
        data_config: Optional[Dict[str, Any]],
        alias: str = "production",
    ) -> GenerationContext:
        """Create generation context from parameters."""
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
        return [fg.model_key for fg in ctx.data_config.feature_generators]

    def _generate_feature_generators_config(self, ctx: GenerationContext) -> str:
        """Generate feature generators config as Python dict."""
        if not ctx.data_config.feature_generators:
            return "{}"

        items = []
        for fg in ctx.data_config.feature_generators:
            items.append(
                f'        "{fg.output_key}": {{'
                f'"model_key": "{fg.model_key}", '
                f'"artifact_name": "{fg.artifact_name}", '
                f'"inference_method": "{fg.inference_method}", '
                f'"step_type": "{fg.step_type}"}}'
            )

        return "{\n" + ",\n".join(items) + "\n    }"

    def _build_fastapi_code(self, ctx: GenerationContext) -> str:
        """Build complete FastAPI code."""
        parts = [
            self._generate_header(ctx),
            self._generate_service_init(ctx),
            self._generate_service_methods(),
            self._generate_tabular_inference(ctx.inference_steps),
            self._generate_timeseries_inference(ctx),
            self._generate_service_entrypoint(),
            self._generate_api_init(ctx),
            self._generate_api_endpoints(ctx),
            self._generate_main(),
        ]
        return "".join(parts)

    def _generate_header(self, ctx: GenerationContext) -> str:
        """Generate file header and imports."""
        desc = (
            "Tabular batch prediction"
            if ctx.data_config.data_type == "tabular"
            else "Timeseries multi-step prediction"
        )
        api_type = ctx.data_config.data_type
        features_str = repr(ctx.data_config.features)

        return f"""# Auto-generated FastAPI serve for {ctx.pipeline_name}
# Generated from serve configuration.
# Supports: {desc}

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

from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader
from mlproject.src.features.facade import FeatureStoreFacade
from mlproject.src.generator.constants import API_DEFAULTS, CONTEXT_KEYS

app = FastAPI(
    title="{ctx.pipeline_name} API",
    version="1.0.0",
    description="Auto-generated serve API for {api_type} data",
)


# Request/Response Schemas

class FeastPredictRequest(BaseModel):
    entities: List[Union[int, str]] = Field(..., description="List of entity IDs")
    entity_key: Optional[str] = Field(None, description="Key to join entities")
    time_point: str = Field(default="now", description="Time point for retrieval")

class PredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(...,
        description="Input data as dict of columns to values"
    )


class BatchPredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(..., description="Input data with multiple rows")
    return_probabilities: bool = Field(
        default=False,
        description="Return prediction probabilities if available"
    )


class MultiStepPredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(..., description="Input timeseries data")
    steps_ahead: int = Field(
        default={ctx.data_config.output_chunk_length},
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
    data_type: str = "{ctx.data_config.data_type}"
    features: List[str] = {features_str}


# Service Implementation

class ServeService:
    DATA_TYPE = "{ctx.data_config.data_type}"
    INPUT_CHUNK_LENGTH = {ctx.data_config.input_chunk_length}
    OUTPUT_CHUNK_LENGTH = {ctx.data_config.output_chunk_length}
    FEATURES = {features_str}
    ADDITIONAL_FEATURE_KEYS = {repr(
        ctx.data_config.additional_feature_keys
    )}
    FEATURE_GENERATORS = {self._generate_feature_generators_config(ctx)}

    def __init__(self, config_path: str) -> None:
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.preprocessor = None
        self.models = {{}}
        self.feature_generators = {{}}
        self.feature_store = None

        if self.mlflow_manager.enabled:
            experiment_name = self.cfg.experiment.get(
                "name", "{ctx.pipeline_name}"
            )
"""

    def _generate_service_init(self, ctx: GenerationContext) -> str:
        """Generate service initialization code."""
        parts = []

        if ctx.data_config.is_feast:
            parts.append(
                """
            print(f"[ModelService] Initializing Feast Facade...")
            try:
                self.feature_store = FeatureStoreFacade(
                    self.cfg, mode="online"
                )
            except Exception as e:
                print(f"[WARNING] Feast initialization failed: {e}")
                self.feature_store = None
"""
            )

        preprocessor_artifact = self._get_preprocessor_artifact_name(
            ctx.preprocessor, ctx.load_map
        )
        if preprocessor_artifact:
            parts.append(
                f"""
            print(
                f"[ModelService] Loading preprocessor: "
                f"{preprocessor_artifact} (alias: {ctx.alias})..."
            )
            component = self.mlflow_manager.load_component(
                name=f"{{experiment_name}}_{preprocessor_artifact}",
                alias="{ctx.alias}",
            )
            if component is not None:
                self.preprocessor = component
"""
            )

        for model_key in set(ctx.model_keys):
            step_id = ctx.load_map.get(model_key, "model")
            parts.append(
                f"""
            print(
                f"[ModelService] Loading model: {model_key} from "
                f"{step_id} (alias: {ctx.alias})..."
            )
            component = self.mlflow_manager.load_component(
                name=f"{{experiment_name}}_{step_id}",
                alias="{ctx.alias}",
            )
            if component is not None:
                self.models["{model_key}"] = component
"""
            )

        for fg in ctx.data_config.feature_generators:
            parts.append(
                f"""
            # Load feature generator: {fg.step_id}
            print(
                f"[ModelService] Loading feature generator: "
                f"{fg.artifact_name} (alias: {ctx.alias})..."
            )
            component = self.mlflow_manager.load_component(
                name=f"{{experiment_name}}_{fg.artifact_name}",
                alias="{ctx.alias}",
            )
            if component is not None:
                self.feature_generators["{fg.output_key}"] = {{
                    "model": component,
                    "method": "{fg.inference_method}",
                    "type": "{fg.step_type}",
                }}
"""
            )

        return "".join(parts)

    def _generate_service_methods(self) -> str:
        """Generate all service helper methods.

        Combines multiple method generation functions to create a complete
        service class with all necessary helper methods.
        """
        parts = [
            self._generate_preprocess_method(),
            self._generate_input_preparation_methods(),
            self._generate_feast_method(),
            self._generate_feature_generation_method(),
            self._generate_feature_composition_method(),
            self._generate_pipeline_runner_method(),
        ]
        return "\n".join(parts)

    def _generate_preprocess_method(self) -> str:
        """Generate preprocessing method."""
        return """
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            return data
        return self.preprocessor.transform(data)
"""

    def _generate_input_preparation_methods(self) -> str:
        """Generate input preparation methods for tabular and timeseries data."""
        return """
    def _prepare_input_tabular(self, features: Any) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            return features.values
        return np.atleast_2d(np.array(features))

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
"""

    def _generate_feast_method(self) -> str:
        """Generate Feast feature store method."""
        return """
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
"""

    def _generate_feature_generation_method(self) -> str:
        """Generate feature generation method using feature generators."""
        return """
    def generate_additional_features(
        self, base_features: pd.DataFrame
    ) -> Dict[str, Any]:
        \"\"\"Generate additional features using feature generators.\"\"\"
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

        for output_key, generator_info in self.feature_generators.items():
            model = generator_info["model"]
            method = generator_info["method"]
            generator_type = generator_info["type"]

            try:
                inference_fn = getattr(model, method, None)
                if inference_fn is None:
                    inference_fn = (
                        getattr(model, "transform", None)
                        or getattr(model, "predict", None)
                    )

                if inference_fn is None:
                    print(
                        f"  Warning: {output_key} has no "
                        f"{method}/transform/predict, skipping"
                    )
                    continue

                if (generator_type != "dynamic_adapter") and (
                    self.DATA_TYPE != "tabular"
                ):
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
                print(f"  + {output_key} ({generator_type}): {result_shape}")

            except Exception as e:
                print(f"  Warning: Failed to generate {output_key}: {e}")
                continue

        return additional_features
"""

    def _generate_feature_composition_method(self) -> str:
        """Generate feature composition method."""
        return """
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

        print(f"[ModelService] Composing features: base {composed.shape}")

        for key, features in additional_features.items():
            if isinstance(features, np.ndarray):
                if features.ndim == 1:
                    feat_df = pd.DataFrame({f"{key}_0": features})
                else:
                    cols = [f"{key}_{i}" for i in range(features.shape[1])]
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
                        [feat_df.iloc[[0]]] * n_pad, ignore_index=True
                    )
                    feat_df = pd.concat([pad_df, feat_df], ignore_index=True)

            feat_df.index = composed.index
            composed = pd.concat([composed, feat_df], axis=1)
            print(f"  + {key}: {feat_df.shape} -> Total: {composed.shape}")

        return composed
"""

    def _generate_pipeline_runner_method(self) -> str:
        """Generate full pipeline runner method."""
        return """
    def run_full_pipeline(
        self,
        raw_data: pd.DataFrame,
        steps_ahead: int = -1
    ) -> Dict[str, Any]:
        \"\"\"Run full inference pipeline including feature generation.\"\"\"
        preprocessed = self.preprocess(raw_data)
        additional_features = self.generate_additional_features(
            preprocessed
        )
        composed = self.compose_features(preprocessed, additional_features)
        context = {CONTEXT_KEYS.PREPROCESSED_DATA: composed}

        if self.DATA_TYPE == "tabular":
            result = self.predict_tabular_batch(context)
        else:
            if steps_ahead == -1:
                steps_ahead = self.OUTPUT_CHUNK_LENGTH
            result = self.predict_timeseries_multistep(
                context,
                steps_ahead=steps_ahead
            )

        return result
"""

    def _generate_tabular_inference(self, inference_steps: List[Dict[str, Any]]) -> str:
        """Generate tabular inference with sequential dependencies."""
        parts = [
            """
    def predict_tabular_batch(
        self,
        context: Dict[str, Any],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        results = {}
        metadata = {"n_samples": 0, "model_type": "tabular"}
"""
        ]

        for inference_step in inference_steps:
            additional_keys = inference_step.get("additional_feature_keys", [])
            features_key = inference_step["features_key"]

            if additional_keys:
                keys_str = ", ".join([f'"{key}"' for key in additional_keys])
                prep = f"""
            # {inference_step['id']}: merge features
            base = context.get("{features_key}")
            additional_features = []
            for key in [{keys_str}]:
                if key in context:
                    value = context[key]
                    if isinstance(value, pd.DataFrame):
                        additional_features.append(value.values)
                    elif isinstance(value, np.ndarray):
                        additional_features.append(value)
                    elif isinstance(value, list):
                        additional_features.append(np.array(value))
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
            if isinstance(base, pd.DataFrame):
                x_input = base.values
            else:
                x_input = np.array(base) if base is not None else None
"""

            parts.append(
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

        parts.append(
            """
        return {"predictions": results, "metadata": metadata}
"""
        )
        return "".join(parts)

    def _generate_timeseries_inference(self, ctx: GenerationContext) -> str:
        """Generate timeseries inference with sequential dependencies."""
        parts = [
            """
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
"""
        ]

        for inference_step in ctx.inference_steps:
            additional_keys = inference_step.get("additional_feature_keys", [])

            if additional_keys:
                keys_str = ", ".join([f'"{key}"' for key in additional_keys])
                merge = f"""
                    if isinstance(current, pd.DataFrame):
                        base = current.values
                    else:
                        base = np.array(current)
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

            parts.append(
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
                if predictions.ndim == 1:
                    output = predictions
                else:
                    output = np.concatenate(predictions, axis=0)
                results["{inference_step['output_key']}"] = output.tolist()
                context["{inference_step['output_key']}"] = output
"""
            )

        parts.append(
            """
        return {"predictions": results, "metadata": metadata}
"""
        )
        return "".join(parts)

    def _generate_service_entrypoint(self) -> str:
        """Generate service entrypoint method."""
        return """
"""

    def _generate_api_init(self, ctx: GenerationContext) -> str:
        """Generate API initialization."""
        return f"""
service = ServeService("{ctx.experiment_config_path}")


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
        result = service.run_full_pipeline(df)
        return MultiPredictResponse(
            predictions=result.get("predictions", {{}}),
            metadata=result.get("metadata")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/predict/feast", response_model=MultiPredictResponse)
def predict_feast(request: FeastPredictRequest) -> MultiPredictResponse:
    try:
        df = service.get_online_dataset(request.entities)
        result = service.run_full_pipeline(df)
        return MultiPredictResponse(
            predictions=result.get("predictions", {{}}),
            metadata=result.get("metadata")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/feast/batch", response_model=MultiPredictResponse)
def predict_feast_batch(request: FeastPredictRequest) -> MultiPredictResponse:
    return predict_feast(request)

"""

    def _generate_api_endpoints(self, ctx: GenerationContext) -> str:
        """Generate data-type specific endpoints."""
        if ctx.data_config.data_type == "tabular":
            return """
@app.post("/predict/batch", response_model=MultiPredictResponse)
def predict_batch(request: BatchPredictRequest) -> MultiPredictResponse:
    return predict(request)

"""
        return """
@app.post("/predict/multistep", response_model=MultiPredictResponse)
def predict_multistep(request: MultiStepPredictRequest) -> MultiPredictResponse:
    try:
        df = pd.DataFrame(request.data)
        if len(df) < service.INPUT_CHUNK_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Input must have at least "
                       f"{service.INPUT_CHUNK_LENGTH} timesteps (got {len(df)})"
            )
        result = service.run_full_pipeline(df, steps_ahead=request.steps_ahead)
        return MultiPredictResponse(
            predictions=result.get("predictions", {}),
            metadata=result.get("metadata")
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e

"""

    def _generate_main(self) -> str:
        """Generate main entry point."""
        return """
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_DEFAULTS.FASTAPI_HOST, port=API_DEFAULTS.FASTAPI_PORT)
"""
