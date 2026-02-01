# Auto-generated FastAPI serve for conditional_branch_tabular_serve
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

from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader

app = FastAPI(
    title="conditional_branch_tabular_serve API",
    version="1.0.0",
    description="Auto-generated serve API for tabular data",
)


# Request/Response Schemas

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
    return_probabilities: bool = Field(
        default=False,
        description="Return prediction probabilities if available"
    )


class MultiStepPredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(
        ..., description="Input timeseries data"
    )
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
    features: List[str] = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']


# Service Implementation

class ServeService:
    DATA_TYPE = "tabular"
    INPUT_CHUNK_LENGTH = 24
    OUTPUT_CHUNK_LENGTH = 6
    FEATURES = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']
    ADDITIONAL_FEATURE_KEYS = []
    FEATURE_GENERATORS = {}

    def __init__(self, config_path: str) -> None:
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.preprocessor = None
        self.models = {}
        self.feature_generators = {}
        self.feature_store = None

        if self.mlflow_manager.enabled:
            experiment_name = self.cfg.experiment.get(
                "name", "conditional_branch_tabular_serve"
            )

            print(
                f"[ModelService] Loading preprocessor: "
                f"preprocess (alias: latest)..."
            )
            component = self.mlflow_manager.load_component(
                name=f"{experiment_name}_preprocess",
                alias="latest",
            )
            if component is not None:
                self.preprocessor = component

            print(
                f"[ModelService] Loading model: fitted_train_tab_catboost from "
                f"train_tab_catboost (alias: latest)..."
            )
            component = self.mlflow_manager.load_component(
                name=f"{experiment_name}_train_tab_catboost",
                alias="latest",
            )
            if component is not None:
                self.models["fitted_train_tab_catboost"] = component

            print(
                f"[ModelService] Loading model: fitted_train_tab_xgboost from "
                f"train_tab_xgboost (alias: latest)..."
            )
            component = self.mlflow_manager.load_component(
                name=f"{experiment_name}_train_tab_xgboost",
                alias="latest",
            )
            if component is not None:
                self.models["fitted_train_tab_xgboost"] = component

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            return data
        return self.preprocessor.transform(data)

    def _prepare_input_tabular(self, features: Any) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            return features.values
        return np.atleast_2d(np.array(features))

    def get_online_dataset(
        self,
        entities: List[Union[int, str]],
        time_point: str = "now"
    ) -> pd.DataFrame:
        if self.feature_store is None:
            raise RuntimeError("Feast not initialized")

        print(f"[ModelService] Fetching features for entities: {entities}")
        df = self.feature_store.load_features(
            time_point=time_point,
            entity_ids=entities
        )
        return df

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
                    print(
                        f"  Warning: {output_key} has no "
                        f"{method}/transform/predict, skipping"
                    )
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

    def run_full_pipeline(
        self, raw_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run full inference pipeline including feature generation."""
        preprocessed = self.preprocess(raw_data)
        additional_features = self.generate_additional_features(
            preprocessed
        )
        composed = self.compose_features(preprocessed, additional_features)
        context = {"preprocessed_data": composed}

        if self.DATA_TYPE == "tabular":
            result = self.predict_tabular_batch(context)
        else:
            result = self.predict_timeseries_multistep(
                context,
                steps_ahead=self.OUTPUT_CHUNK_LENGTH
            )

        return result

    def _prepare_input_timeseries(
        self, features: Any, model_type: str
    ) -> np.ndarray:
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

        model = self.models.get("fitted_train_tab_catboost")
        if model is not None:

            base = context.get("preprocessed_data")
            if isinstance(base, pd.DataFrame):
                x = base.values
            else:
                x = np.array(base) if base is not None else None

            if x is not None:
                inp = self._prepare_input_tabular(x)
                metadata["n_samples"] = len(inp)
                p = model.predict(inp)
                results["train_tab_catboost_predictions"] = p.tolist()
                context["train_tab_catboost_predictions"] = p
                if return_probabilities and hasattr(model, "predict_proba"):
                    try:
                        pb = model.predict_proba(inp)
                        results["train_tab_catboost_predictions_probabilities"] = (
                            pb.tolist()
                        )
                    except Exception:
                        pass

        model = self.models.get("fitted_train_tab_xgboost")
        if model is not None:

            base = context.get("preprocessed_data")
            if isinstance(base, pd.DataFrame):
                x = base.values
            else:
                x = np.array(base) if base is not None else None

            if x is not None:
                inp = self._prepare_input_tabular(x)
                metadata["n_samples"] = len(inp)
                p = model.predict(inp)
                results["train_tab_xgboost_predictions"] = p.tolist()
                context["train_tab_xgboost_predictions"] = p
                if return_probabilities and hasattr(model, "predict_proba"):
                    try:
                        pb = model.predict_proba(inp)
                        results["train_tab_xgboost_predictions_probabilities"] = (
                            pb.tolist()
                        )
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

        model = self.models.get("fitted_train_tab_catboost")
        if model is not None:
            feat = context.get("preprocessed_data")
            if feat is not None:
                preds = []
                cur = feat.copy() if isinstance(feat, pd.DataFrame) else feat
                for bi in range(n_blocks):
                    if len(cur) < self.INPUT_CHUNK_LENGTH:
                        break

                    merged = cur

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
                if preds.ndim == 1:
                    out = preds
                else:
                    out = np.concatenate(preds, axis=0)
                results["train_tab_catboost_predictions"] = out.tolist()
                context["train_tab_catboost_predictions"] = out

        model = self.models.get("fitted_train_tab_xgboost")
        if model is not None:
            feat = context.get("preprocessed_data")
            if feat is not None:
                preds = []
                cur = feat.copy() if isinstance(feat, pd.DataFrame) else feat
                for bi in range(n_blocks):
                    if len(cur) < self.INPUT_CHUNK_LENGTH:
                        break

                    merged = cur

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
                if preds.ndim == 1:
                    out = preds
                else:
                    out = np.concatenate(preds, axis=0)
                results["train_tab_xgboost_predictions"] = out.tolist()
                context["train_tab_xgboost_predictions"] = out

        return {"predictions": results, "metadata": metadata}

    def run_inference_pipeline(
        self, context: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        if self.DATA_TYPE == "tabular":
            result = self.predict_tabular_batch(context)
        else:
            result = self.predict_timeseries_multistep(
                context,
                steps_ahead=self.OUTPUT_CHUNK_LENGTH
            )
        return result.get("predictions", {})


service = ServeService("mlproject/configs/experiments/tabular.yaml")


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
            predictions=result.get("predictions", {}),
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
            predictions=result.get("predictions", {}),
            metadata=result.get("metadata")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/feast/batch", response_model=MultiPredictResponse)
def predict_feast_batch(
    request: FeastPredictRequest
) -> MultiPredictResponse:
    return predict_feast(request)


@app.post("/predict/batch", response_model=MultiPredictResponse)
def predict_batch(request: BatchPredictRequest) -> MultiPredictResponse:
    try:
        df = pd.DataFrame(request.data)
        result = service.run_full_pipeline(df)
        if request.return_probabilities:
            for key, model in service.models.items():
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(df.values)
                        result["predictions"][f"{key}_probabilities"] = (
                            proba.tolist()
                        )
                    except Exception:
                        pass
        return MultiPredictResponse(
            predictions=result.get("predictions", {}),
            metadata=result.get("metadata")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
