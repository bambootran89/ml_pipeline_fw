"""ModelsService module with Feast integration.

Contains ModelsService which loads a model and runs inference
with support for both traditional and Feast-based predictions.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.serve.schemas import (
    FeastBatchPredictRequest,
    FeastPredictRequest,
    PredictRequest,
)
from mlproject.src.features.facade import FeatureStoreFacade
from mlproject.src.preprocess.transform_manager import TransformManager
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader


class ModelsService:
    """
    Unified service for loading models and serving predictions.

    Supports both traditional data-in-payload and Feast-native
    prediction modes.

    Attributes
    ----------
    cfg : DictConfig
        Configuration object.
    model : Optional[Any]
        Loaded model instance.
    preprocessor_model : Optional[Any]
        Loaded preprocessor from MLflow.
    local_transform_manager : Optional[TransformManager]
        Local preprocessing fallback.
    mlflow_manager : MLflowManager
        MLflow client manager.
    feast_facade : Optional[FeatureStoreFacade]
        Feast facade for online retrieval.
    """

    model: Optional[Any]
    preprocessor_model: Optional[Any]
    local_transform_manager: Optional[TransformManager]
    feast_facade: Optional[FeatureStoreFacade]

    def __init__(self, cfg_path: str = "") -> None:
        """
        Initialize service using a config path.

        Parameters
        ----------
        cfg_path : str
            Path to the configuration YAML/JSON file.
        """
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        self.model = None
        self.preprocessor_model = None
        self.local_transform_manager = None
        self.mlflow_manager = MLflowManager(self.cfg)

        # Initialize Feast facade
        try:
            self.feast_facade = FeatureStoreFacade(self.cfg, mode="online")
        except Exception as e:
            print(f"[WARNING] Feast not available: {e}")
            self.feast_facade = None

        # Load model and preprocessor
        if self.mlflow_manager.enabled:
            experiment_name: str = self.cfg.experiment["name"]
            self.preprocessor_model = self.mlflow_manager.load_component(
                name=f"{experiment_name}_preprocessor",
                alias="production",
            )
            self.model = self.mlflow_manager.load_component(
                name=f"{experiment_name}_model", alias="production"
            )

    def _get_input_chunk_length(self) -> int:
        """
        Get input chunk length from config.

        Returns
        -------
        int
            Number of input time steps.
        """
        if hasattr(self.cfg, "experiment") and hasattr(
            self.cfg.experiment, "hyperparams"
        ):
            return int(self.cfg.experiment.hyperparams.get("input_chunk_length", 24))

        if hasattr(self.cfg, "approaches") and self.cfg.approaches:
            return int(self.cfg.approaches[0].hyperparams.get("input_chunk_length", 24))

        return 24

    def _prepare_input_window(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build model input window from preprocessed DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed feature DataFrame.

        Returns
        -------
        np.ndarray
            Model input array with shape (1, seq_len, n_features).

        Raises
        ------
        ValueError
            If insufficient data for input window.
        """
        input_chunk_length = self._get_input_chunk_length()

        if len(df) < input_chunk_length:
            raise ValueError(
                f"Not enough data. Needed {input_chunk_length}, " f"got {len(df)}"
            )

        window = df.iloc[-input_chunk_length:].values
        return window[np.newaxis, :].astype(np.float32)

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data using available preprocessing pipeline.

        Priority order:
        1. MLflow PyFunc preprocessor model (if available)
        2. Local TransformManager as fallback

        Parameters
        ----------
        data : pd.DataFrame
            Raw input data.

        Returns
        -------
        pd.DataFrame
            Transformed feature data.

        Raises
        ------
        RuntimeError
            If no preprocessing pipeline available.
        """
        if self.preprocessor_model is not None:
            print("[Preprocess] Using MLflow PyFunc preprocessor.")
            return self.preprocessor_model.transform(data)

        if self.local_transform_manager is not None:
            print("[Preprocess] Using local preprocessor fallback.")
            return self.local_transform_manager.transform(data)

        raise RuntimeError("No preprocessing pipeline available.")

    def predict(self, request: PredictRequest) -> Dict[str, Any]:
        """
        Run traditional prediction pipeline (data in payload).

        Parameters
        ----------
        request : PredictRequest
            Input request containing raw data.

        Returns
        -------
        Dict[str, Any]
            Dictionary with 'prediction' key containing list.

        Raises
        ------
        RuntimeError
            If model not loaded.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        try:
            data = pd.DataFrame(request.data)
            df_transformed = self._transform_data(data)
            x_input = self._prepare_input_window(df_transformed)
            preds = self.model.predict(x_input)
            return {"prediction": preds.flatten().tolist()}

        except Exception as e:
            print(f"[ModelsService] Error during prediction: {e}")
            raise

    def predict_feast(self, request: FeastPredictRequest) -> Dict[str, List[float]]:
        """
        Run Feast-native prediction for single or few entities.

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
        RuntimeError
            If model or Feast not available.
        ValueError
            If insufficient data or invalid request.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        if self.feast_facade is None:
            raise RuntimeError("Feast facade not available.")

        try:
            # 1. Fetch features from Feast
            df = self.feast_facade.load_features(
                time_point=request.time_point,
                entity_ids=request.entities,
            )

            # 2. For multi-entity, take first entity
            entity_key = request.entity_key or self.cfg.data.get(
                "entity_key", "location_id"
            )

            if request.entities and len(request.entities) > 1:
                if entity_key in df.columns:
                    first_entity = request.entities[0]
                    df = df[df[entity_key] == first_entity].copy()

            # 3. Preprocess
            df_transformed = self._transform_data(df)

            # 4. Drop entity key if present
            if entity_key in df_transformed.columns:
                df_transformed = df_transformed.drop(columns=[entity_key])

            # 5. Prepare input and predict
            x_input = self._prepare_input_window(df_transformed)
            preds = self.model.predict(x_input)

            # 6. Sanitize JSON-unsafe values
            preds = np.asarray(preds, dtype=np.float32).flatten()
            preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

            return {"prediction": preds.tolist()}

        except Exception as e:
            print(f"[ModelsService] Feast prediction error: {e}")
            raise

    def predict_feast_batch(
        self, request: FeastBatchPredictRequest
    ) -> Dict[str, Dict[Union[int, str], List[float]]]:
        """
        Run batch prediction for multiple entities from Feast.

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
        RuntimeError
            If model or Feast not available.
        ValueError
            If no successful predictions.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        if self.feast_facade is None:
            raise RuntimeError("Feast facade not available.")

        try:
            # 1. Fetch features for ALL entities (single query!)
            df_all = self.feast_facade.load_features(
                time_point=request.time_point,
                entity_ids=request.entities,
            )
            # 2. Get entity key
            entity_key = request.entity_key or self.cfg.data.get(
                "entity_key", "location_id"
            )

            # 3. Predict for each entity
            predictions: Dict[Union[int, str], List[float]] = {}
            grouped = df_all.groupby(entity_key)
            input_length = self._get_input_chunk_length()

            for entity_id in request.entities:
                try:
                    # Check if entity exists
                    if entity_id not in grouped.groups:
                        print(f"[WARNING] Entity {entity_id} not found")
                        continue

                    entity_df = grouped.get_group(entity_id).copy()

                    # Preprocess
                    df_transformed = self._transform_data(entity_df)

                    # Drop entity key
                    if entity_key in df_transformed.columns:
                        df_transformed = df_transformed.drop(columns=[entity_key])

                    # Check sufficient data
                    if len(df_transformed) < input_length:
                        print(f"[WARNING] Entity {entity_id}: " f"insufficient data")
                        continue

                    # Prepare and predict
                    x_input = self._prepare_input_window(df_transformed)
                    preds = self.model.predict(x_input)
                    # Sanitize and store
                    preds = np.asarray(preds, dtype=np.float32).flatten()
                    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
                    predictions[str(entity_id)] = preds.tolist()

                except Exception as e:
                    print(f"[ERROR] Entity {entity_id} failed: {e}")
                    continue

            if not predictions:
                raise ValueError("No successful predictions for any entity")

            return {"predictions": predictions}

        except Exception as e:
            print(f"[ModelsService] Batch prediction error: {e}")
            raise
