"""
Request/Response schemas for Ray Serve API.

Extends existing PredictRequest with Feast-native request types.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    """
    Request model for the prediction endpoint (existing).

    Attributes:
        data (Dict[str, List]): A dictionary containing historical data.
            - Keys: feature names, e.g., "date", "HUFL", "MUFL", etc.
            - Values: Lists of values corresponding to each feature.

    Example:
        {
            "data": {
                "date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
                "HUFL": [5.827, 5.8],
                "MUFL": [1.599, 1.492],
                "mobility_inflow": [1.234, 2.345]
            }
        }
    """

    data: Dict[str, List[Any]]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": {
                    "date": [
                        "2020-01-01 00:00:00",
                        "2020-01-01 01:00:00",
                    ],
                    "HUFL": [5.827, 5.8],
                    "MUFL": [1.599, 1.492],
                    "mobility_inflow": [1.234, 2.345],
                }
            }
        }
    )


class FeastPredictRequest(BaseModel):
    """
    Feast-native prediction request for single or few entities.

    The API fetches feature data from Feast automatically.
    Client only needs to specify when and for which entities.

    Attributes:
        time_point: Reference time point. Can be "now" or ISO datetime.
        entities: Entity IDs to fetch features for. If None, uses default.
        entity_key: Entity join key name. If None, uses config default.

    Examples:
        Single entity (uses config default):
        >>> {
        ...     "time_point": "now"
        ... }

        Explicit single entity:
        >>> {
        ...     "time_point": "2024-01-01T12:00:00",
        ...     "entities": [1],
        ...     "entity_key": "location_id"
        ... }

        Multiple entities (returns first):
        >>> {
        ...     "time_point": "now",
        ...     "entities": [1, 2, 3]
        ... }
    """

    time_point: str = "now"
    entities: Optional[List[Union[int, str]]] = None
    entity_key: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"time_point": "now"},
                {
                    "time_point": "2024-01-01T12:00:00",
                    "entities": [1],
                    "entity_key": "location_id",
                },
            ]
        }
    )

    @field_validator("time_point")
    @classmethod
    def validate_time_point(cls, v: str) -> str:
        """Validate time_point format."""
        if v != "now":
            # Try parsing as ISO datetime

            try:
                datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError as e:
                raise ValueError(
                    "time_point must be 'now' or ISO datetime string "
                    "(e.g., '2024-01-01T12:00:00')"
                ) from e
        return v


class FeastBatchPredictRequest(BaseModel):
    """
    Batch prediction request for multiple entities from Feast.

    Optimized for predicting many entities simultaneously.
    Fetches all features in a single Feast query.

    Attributes:
        time_point: Reference time point for all entities.
        entities: List of entity IDs (required, min 1).
        entity_key: Entity join key name.

    Examples:
        >>> {
        ...     "time_point": "now",
        ...     "entities": [1, 2, 3, 4, 5],
        ...     "entity_key": "location_id"
        ... }
    """

    time_point: str = "now"
    entities: List[Union[int, str]] = Field(..., min_length=1)
    entity_key: str = "location_id"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "time_point": "now",
                "entities": [1, 2, 3, 4, 5],
                "entity_key": "location_id",
            }
        }
    )

    @field_validator("entities")
    @classmethod
    def validate_entities_not_empty(
        cls, v: List[Union[int, str]]
    ) -> List[Union[int, str]]:
        """Ensure entities list is not empty."""
        if not v:
            raise ValueError("entities list cannot be empty")
        return v

    @field_validator("time_point")
    @classmethod
    def validate_time_point(cls, v: str) -> str:
        """Validate time_point format."""
        if v != "now":
            try:
                datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError as e:
                raise ValueError(
                    "time_point must be 'now' or ISO datetime string"
                ) from e
        return v


class PredictResponse(BaseModel):
    """
    Response for single prediction request.

    Attributes:
        prediction: List of predicted values.

    Example:
        >>> {"prediction": [25.5, 26.0, 24.8]}
    """

    prediction: List[float]

    model_config = ConfigDict(
        json_schema_extra={"example": {"prediction": [25.5, 26.0, 24.8]}}
    )


class BatchPredictResponse(BaseModel):
    """
    Response for batch prediction request.

    Returns predictions grouped by entity ID.

    Attributes:
        predictions: Dictionary mapping entity ID to prediction list.

    Example:
        >>> {
        ...     "predictions": {
        ...         "1": [25.5, 26.0, 24.8],
        ...         "2": [22.3, 23.1, 22.7],
        ...         "3": [28.9, 29.5, 28.3]
        ...     }
        ... }
    """

    predictions: Dict[Union[int, str], List[float]]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": {
                    "1": [25.5, 26.0, 24.8],
                    "2": [22.3, 23.1, 22.7],
                    "3": [28.9, 29.5, 28.3],
                }
            }
        }
    )


class HealthResponse(BaseModel):
    """
    Health check response.

    Attributes:
        status: Overall API status.
        model_loaded: Model loading status.
        feast_available: Feast service availability (optional).

    Example:
        >>> {
        ...     "status": true,
        ...     "model_loaded": true,
        ...     "feast_available": true
        ... }
    """

    status: bool
    model_loaded: bool
    feast_available: Optional[bool] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": True,
                "model_loaded": True,
                "feast_available": True,
            }
        }
    )
