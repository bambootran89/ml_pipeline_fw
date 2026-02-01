"""
Test API with synthetic data using mocked ModelsService.

This test suite covers:
- Standard predictions with exact 24 rows.
- Error handling for insufficient rows (<24).
- Edge cases (all zeros, negative values).
- Multiple random samples to ensure determinism.

Run with:
    pytest mlproject/tests/test_api_synthetic.py -v -s
"""
import random
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlproject.serve.api import app, service
from mlproject.serve.schemas import PredictRequest


@pytest.fixture(autouse=True)
def mock_service_predict():
    """
    Mock service.predict to avoid real artifacts and preprocessing.
    Ensures deterministic predictions and correct error handling.
    """
    original_predict = service.predict

    def fake_predict(request: PredictRequest):
        data = request.data
        # Determine number of rows
        if isinstance(data, dict) and "date" in data:
            n_rows = len(data["date"])
        elif isinstance(data, dict) and "data" in data and "date" in data["data"]:
            n_rows = len(data["data"]["date"])
        else:
            n_rows = len(data)

        if n_rows < 24:
            raise ValueError(f"Input data has {n_rows} rows, need at least 24")

        # Deterministic prediction
        return {"prediction": np.arange(1, 7).tolist()}

    service.predict = fake_predict
    yield
    service.predict = original_predict


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


def generate_synthetic_data(n_rows: int = 24):
    """Generate synthetic time series data with n_rows."""
    start = datetime(2020, 1, 1)
    return {
        "data": {
            "date": [
                (start + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
                for i in range(n_rows)
            ],
            "HUFL": [random.uniform(-0.5, 0.5) for _ in range(n_rows)],
            "MUFL": [random.uniform(0.8, 1.2) for _ in range(n_rows)],
            "mobility_inflow": [random.uniform(0.5, 8.0) for _ in range(n_rows)],
        }
    }


class TestSyntheticAPI:
    """Test suite for synthetic API predictions using mocked service."""

    def test_predict_exact_24_rows(self, client):
        """Test prediction with exactly 24 rows."""
        data = generate_synthetic_data(24)
        resp = client.post("/predict", json=data)
        assert resp.status_code == 200
        pred = resp.json()["prediction"]
        assert pred == [1, 2, 3, 4, 5, 6]

    def test_predict_more_than_24_rows(self, client):
        """Test prediction with more than 24 rows (last 24 used)."""
        data = generate_synthetic_data(48)
        resp = client.post("/predict", json=data)
        assert resp.status_code == 200
        pred = resp.json()["prediction"]
        assert pred == [1, 2, 3, 4, 5, 6]

    def test_predict_less_than_24_rows(self, client):
        """Test prediction with less than 24 rows (should return 400)."""
        data = generate_synthetic_data(10)
        resp = client.post("/predict", json=data)
        assert resp.status_code == 400
        assert "need at least" in resp.json()["detail"].lower()

    def test_multiple_predictions_consistency(self, client):
        """Ensure repeated calls with same data return same predictions."""
        data = generate_synthetic_data(24)
        resp1 = client.post("/predict", json=data)
        resp2 = client.post("/predict", json=data)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["prediction"] == resp2.json()["prediction"]

    def test_edge_case_all_zeros(self, client):
        """Test prediction when all features are zero."""
        data = {
            "data": {
                "date": [f"2020-01-01 {i:02d}:00:00" for i in range(24)],
                "HUFL": [0.0] * 24,
                "MUFL": [0.0] * 24,
                "mobility_inflow": [0.0] * 24,
            }
        }
        resp = client.post("/predict", json=data)
        assert resp.status_code == 200
        pred = resp.json()["prediction"]
        assert pred == [1, 2, 3, 4, 5, 6]

    def test_edge_case_negative_values(self, client):
        """Test prediction when all features are negative."""
        data = {
            "data": {
                "date": [f"2020-01-01 {i:02d}:00:00" for i in range(24)],
                "HUFL": [-1.0] * 24,
                "MUFL": [-1.0] * 24,
                "mobility_inflow": [-1.0] * 24,
            }
        }
        resp = client.post("/predict", json=data)
        assert resp.status_code == 200
        pred = resp.json()["prediction"]
        assert pred == [1, 2, 3, 4, 5, 6]

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_random_seeds(self, client, seed):
        """Test predictions with different random seeds (should be deterministic)."""
        random.seed(seed)
        data = generate_synthetic_data(24)
        resp = client.post("/predict", json=data)
        assert resp.status_code == 200
        pred = resp.json()["prediction"]
        assert pred == [1, 2, 3, 4, 5, 6]

    def test_batch_predictions(self, client):
        """Test consecutive multiple calls."""
        results = []
        for i in range(5):
            random.seed(i)
            data = generate_synthetic_data(24)
            resp = client.post("/predict", json=data)
            assert resp.status_code == 200
            results.append(resp.json()["prediction"])
        for pred in results:
            assert pred == [1, 2, 3, 4, 5, 6]
