# API Generation Guide

This guide explains how to generate and test FastAPI/Ray Serve APIs from ML pipeline configurations.

lsof -nP -iTCP:8000 -sTCP:LISTEN

## Overview

The API generator supports two data types:
- **Tabular data**: Batch prediction for multiple rows
- **Timeseries data**: Multi-step prediction for forecasting

## Quick Start

### 0. Generate artifacts + serve.yaml
```bash
# generate eval, serve, tune yaml
python -m mlproject.src.pipeline.dag_run generate \
    --train-config mlproject/configs/pipelines/standard_train.yaml \
    --output-dir mlproject/configs/generated \
    --alias latest
```

```bash
# Tabular
python -m mlproject.src.pipeline.dag_run train \
    --experiment mlproject/configs/experiments/tabular.yaml \
    --pipeline mlproject/configs/pipelines/standard_train.yaml

# Timeseries
python -m mlproject.src.pipeline.dag_run train \
    --experiment mlproject/configs/experiments/etth3.yaml \
    --pipeline mlproject/configs/pipelines/standard_train.yaml
```

### 1. Generate APIs

```bash
# Generate APIs for all pipelines (timeseries + tabular)
python examples/generate_serve_apis.py
```

Generated files will be in `mlproject/serve/generated/`:
- `*_fastapi.py` - FastAPI standalone server
- `*_ray.py` - Ray Serve distributed deployment

### 2. Run the API

**FastAPI (Timeseries):**
```bash
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml -f fastapi mlproject/configs/generated/standard_train_serve.yaml
```

**FastAPI (Tabular):**
```bash
mlproject/serve_api.sh -e mlproject/configs/experiments/tabular.yaml -f fastapi mlproject/configs/generated/standard_train_serve.yaml
```

**Ray Serve:**
```bash
python mlproject/serve/generated/***_ray.py
```

## API Endpoints

### Common Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Standard prediction |

### Tabular-specific Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/batch` | POST | Batch prediction with optional probabilities |

### Timeseries-specific Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/multistep` | POST | Multi-step prediction with configurable horizon |

## Testing with curl

### Health Check

```bash
curl http://localhost:8000/health
```

### Tabular Predictions

**Single prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "Pclass": [3],
      "Age": [25.0],
      "SibSp": [0],
      "Parch": [0],
      "Fare": [7.25],
      "Sex": [1],
      "Embarked": [2]
    }
  }'
```

**Batch prediction (10 samples):**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "Pclass": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
      "Age": [22.0, 38.0, 26.0, 35.0, 28.0, 45.0, 19.0, 55.0, 32.0, 41.0],
      "SibSp": [1, 1, 0, 1, 0, 0, 3, 0, 1, 0],
      "Parch": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
      "Fare": [7.25, 71.28, 7.92, 53.1, 8.05, 8.05, 21.07, 30.5, 15.55, 26.55],
      "Sex": [1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
      "Embarked": [2, 0, 2, 2, 2, 1, 2, 2, 2, 2]
    },
    "return_probabilities": false
  }'
```

### Timeseries Predictions

**Important:** All arrays must have the same length (24 timesteps for input_chunk_length=24).

**Standard prediction (output_chunk_length steps):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00", "2020-01-01 03:00:00", "2020-01-01 04:00:00", "2020-01-01 05:00:00", "2020-01-01 06:00:00", "2020-01-01 07:00:00", "2020-01-01 08:00:00", "2020-01-01 09:00:00", "2020-01-01 10:00:00", "2020-01-01 11:00:00", "2020-01-01 12:00:00", "2020-01-01 13:00:00", "2020-01-01 14:00:00", "2020-01-01 15:00:00", "2020-01-01 16:00:00", "2020-01-01 17:00:00", "2020-01-01 18:00:00", "2020-01-01 19:00:00", "2020-01-01 20:00:00", "2020-01-01 21:00:00", "2020-01-01 22:00:00", "2020-01-01 23:00:00"],
      "HUFL": [5.827, 5.8, 5.969, 6.372, 7.153, 7.976, 8.715, 9.340, 9.763, 9.986, 10.040, 9.916, 9.609, 9.156, 8.591, 7.970, 7.338, 6.745, 6.233, 5.838, 5.582, 5.465, 5.465, 5.557],
      "MUFL": [1.599, 1.492, 1.492, 1.492, 1.492, 1.509, 1.582, 1.711, 1.896, 2.113, 2.337, 2.552, 2.742, 2.902, 3.024, 3.104, 3.137, 3.125, 3.067, 2.969, 2.838, 2.683, 2.515, 2.346],
      "mobility_inflow": [1.234, 1.456, 1.678, 1.890, 2.123, 2.456, 2.789, 3.012, 3.234, 3.456, 3.678, 3.890, 4.012, 4.123, 4.234, 4.345, 4.456, 4.567, 4.678, 4.789, 4.890, 4.901, 4.912, 4.923]
    }
  }'
```

**Multi-step prediction (18 steps = 3 blocks of 6):**
```bash
curl -X POST http://localhost:8000/predict/multistep \
  -H "Content-Type: application/json" \
  -d '{
  "data": {
    "date": ["2020-01-01 00:00:00","2020-01-01 01:00:00","2020-01-01 02:00:00","2020-01-01 03:00:00","2020-01-01 04:00:00","2020-01-01 05:00:00","2020-01-01 06:00:00","2020-01-01 07:00:00","2020-01-01 08:00:00","2020-01-01 09:00:00","2020-01-01 10:00:00","2020-01-01 11:00:00","2020-01-01 12:00:00","2020-01-01 13:00:00","2020-01-01 14:00:00","2020-01-01 15:00:00","2020-01-01 16:00:00","2020-01-01 17:00:00","2020-01-01 18:00:00","2020-01-01 19:00:00","2020-01-01 20:00:00","2020-01-01 21:00:00","2020-01-01 22:00:00","2020-01-01 23:00:00","2020-01-02 00:00:00","2020-01-02 01:00:00","2020-01-02 02:00:00","2020-01-02 03:00:00","2020-01-02 04:00:00","2020-01-02 05:00:00","2020-01-02 06:00:00","2020-01-02 07:00:00","2020-01-02 08:00:00","2020-01-02 09:00:00","2020-01-02 10:00:00","2020-01-02 11:00:00"],
    "HUFL": [5.827,5.8,5.969,6.372,7.153,7.976,8.715,9.34,9.763,9.986,10.04,9.916,9.609,9.156,8.591,7.97,7.338,6.745,6.233,5.838,5.582,5.465,5.465,5.557,5.607,5.657,5.707,5.757,5.807,5.857,5.907,5.957,6.007,6.057,6.107,6.157],
    "MUFL": [1.599,1.492,1.492,1.492,1.492,1.509,1.582,1.711,1.896,2.113,2.337,2.552,2.742,2.902,3.024,3.104,3.137,3.125,3.067,2.969,2.838,2.683,2.515,2.346,2.366,2.386,2.406,2.426,2.446,2.466,2.486,2.506,2.526,2.546,2.566,2.586],
    "mobility_inflow": [1.234,1.456,1.678,1.89,2.123,2.456,2.789,3.012,3.234,3.456,3.678,3.89,4.012,4.123,4.234,4.345,4.456,4.567,4.678,4.789,4.89,4.901,4.912,4.923,4.953,4.983,5.013,5.043,5.073,5.103,5.133,5.163,5.193,5.223,5.253,5.283]
  },
  "steps_ahead": 18
}'
```

**Using Python to generate test data (recommended):**
```python
# This generates properly formatted test data
python -c "
from examples.generate_test_data import generate_timeseries_data
import json
data = generate_timeseries_data(num_timesteps=24)
print(json.dumps({'data': data}))
"
```

## Testing with Python

### Generate Test Data

```python
from examples.generate_test_data import (
    generate_tabular_data,
    generate_timeseries_data,
    generate_long_timeseries_data,
)

# Tabular data (10 samples)
tabular_data = generate_tabular_data(num_samples=10)

# Timeseries data (24 timesteps)
ts_data = generate_timeseries_data(num_timesteps=24)

# Long timeseries for multi-step prediction
long_ts = generate_long_timeseries_data(
    input_chunk_length=24,
    output_chunk_length=6,
    num_prediction_blocks=3,
)
```

### Tabular Batch Prediction

```python
import requests
from examples.generate_test_data import generate_tabular_data

# Generate test data
data = generate_tabular_data(num_samples=50)

# Batch prediction with probabilities
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "data": data,
        "return_probabilities": True
    }
)

result = response.json()
print(f"Predictions: {result['predictions']}")
print(f"Metadata: {result['metadata']}")
# Output:
# Predictions: {'inference_predictions': [0, 1, 1, ...]}
# Metadata: {'n_samples': 50, 'model_type': 'tabular'}
```

### Timeseries Multi-step Prediction

```python
import requests
from examples.generate_test_data import generate_timeseries_data

# Generate input data (at least input_chunk_length timesteps)
data = generate_timeseries_data(num_timesteps=24)

# Multi-step prediction (18 steps = 3 prediction blocks)
response = requests.post(
    "http://localhost:8000/predict/multistep",
    json={
        "data": data,
        "steps_ahead": 18  # 3 blocks * 6 output_chunk_length
    }
)

result = response.json()
print(f"Predictions: {len(result['predictions']['inference_predictions'])} steps")
print(f"Metadata: {result['metadata']}")
# Output:
# Predictions: 18 steps
# Metadata: {'steps_ahead': 18, 'output_chunk_length': 6, 'n_blocks': 3, 'model_type': 'timeseries'}
```
