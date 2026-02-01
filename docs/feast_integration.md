# Feast Integration Guide

This guide details how to use Feast for feature retrieval within the ML Pipeline, covering configuration, architecture, and API usage patterns.

## 1. Architecture

The pipeline integrates Feast using a **Facade Pattern** (`FeatureStoreFacade`) which abstracts:
- **Online Retrieval**: Fetching latest values from the online store (Redis/SQLite).
- **Historical Retrieval**: Fetching point-in-time correct features for training.
- **Timeseries Windowing**: Automatically constructing sequence windows (e.g., last 24 hours) for Deep Learning models.

### Key Components
- **`FeastFeatureStore`**: The low-level client wrapper.
- **`FeatureStoreFacade`**: The high-level API that handles strategy selection (Tabular vs. Timeseries).
- **`Ray Serve / FastAPI`**: The serving layers that use the Facade to fetch data at runtime.

## 2. Configuration

To enable Feast, update your experiment YAML config (`configs/experiments/your_exp.yaml`):

```yaml
data:
  # URI format: feast://<path_to_repo>
  path: "feast://feature_repo_etth1"

  # The name of the FeatureView defined in your feature_store.yaml
  featureview: "etth1_features"

  # List of feature names (without view prefix)
  features:
    - "HUFL"
    - "MUFL"
    - "mobility_inflow"

  # Join key for entities
  entity_key: "location_id"

  # (Optional) For Timeseries: Window size
  input_chunk_length: 24
```

## 3. API Usage Patterns

The generated API provides three endpoints for different use cases.

### A. Single/Few Entity Prediction (`/predict/feast`)
Use this for real-time inference where you want to predict for specific entities.

**Request:**
```json
{
  "time_point": "now",
  "entities": [1]
}
```

**Flow:**
1. API receives Entity ID `1`.
2. Facade fetches features for `location_id=1`.
   - **Tabular**: Returns 1 row (latest values).
   - **Timeseries**: Returns `input_chunk_length` rows (latest sequence).
3. Preprocessor transforms the data.
4. Model runs inference.

### B. Batch Prediction (`/predict/feast/batch`)
Use this for high-throughput scoring of many entities efficiently.

**Request:**
```json
{
  "time_point": "now",
  "entities": [1, 2, 3, 4, 5],
  "entity_key": "location_id"
}
```

**Flow:**
1. API receives list of entities.
2. Facade issues a **single** batched call to the Feature Store to fetch data for ALL entities.
3. System groups data by entity and runs inference in parallel (using Ray).
4. Returns a dictionary map: `{"1": [...], "2": [...]}`.

### C. Timeseries Support
For timeseries models (e.g., TFT, LSTM), the system automatically handles the "lookback window".

- If `input_chunk_length: 24` is set, the API will fetch the **latest 24 hours** of data for the requested entity ending at `time_point`.
- It handles missing data steps by attempting to fallback or fill (depending on strategy).

## 4. Development Workflow

1. **Define Features**: Create `feature_store.yaml` and python definitions in your Feast repo.
2. **Apply**: Run `feast apply` to register them.
3. **Materialize**: Run `feast materialize-incremental $(date +%Y-%m-%d)` to load data into the online store.
4. **Configure**: Update your pipeline YAML.
5. **Run**: Use `mlproject.src.pipeline.dag_run` to train and serve.
