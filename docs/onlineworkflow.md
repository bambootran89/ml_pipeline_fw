# Online Workflow (Serving Pipeline)

## Overview

The serving pipeline delivers real-time predictions with:
- **Distributed services**: Ray Serve for independent scaling
- **Feast integration**: Real-time feature retrieval
- **Dual PyFunc loading**: Separate preprocessor and model artifacts paired by alias
- **Multi-entity batch**: Efficient batch predictions
- **Async coordination**: Non-blocking I/O

---

## Complete Serving Architecture

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 25,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "15px",
    "fontWeight": "600",
    "lineColor": "#2196f3"
  }
}}%%

flowchart TB
    %% Styles
    classDef client fill:#fff9c4,stroke:#f9a825,stroke-width:2.5px,color:#000,font-weight:700
    classDef api fill:#ffccbc,stroke:#d84315,stroke-width:3px,color:#000,font-weight:700
    classDef service fill:#e8f5e9,stroke:#2e7d32,stroke-width:2.5px,color:#000,font-weight:700
    classDef storage fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2.5px,color:#000,font-weight:700

    %% Nodes
    Client["<br/>  HTTP Client  <br/><br/>"]:::client

    API["<br/>  ForecastAPI  <br/>  (API Gateway)  <br/><br/>"]:::api

    FeastSvc["<br/>  FeastService  <br/>  2 replicas  <br/>  I/O bound  <br/><br/>"]:::service

    PrepSvc["<br/>  PreprocessingService  <br/>  2 replicas  <br/>  CPU bound  <br/><br/>"]:::service

    ModelSvc["<br/>  ModelService  <br/>  1-4 replicas  <br/>  GPU/CPU bound  <br/><br/>"]:::service

    FeastStore["<br/>  Feast Online  <br/>  Redis/DynamoDB  <br/><br/>"]:::storage

    MLflowReg["<br/>  MLflow Registry  <br/>  preprocessor@production  <br/>  model@production  <br/><br/>"]:::storage

    %% Flow
    Client -->|"1. Request"| API
    API -->|"2. Fetch features"| FeastSvc
    FeastSvc -->|"Query"| FeastStore

    FeastSvc -->|"3. Preprocess"| PrepSvc
    PrepSvc -->|"Load preprocessor"| MLflowReg

    PrepSvc -->|"4. Predict"| ModelSvc
    ModelSvc -->|"Load model"| MLflowReg

    ModelSvc -->|"5. Response"| API
    API -->|"6. JSON"| Client

    %% Global styling
    linkStyle default stroke:#2196f3,stroke-width:5px

```

---

## Request  Types

### 1. Traditional Prediction (Data in Payload)

### 2. Feast-Native Prediction (Single Entity)

### 3. Batch Prediction (Multi-Entity)

## Summary

**Key Features:**
- **Distributed**: Ray Serve for independent scaling
- **Fast**: Async coordination, sub-100ms latency
- **Dual PyFunc**: Separate preprocessor and model artifacts paired by alias
- **Multi-entity**: Efficient batch predictions
- **Observable**: Built-in metrics and dashboards

**Deployment Command:**
```bash
# Ray Serve (production)
python mlproject/serve/ray_deploy.py

# FastAPI (development/simple)
python mlproject/serve/api.py
```

**Endpoints:**
- `POST /predict` - Traditional (data in payload)
- `POST /predict/feast` - Feast single/multi-entity
- `POST /predict/feast/batch` - Batch multi-entity (efficient)
- `GET /health` - Health check

**Serving Flow:**
1. Load TWO artifacts from MLflow by SAME alias (cached)
   - xgboost_preprocessor@production
   - xgboost_model@production
2. Fetch features from Feast Online Store
3. Apply preprocessing using preprocessor PyFunc
4. Run inference using model PyFunc
5. Return predictions

**Artifact Pairing:**
- Preprocessor and Model loaded by SAME alias
- Version matching enforced by alias system
- Impossible to load mismatched versions
- Rollback by changing alias on both
