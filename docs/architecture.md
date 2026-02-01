# Architecture & Design Philosophy

## Overview

This ML platform is designed for **production-ready ML projects**, emphasizing:
- **Eliminating training-serving skew** through versioned artifact pairing
- **Feature Store integration** for consistent feature engineering
- **Distributed serving** with Ray for scalability
- **Design patterns** (Strategy, Facade, Factory) for maintainability
- **MLOps best practices** (versioning, monitoring, reproducibility)

---

## 1. System Architecture Overview

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "12px"
  }
}}%%

flowchart TB
    %% Styles
    classDef dataSource fill:#fff9c4,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700
    classDef featureStore fill:#c8e6c9,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700
    classDef training fill:#bbdefb,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700
    classDef registry fill:#e1bee7,stroke:#2196f3,stroke-width:3px,color:#000, font-weight:700
    classDef serving fill:#ffccbc,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700
    classDef monitoring fill:#b0bec5,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700

    %% Data Layer
    subgraph DataLayer["Data Layer"]
        RawData["Raw Data<br/>(CSV/DB/Streaming)"]:::dataSource
        Feast["Feast <br/>Feature Store<br/>Offline + Online"]:::featureStore
    end

    %% Training Layer
    subgraph TrainingLayer["Training Layer"]
        Pipeline["Training Pipeline<br/>- CV/Tuning<br/>- Fit Preprocessing<br/>- Fit Model"]:::training
        Artifacts["Artifacts<br/>- Preprocessor <br/>(PyFunc)<br/>- Model <br/>(PyFunc)<br/>(Paired by version)"]:::training
    end

    %% Registry Layer
    subgraph RegistryLayer["Model Registry"]
        MLflow["MLflow Registry<br/>- xgboost_preprocessor<br/>@production<br/>- xgboost_model<br/>@production<br/>(Same alias pairing)"]:::registry
    end


    %% Serving Layer
    subgraph ServingLayer["Serving Layer"]
        FastAPI["FastAPI<br/>(Dev/Small Scale)"]:::serving
        RayServe["Ray Serve<br/>(Production Scale)"]:::serving
    end

    %% Monitoring Layer
    subgraph MonitoringLayer["Monitoring"]
        Metrics["Metrics<br/>- Latency<br/>-Throughput"]:::monitoring
        Dashboard["Ray <br/>Dashboard<br/>MLflow UI"]:::monitoring
    end

    %% Connections
    RawData ==> Feast
    Feast ==> Pipeline
    Pipeline ==> Artifacts
    Artifacts ==> MLflow

    MLflow ==> FastAPI
    MLflow ==> RayServe

    Feast ==> FastAPI
    Feast ==> RayServe

    FastAPI ==> Metrics
    RayServe ==> Metrics
    Metrics ==> Dashboard

    %% Edge emphasis
    linkStyle default stroke:#2196f3,stroke-width:5px
```

---

## 2. Dual PyFunc Artifact Packaging (Anti-Skew Design)

**Core Principle:** Preprocessor and Model are **separate PyFunc models** but **paired by version/alias**.

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "12px"
  }
}}%%
flowchart TB
    %% Stronger Styles
    classDef training fill:#bbdefb,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:800
    classDef artifact fill:#e1bee7,stroke:#2196f3,stroke-width:4px,color:#000,font-weight:800
    classDef registry fill:#f3e5f5,stroke:#6a1b9a,stroke-width:4px,color:#000,font-weight:800
    classDef serving fill:#ffccbc,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:800
    classDef note fill:#ffffff,stroke:#2196f3,stroke-width:2px,color:#000,font-weight:600,font-size:13px,stroke-dasharray:4 4

    %% Training Phase
    subgraph Training["Training Phase"]
        direction TB
        TrainData["Training <br/>Data"]:::training
        FitPreproc["Fit Preprocessor<br/>Learn: <br/>mean, std, encoders"]:::training
        TrainModel["Train Model<br/>- XGBoost<br/>- TFT<br/>- NLinear"]:::training
    end

    %% Separate PyFunc Artifacts
    subgraph Artifacts["Separate PyFunc Models (Paired)"]
        direction LR

        subgraph PrepArtifact["Preprocessor PyFunc"]
            PrepModel["xgboost_preprocessor<br/>Version: 3<br/>- Fitted transformers<br/>- Imputers<br/>- Scalers<br/>- Encoders"]:::artifact
        end

        subgraph ModelArtifact["Model PyFunc"]
            TrainedModel["xgboost_model<br/>Version: 3<br/>- Weights<br/>- Hyperparams<br/>- Architecture"]:::artifact
        end
    end

    %% MLflow Registry
    subgraph Registry["MLflow Registry (Paired by Alias)"]
        direction TB
        PrepReg["xgboost_preprocessor<br/>Alias:<br/> production -> v3"]:::registry
        ModelReg["xgboost_model<br/>Alias: <br/>production -> v3"]:::registry

    end

    %% Serving Phase
    subgraph Serving["Serving Phase"]
        direction TB
        RawInput["Raw Input<br/>(Same format as training)"]:::serving
        LoadBoth["Load BOTH artifacts<br/>by SAME alias"]:::serving
        ApplyPrep["Apply Preprocessing<br/>preprocessor.transform()"]:::serving
        Inference["Model Inference<br/>model.predict()"]:::serving
        Output["Predictions"]:::serving
    end

    %% Flow
    TrainData ==> FitPreproc
    FitPreproc ==> TrainModel

    FitPreproc ==>|"Log as PyFunc"| PrepModel
    TrainModel ==>|"Log as PyFunc"| TrainedModel

    PrepModel ==> PrepReg
    TrainedModel ==> ModelReg

    PrepReg & ModelReg ==>|"Load by alias"| LoadBoth
    RawInput ==> LoadBoth
    LoadBoth ==> ApplyPrep
    ApplyPrep ==> Inference
    Inference ==> Output

    %% Notes
    Note1["Same version & alias<br/>ensures pairing"]:::note
    Note2["Load both by alias<br/>production/staging"]:::note
    Note3["Zero skew guaranteed"]:::note

    Note1 -.-> Registry
    Note2 -.-> LoadBoth
    Note3 -.-> Output

    %% Global link emphasis
    linkStyle default stroke:#2196f3,stroke-width:5px
```

**Key Design:**
- **Two separate PyFunc models**: `{model}_preprocessor` and `{model}_model`
- **Paired by version**: Both get same version number (e.g., v3)
- **Paired by alias**: Both assigned same alias (e.g., production)
- **Atomic deployment**: Load both by alias ensures consistency
- **Independent versioning**: Can track preprocessor vs model changes

---

## 3. MLflow Registry Structure

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "12px"
  }
}}%%

flowchart LR
    %% Styles
    classDef training fill:#c8e6c9,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700
    classDef registry fill:#e1bee7,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700
    classDef alias fill:#ffccbc,stroke:#d84315,stroke-width:4px,color:#000,font-weight:800
    classDef serving fill:#bbdefb,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700

    %% Training
    subgraph Train["Training Pipeline"]
        T1["Train v1<br/>MAE: 0.20"]:::training
        T2["Train v2<br/>MAE: 0.18"]:::training
        T3["Train v3<br/>MAE: 0.15"]:::training
    end

    %% Registry - Preprocessor
    subgraph RegPrep["MLflow: xgboost_preprocessor"]
        VP1["Version 1"]:::registry
        VP2["Version 2"]:::registry
        VP3["Version 3"]:::registry
    end

    %% Registry - Model
    subgraph RegModel["MLflow: xgboost_model"]
        VM1["Version 1"]:::registry
        VM2["Version 2"]:::registry
        VM3["Version 3"]:::registry
    end

    %% Aliases (Shared)
    subgraph Aliases["Shared Aliases (Pairing)"]
        Latest["latest<br/>-> v3"]:::alias
        Staging["staging<br/>-> v2"]:::alias
        Production["production<br/>-> v1"]:::alias
    end

    %% Serving
    subgraph Serve["Serving Environments"]
        Dev["Dev<br/>Load @latest"]:::serving
        Test["Test<br/>Load @staging"]:::serving
        Prod["Prod<br/>Load @production"]:::serving
    end

    %% Flow
    T1 ==>|"Log both"| VP1 & VM1
    T2 ==>|"Log both"| VP2 & VM2
    T3 ==>|"Log both"| VP3 & VM3

    VP3 & VM3 -.->|"Auto"| Latest
    VP2 & VM2 -.->|"Promote"| Staging
    VP1 & VM1 -.->|"Promote"| Production

    Latest ==>|"Load pair"| Dev
    Staging ==>|"Load pair"| Test
    Production ==>|"Load pair"| Prod

    %% Global link emphasis
    linkStyle default stroke:#2196f3,stroke-width:5px
```

**MLflow UI View:**
```
Models/
|-- xgboost_preprocessor
|   |-- Version 3 (latest)    <- Latest training
|   |-- Version 2 (staging)   <- Testing
|   +-- Version 1 (production) <- Live traffic
|
+-- xgboost_model
    |-- Version 3 (latest)    <- Paired with preprocessor v3
    |-- Version 2 (staging)   <- Paired with preprocessor v2
    +-- Version 1 (production) <- Paired with preprocessor v1
```

**Benefits:**
- **Independent tracking**: See preprocessor vs model changes separately
- **Alias pairing**: Load both by same alias guarantees consistency
- **A/B testing**: Run staging (v2) and production (v1) in parallel
- **Rollback safety**: Revert both by changing alias
- **Audit trail**: Clear version history for each component

---

## 4. Feast Feature Store Integration

**Key Features:**

| Feature | Training (Offline) | Serving (Online) |
|---------|-------------------|------------------|
| **Storage** | Parquet/BigQuery/Snowflake | Redis/DynamoDB/Cassandra |
| **Latency** | Seconds to minutes | <10ms |
| **Data Volume** | TB+ historical data | KB per entity |
| **Use Case** | Backtesting, CV, Training | Real-time inference |
| **Query Type** | Batch, time-travel | Point lookup |

---

## 5. Design Patterns

### 5.1 Strategy Pattern (Feature Retrieval)

**Benefits:**
- **Open-Closed Principle**: Add new strategies without modifying client
- **Single Responsibility**: Each strategy handles one retrieval type
- **Testability**: Easy to mock strategies

### 5.2 Facade Pattern (Simplified Feature Access)

**Benefits:**
- **Simplified interface**: 3 lines vs 30 lines of Feast API calls
- **Encapsulates complexity**: Hides Feature Store, Strategy, Factory details
- **Easy testing**: Single point to mock

---

## 6. Distributed Serving Architecture (Ray Serve)

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "12px"
  }
}}%%

flowchart TB
    %% Styles
    classDef client fill:#fff9c4,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700
    classDef ingress fill:#ffccbc,stroke:#d84315,stroke-width:4px,color:#000,font-weight:800
    classDef service fill:#c8e6c9,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700
    classDef storage fill:#e1bee7,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700

    %% Client
    Client["HTTP Client"]:::client

    %% Ingress
    subgraph Ingress["API Gateway (1 replica)"]
        API["ForecastAPI<br/>- Request validation<br/>- Response formatting"]:::ingress
    end

    %% Services
    subgraph Services["Distributed Services"]
        subgraph Feast["FeastService (2 replicas)"]
            F1["Replica 1<br/>I/O bound"]:::service
            F2["Replica 2"]:::service
        end

        subgraph Preprocess["PreprocessingService (2 replicas)"]
            P1["Replica 1<br/>CPU bound"]:::service
            P2["Replica 2"]:::service
        end

        subgraph Model["ModelService (1-4 replicas)"]
            M1["Replica 1<br/>GPU/CPU"]:::service
            M2["Replica 2"]:::service
        end
    end

    %% Storage
    subgraph Storage["External"]
        FeastStore["Feast<br/>Online Store"]:::storage
        MLflowReg["MLflow<br/>Registry<br/>Load @production"]:::storage
    end

    %% Flow
    Client ==>|"1. Request"| API
    API ==>|"2. async fetch"| F1
    API -.->|"Load balance"| F2
    F1 & F2 ==>|"Query"| FeastStore

    F1 ==>|"3. async preprocess"| P1
    F1 -.-> P2
    P1 & P2 ==>|"Load preprocessor<br/>@production"| MLflowReg

    P1 ==>|"4. async predict"| M1
    P1 -.-> M2
    M1 & M2 ==>|"Load model<br/>@production"| MLflowReg

    M1 ==>|"5. Response"| API
    API ==>|"6. JSON"| Client

    %% Global link emphasis
    linkStyle default stroke:#2196f3,stroke-width:5px
```

**Service Scaling:**

| Service | Bottleneck | Replicas | Resource | Auto-scale Trigger |
|---------|-----------|----------|----------|-------------------|
| **ForecastAPI** | HTTP routing | 1 | 0.5 CPU | Request queue |
| **FeastService** | I/O latency | 2-4 | 0.5 CPU | Feast query time |
| **PreprocessingService** | CPU computation | 2-4 | 1-2 CPU | CPU >70% |
| **ModelService** | GPU/CPU inference | 1-4 | 1 GPU or 4 CPU | GPU >80% |

**Key Features:**
- **Independent Scaling**: Each service scales based on bottleneck
- **Async Non-blocking**: Parallel execution
- **Fault Isolation**: Service crash doesn't cascade
- **Resource Efficiency**: GPU only for inference

---

## 7. Complete End-to-End Flow

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "12px"
  }
}}%%

flowchart TB
    %% Softer, modern card styles
    classDef data fill:#fffde7,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:600,rx:12,ry:12
    classDef feast fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000,font-weight:600,rx:12,ry:12
    classDef training fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000,font-weight:600,rx:12,ry:12
    classDef registry fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2.4px,color:#000,font-weight:700,rx:14,ry:14
    classDef serving fill:#fbe9e7,stroke:#d84315,stroke-width:2px,color:#000,font-weight:600,rx:12,ry:12
    classDef monitor fill:#eceff1,stroke:#37474f,stroke-width:2px,color:#000,font-weight:600,rx:12,ry:12

    %% Phase 1: Data Ingestion
    subgraph Phase1["Phase 1: Data Ingestion"]
        Raw["Raw Data<br/>CSV/DB/Stream"]:::data
        Ingest["Feature Engineering<br/>& Ingestion Script"]:::data
        FeastOff["Feast Offline Store<br/>(Parquet)"]:::feast
    end

    %% Phase 2: Materialization
    subgraph Phase2["Phase 2: Materialization"]
        Material["Materialize Script<br/>(Scheduled/On-demand)"]:::feast
        FeastOn["Feast Online Store<br/>(Redis)"]:::feast
    end

    %% Phase 3: Training
    subgraph Phase3["Phase 3: Training"]
        Pipeline["Training Pipeline<br/>- Load from Feast<br/>- CV/Tuning<br/>- Preprocessing fit"]:::training
        LogBoth["Log TWO PyFunc models<br/>xgboost_preprocessor v3<br/>xgboost_model v3"]:::training
        AssignAlias["Assign SAME alias<br/>@production -> v3<br/>(Both models)"]:::registry
    end

    %% Phase 4: Deployment
    subgraph Phase4["Phase 4: Deployment"]
        Deploy["Deploy Service<br/>FastAPI or Ray Serve"]:::serving
        LoadPair["Load BOTH by alias<br/>@production<br/>preprocessor & model v3"]:::serving
    end

    %% Phase 5: Serving
    subgraph Phase5["Phase 5: Serving"]
        Request["API Request<br/>POST /predict/feast/batch"]:::serving
        FetchFeast["Fetch Features<br/>from Feast Online"]:::feast
        ApplyPrep["Apply Preprocessing<br/>preprocessor.predict()"]:::serving
        Inference["Model Inference<br/>model.predict()"]:::serving
        Response["JSON Response"]:::serving
    end

    %% Phase 6: Monitoring
    subgraph Phase6["Phase 6: Monitoring"]
        Metrics["Collect Metrics<br/>Latency, Throughput"]:::monitor
        Dashboard["Ray Dashboard<br/>MLflow UI"]:::monitor
        Alert["Alerting<br/>Anomaly detection"]:::monitor
    end

    %% Flow connections
    Raw --> Ingest
    Ingest --> FeastOff

    FeastOff --> Material
    Material --> FeastOn

    FeastOff -->|"Historical"| Pipeline
    Pipeline --> LogBoth
    LogBoth --> AssignAlias

    AssignAlias --> Deploy
    Deploy --> LoadPair

    Request --> FetchFeast
    FetchFeast -->|"Real-time"| FeastOn
    FetchFeast --> ApplyPrep
    ApplyPrep --> Inference
    Inference --> Response

    Response --> Metrics
    Metrics --> Dashboard
    Dashboard --> Alert

    %% Feedback loop
    Alert -.->|"Retrain trigger"| Pipeline

    %% Edge styling
    linkStyle default stroke:#2196f3,stroke-width:5px
```

**Timeline:**
1. **Data Ingestion**: Daily/Hourly (Batch)
2. **Materialization**: Hourly/Real-time (Sync)
3. **Training**: Weekly/On-demand (CV + Tuning)
4. **Deployment**: After validation (Alias assignment to both models)
5. **Serving**: 24/7 (Load both by alias, apply sequentially)
6. **Monitoring**: Continuous (Metrics collection)

---

## Key Takeaways

### For Data Scientists
- **Focus on modeling**: Preprocessing handled automatically
- **Experiment tracking**: All runs logged to MLflow
- **Easy experimentation**: Change config, not code
- **Reproducibility**: Versioned artifacts + configs

### For MLOps Engineers
- **Dual PyFunc deployment**: Separate but paired models
- **Alias-based pairing**: Load both by same alias (production/staging/latest)
- **Atomic consistency**: Version matching enforced by alias
- **Scalable serving**: Ray Serve auto-scaling
- **Monitoring**: Built-in metrics and dashboards

### For Data Engineers
- **Feature Store**: Feast for feature management
- **Consistent features**: Same definitions for training/serving
- **Materialization**: Offline -> Online sync
- **Multi-entity**: Batch queries for efficiency

---


**Safety Guarantees:**
- **Version tracking**: Each component has independent version history
- **Alias pairing**: Loading by alias ensures matching versions
- **Rollback**: Change alias on both to revert
- **A/B testing**: Different aliases for different environments
- **Zero skew**: Impossible to load mismatched preprocessor+model

---
