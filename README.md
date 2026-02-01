# Production-Grade ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-2b5b84?logo=python&logoColor=white)](https://www.python.org/)
[![Feast](https://img.shields.io/badge/Feast-Feature%20Store-388e3c?logo=feast&logoColor=white)](https://feast.dev/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20Server-0d47a1?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Tuning-6a1b9a?logo=optuna&logoColor=white)](https://optuna.org/)
[![Ray Serve](https://img.shields.io/badge/Ray%20Serve-Serving-1565c0?logo=ray&logoColor=white)](https://docs.ray.io/en/latest/serve/index.html)
[![Hydra](https://img.shields.io/badge/Hydra-Config-ef6c00?logo=hydra&logoColor=white)](https://hydra.cc/)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20%7C%20K8s-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

## Overview
This is a **DAG-driven ML framework** built for production-grade scalability and reliability. It bridges the gap between research and deployment by focusing on:
- **DAG-Based Orchestration**: Unified execution engine (`dag_run`) for training, tuning, and serving.
- **Feature Store-Centric**: Native **Feast** integration for consistent offline/online feature management.
- **Zero-Skew Deployment**: Bundled preprocessors and models to eliminate training-serving discrepancy.
- **Hybrid Serving**: High-performance inference via **FastAPI** or distributed scaling with **Ray Serve**.
- **Automated MLOps**: Built-in **Optuna** tuning, **MLflow** tracking, and **Kubernetes** readiness.

## Key Takeaways

### For Data Scientists
- **Focus on modeling**: Preprocessing handled automatically
- **Easy experimentation**: Change config, not code
- **Config-Driven**: Define complex pipelines and hyperparameters in YAML; no code changes needed for experiments.
- **Auto-Optimization**: Integrated Bayesian tuning (Optuna) and automated backtesting.
- **Reproducibility**: Full experiment tracking and versioned artifacts via MLflow.

### For MLOps Engineers
- **Zero-skew deployment**: Preprocessor + Model bundled
- **Alias-based rollout**: production/staging/latest
- **Scalable serving**: Ray Serve auto-scaling
- **Monitoring**: Built-in metrics and dashboards

### For Data Engineers
- **Feature Store Integration**: Centralized management using Feast.
- **Consistent features**: Same feature definitions used for both training and online serving.
- **Materialization**: Automated Offline -> Online sync for real-time feature availability.
- **Multi-entity support**: Optimized batch queries and point-in-time joins for efficiency.

## Documentation
- [Architecture](docs/architecture.md)
- [Directory Structure](docs/directorystructure.md)
- [Offline Workflow](docs/offlineworkflow.md)
- [Online Workflow](docs/onlineworkflow.md)
- [Preprocessing](docs/preprocessing.md)
- [Adding New Model](docs/adding_new_model.md)
- [Pipeline Orchestration](docs/pipeline_orchestration.md)
- **[Simple API Generation Guide](docs/api_generation_guide.md)** - Quick start for FastAPI and Ray Serve
- [API Generation and Running](docs/readme_api.md) - Complete reference
- **[Docker Setup Guide](README.Docker.md)** - Docker build, test, and run
- **[Deployment Guide](docs/deployment_guide.md)** - **Primary Guide** for Kubernetes deployment (Zero-to-Hero)
- **[Separated Docker Architecture](README.Docker.Separated.md)** - Best practices for Train vs. Serve images
- [Docker Setup Guide (Legacy/All-in-One)](README.Docker.md) - Optional multi-stage build reference
- **[Verification Guide](docs/verification_guide.md)** - Testing and verification scripts

# Getting Started
## 1. Prerequisites
Python 3.10+

Virtual Environment (recommended)

## 2. Installation

```bash
# Clone the repository
git clone https://github.com/bambootran89/ml_pipeline_template.git
cd ml_pipeline_template

# 1. Create virtual environment using Python 3.10
python3.10 -m venv py3.10

# 2. Activate virtual environment (Linux/macOS)
source py3.10/bin/activate

# 3. Install dependencies
pip install -r requirements/prod.txt
pip install -r requirements/dev.txt   # optional, for testing

# 4. Install project in editable mode
pip install -e .
```

## 3. Quality Assurance
We enforce code quality via pre-commit hooks.
```bash
# Run all tests
make test

# Auto-format code
make style
```

## 4. Quick Start
Start the MLflow server to visualize results:
```bash
mlflow ui --port 5000
```

Run a standard training experiment:
```bash
python -m mlproject.src.pipeline.compat.v1.run train \
  --config mlproject/configs/experiments/etth1.yaml
```

Run a eval experiment.  (alias: latest, production, staging)
```bash
python -m mlproject.src.pipeline.compat.v1.run eval \
  --config mlproject/configs/experiments/etth1.yaml \
  --alias latest
```

Run a serving. (alias: latest, production, staging)
```bash
python -m mlproject.src.pipeline.compat.v1.run serve \
  --config mlproject/configs/experiments/etth1.yaml \
  --input ./sample_input.csv \
  --alias latest
```

## DAG-Based Pipeline System

### Training
```bash
# Long form, not using feast
python -m mlproject.src.pipeline.dag_run train \
    --experiment mlproject/configs/experiments/etth3.yaml \
    --pipeline mlproject/configs/pipelines/standard_train.yaml
```

### Evaluation
```bash
python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/standard_eval.yaml \
    -a latest  # or production, staging
```

### Serving (CSV input)
```bash
python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/standard_serve.yaml \
    -i ./sample_input.csv \
    -a latest
```

### Hyperparameter Tuning
```bash
python -m mlproject.src.pipeline.dag_run tune \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/standard_tune.yaml \
    -n 5  # number of trials
```

# Feature Store Integration

## Local Development

For local development, create and populate the Feast feature store manually.

```bash
# Ingest Batch: Typically runs in large-scale data processing jobs (Spark/Pandas)
python -m mlproject.src.pipeline.feature_ops.ingest_batch_etth1 \
    --csv mlproject/data/ETTh1.csv \
    --repo feature_repo_etth1

python -m mlproject.src.pipeline.feature_ops.ingest_titanic \
    --csv mlproject/data/titanic.csv \
    --repo titanic_repo
```

Sync data to the online store for API inference:
```bash
# Materialize: Updates the online store from offline features
python -m mlproject.src.pipeline.feature_ops.materialize_etth1 \
    --repo feature_repo_etth1 \
    --data feature_repo_etth1/data/features.parquet

python -m mlproject.src.pipeline.feature_ops.materialize_titanic \
    --repo titanic_repo \
    --data titanic_repo/data/titanic.parquet
```

## Kubernetes Deployment

Feature ingestion is handled automatically via init containers. No manual preparation needed.
See [Deployment Guide](docs/deployment_guide.md) for details.

Training with automatic data loading from Feast:
```bash
python -m mlproject.src.pipeline.compat.v1.run train \
  --config mlproject/configs/experiments/etth1_feast.yaml
```

Evaluation with automatic data loading from Feast:
```bash
python -m mlproject.src.pipeline.compat.v1.run eval \
  --config mlproject/configs/experiments/etth1_feast.yaml
```

Run the serving stage with automatic feature loading from Feast
```bash
python -m mlproject.src.pipeline.compat.v1.run serve \
    --config mlproject/configs/experiments/etth1_feast.yaml
```

```bash
# Training
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/etth3_feast.yaml \
    -p mlproject/configs/pipelines/standard_train.yaml

# Evaluation
python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/experiments/etth3_feast.yaml \
    -p mlproject/configs/pipelines/standard_eval.yaml \
    -a latest  # or production, staging

# Serving
python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/experiments/etth3_feast.yaml \
    -p mlproject/configs/pipelines/standard_serve.yaml \
    -a latest \
    --time_point "now"

# Hyperparameter Tuning
python -m mlproject.src.pipeline.dag_run tune \
    -e mlproject/configs/experiments/etth3_feast.yaml \
    -p mlproject/configs/pipelines/standard_tune.yaml \
    -n 5  # number of trials
```

# Workflows & Capabilities

## 1. Cross-Validation (Backtesting)
Validates model stability across time folds.
```bash
python -m mlproject.src.pipeline.compat.v1.run cv \
  --config mlproject/configs/experiments/etth2.yaml
```

## 2. Hyperparameter Tuning (Optuna)
Runs Bayesian Optimization to find best parameters, then auto-retrains the best model.
```bash
python -m mlproject.src.pipeline.compat.v1.run tune \
  --config mlproject/configs/experiments/etth1_tuning.yaml


python -m mlproject.src.pipeline.compat.v1.run tune \
  --config mlproject/configs/experiments/etth1_feast.yaml
```

## 3. Serving (Inference)
Deploys the model using FastAPI.
```bash

# Start FastAPI
CONFIG_PATH=mlproject/configs/experiments/etth1_feast.yaml uvicorn mlproject.serve.api:app --reload
```
Deploys the model using Ray Serve.
```bash

# OR Start Ray Serve
CONFIG_PATH=mlproject/configs/experiments/etth1_feast.yaml python mlproject/serve/ray/ray_deploy.py
```
Test API services: mlproject/configs/experiments/etth1.yaml
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json"
  -d '{
    "data": {
      "HUFL": [-0.15, 0.08, 0.01, -0.01, 0.21, -0.15, 0.12, 0.05, -0.08, 0.18, -0.12, 0.22, 0.03, -0.18, 0.15, -0.05, 0.28, -0.22, 0.08, -0.15, 0.32, -0.28, 0.12, -0.18],
      "MUFL": [1.14, 1.06, 0.93, 1.11, 0.96, 1.05, 0.98, 1.12, 0.95, 1.08, 0.92, 1.15, 0.88, 1.22, 0.85, 1.18, 0.82, 1.25, 0.78, 1.32, 0.75, 1.38, 0.72, 1.42],
      "mobility_inflow": [1.24, 4.42, 7.28, 1.03, 0.73, 2.5, 3.2, 4.1, 1.8, 5.3, 2.1, 6.4, 1.5, 7.8, 3.6, 4.9, 2.7, 8.2, 1.9, 5.5, 3.8, 6.7, 2.3, 4.2]
    }
  }'
```

Test API services: mlproject/configs/experiments/etth1_feast.yaml
```bash
curl -X POST http://localhost:8000/predict/feast/batch \
  -H "Content-Type: application/json" \
  -d '{"time_point":"now","entities":[1,2,3,4,5],"entity_key":"location_id"}'

curl -X POST http://localhost:8000/predict/feast \
  -H "Content-Type: application/json" \
  -d '{"time_point":"now","entities":[1,2,3,4,5],"entity_key":"location_id"}'
```

# Docker & Kubernetes

For professional deployment, use the automated build and deployment scripts.

### 1. Build Images
```bash
# Build training and serving images
./docker-build.sh -i train
./docker-build.sh -i serve
```

### 2. Deploy to Kubernetes (Local or Cluster)
Use the enhanced deployment script with automatic feature ingestion.
```bash
# Deploy in Feast mode (Advanced)
./deploy.sh -m feast

# Deploy in Standard mode (Simple)
./deploy.sh -m standard

# Deploy specific scenario (e.g., Tabular)
./deploy.sh -m feast -p conditional_branch_tabular.yaml -e tabular.yaml
```

Feast mode automatically:
- Runs feature ingestion in init containers
- Creates necessary ConfigMaps
- Sets up feature repositories
- No manual preparation required

### 3. Verify Deployment
```bash
# Quick test of currently deployed API
./test_api.sh

# OR comprehensive test of all deployment scenarios (4 tests)
./verify_all_deployments.sh
```

## Cleanup & Maintenance

To remove all deployed resources and stop local infrastructure:
```bash
# Cleanup K8s resources
./cleanup.sh

# Stop Minikube (if applicable)
minikube stop
```

---
For more details, see the **[Zero-to-Hero Deployment Guide](docs/deployment_guide.md)**.
