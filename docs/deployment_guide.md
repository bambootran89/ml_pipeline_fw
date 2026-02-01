# Complete Deployment Guide - ML Pipeline Template

This comprehensive guide will walk you through deploying the ML Pipeline Template from scratch. Follow the steps exactly as written to avoid common pitfalls.

## Table of Contents
1. [Prerequisites & Setup](#1-prerequisites--setup)
2. [Quick Start - Standard Mode](#2-quick-start---standard-mode)
3. [Quick Start - Feast Mode](#3-quick-start---feast-mode)
4. [Testing Your Deployment](#4-testing-your-deployment)
5. [Understanding the Two Modes](#5-understanding-the-two-modes)
6. [Troubleshooting](#6-troubleshooting)
7. [Advanced Topics](#7-advanced-topics)

---

## 1. Prerequisites & Setup

### 1.1 Required Software

You need Docker, kubectl, and Minikube installed.

#### Install Docker
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
# IMPORTANT: Log out and log back in for group changes to take effect
```

#### Install kubectl
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

#### Install Minikube
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

### 1.2 Start Minikube

```bash
# Start Minikube with sufficient resources
minikube start --driver=docker --cpus=4 --memory=8192

# CRITICAL: Point your shell to Minikube's Docker daemon
eval $(minikube docker-env)

# Verify it's working
minikube status
docker ps
```

**IMPORTANT**: Every time you open a new terminal, run `eval $(minikube docker-env)` before building images!

### 1.3 Build Docker Images

```bash
# Navigate to project root
cd /path/to/ml_pipeline_template

# Build training image
./docker-build.sh -i train

# Build serving image
./docker-build.sh -i serve

# Verify images are built
docker images | grep ml-pipeline
```

You should see:
- `ml-pipeline-train:latest`
- `ml-pipeline-serve:latest`

---

## 2. Quick Start - Standard Mode

Standard mode is simpler and doesn't require Feast feature store. Use this for learning or simple ML workflows.

### Step 1: Deploy Everything

```bash
# Deploy standard mode with etth3.yaml experiment config
./deploy.sh -m standard -e etth3.yaml
```

This command will:
1. Deploy MLflow server
2. Create and run a training job
3. Deploy the prediction API

### Step 2: Wait for Training to Complete

```bash
# Wait for training job to finish (max 10 minutes)
kubectl wait --for=condition=complete job/training-job-standard-etth1 -n ml-pipeline --timeout=600s
```

Expected output: `job.batch/training-job-standard-etth1 condition met`

### Step 3: Verify Training Success

```bash
# Check training logs for model registration
kubectl logs -n ml-pipeline job/training-job-standard-etth1 --tail=30 | grep "Created version"
```

You should see something like:
```
Created version '20' of model 'xgboost_train_model'.
Created version '21' of model 'xgboost_preprocess'.
```

### Step 4: Restart API to Load New Model

The API starts before training completes, so we need to restart it to load the newly trained model.

```bash
# Restart the API deployment
kubectl rollout restart deployment ml-prediction-api-standard -n ml-pipeline

# Wait for new pods to be ready (this takes about 30 seconds)
kubectl wait --for=condition=ready pod -l app=ml-prediction -n ml-pipeline --timeout=120s
```

### Step 5: Test the API

```bash
# Set up port forwarding in background
kubectl port-forward -n ml-pipeline service/ml-prediction-service 8000:80 > /dev/null 2>&1 &

# Wait a moment for port forward to establish
sleep 5

# Run automated tests
./test_api.sh
```

**Expected Results**:
- Health check: ✓ (shows 3 features: HUFL, MUFL, mobility_inflow)
- `/predict`: ✓ (returns predictions)
- `/predict/multistep`: ✓ (returns predictions)

---

## 3. Quick Start - Feast Mode

Feast mode adds feature engineering with lag features and rolling statistics.

### Step 1: Prepare Feast Feature Store

Before deploying, you need to initialize the Feast feature store locally:

```bash
# 1. Ingest data into feature store
python -m mlproject.src.pipeline.feature_ops.ingest_batch_etth1 \
  --csv mlproject/data/ETTh1.csv \
  --repo feature_repo_etth1

# 2. Materialize features (Offline -> Online store)
python -m mlproject.src.pipeline.feature_ops.materialize_etth1 \
  --repo feature_repo_etth1 \
  --data feature_repo_etth1/data/features.parquet
```

### Step 2: Deploy Feast Mode

```bash
# Deploy feast mode with etth3_feast.yaml experiment config
./deploy.sh -m feast -e etth3_feast.yaml
```

### Step 3: Wait for Training

```bash
# Wait for training job to complete
kubectl wait --for=condition=complete job/training-job-feast-etth1 -n ml-pipeline --timeout=600s

# Verify model was registered
kubectl logs -n ml-pipeline job/training-job-feast-etth1 --tail=30 | grep "Created version"
```

### Step 4: Restart API to Load New Model

```bash
# Restart feast API
kubectl rollout restart deployment ml-prediction-api-feast -n ml-pipeline

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=ml-prediction -n ml-pipeline --timeout=120s
```

### Step 5: Test Feast API

```bash
# Kill any existing port forwards
pkill -f "kubectl port-forward"

# Set up port forwarding
kubectl port-forward -n ml-pipeline service/ml-prediction-service 8000:80 > /dev/null 2>&1 &

sleep 5

# Run tests
./test_api.sh
```

**Expected Results**:
- Health check: ✓ (shows 9 features including lag and engineered features)
- `/predict/feast/batch`: ✓ (returns predictions)

---

## 4. Testing Your Deployment

### 4.1 Using the Automated Test Script

The `test_api.sh` script automatically detects which mode is deployed and runs appropriate tests:

```bash
./test_api.sh
```

### 4.2 Manual Testing

#### Health Check
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

**Standard mode response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "data_type": "timeseries",
  "features": ["HUFL", "MUFL", "mobility_inflow"]
}
```

**Feast mode response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "data_type": "timeseries",
  "features": ["HUFL", "MUFL", "mobility_inflow", "HUFL_lag24", "MUFL_lag24", "HUFL_roll12_mean", "MUFL_roll12_mean", "hour_sin", "hour_cos"]
}
```

#### Standard Prediction
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00", "2020-01-01 03:00:00", "2020-01-01 04:00:00", "2020-01-01 05:00:00", "2020-01-01 06:00:00", "2020-01-01 07:00:00", "2020-01-01 08:00:00", "2020-01-01 09:00:00", "2020-01-01 10:00:00", "2020-01-01 11:00:00", "2020-01-01 12:00:00", "2020-01-01 13:00:00", "2020-01-01 14:00:00", "2020-01-01 15:00:00", "2020-01-01 16:00:00", "2020-01-01 17:00:00", "2020-01-01 18:00:00", "2020-01-01 19:00:00", "2020-01-01 20:00:00", "2020-01-01 21:00:00", "2020-01-01 22:00:00", "2020-01-01 23:00:00"],
      "HUFL": [5.827, 5.8, 5.969, 6.372, 7.153, 7.976, 8.715, 9.340, 9.763, 9.986, 10.040, 9.916, 9.609, 9.156, 8.591, 7.970, 7.338, 6.745, 6.233, 5.838, 5.582, 5.465, 5.465, 5.557],
      "MUFL": [1.599, 1.492, 1.492, 1.492, 1.492, 1.509, 1.582, 1.711, 1.896, 2.113, 2.337, 2.552, 2.742, 2.902, 3.024, 3.104, 3.137, 3.125, 3.067, 2.969, 2.838, 2.683, 2.515, 2.346],
      "mobility_inflow": [1.234, 1.456, 1.678, 1.890, 2.123, 2.456, 2.789, 3.012, 3.234, 3.456, 3.678, 3.890, 4.012, 4.123, 4.234, 4.345, 4.456, 4.567, 4.678, 4.789, 4.890, 4.901, 4.912, 4.923]
    }
  }' | python3 -m json.tool
```

#### Feast Batch Prediction
```bash
curl -s -X POST http://localhost:8000/predict/feast/batch \
  -H "Content-Type: application/json" \
  -d '{
    "time_point": "2024-01-09T00:00:00",
    "entities": [1, 2, 3]
  }' | python3 -m json.tool
```

### 4.3 Checking Pod Status

```bash
# View all pods in ml-pipeline namespace
kubectl get pods -n ml-pipeline

# Check specific deployment
kubectl get deployment -n ml-pipeline

# View logs from API
kubectl logs -n ml-pipeline deployment/ml-prediction-api-standard --tail=50

# View logs from training job
kubectl logs -n ml-pipeline job/training-job-standard-etth1 --tail=100
```

### 4.4 Checking Pod Status

## 5. Understanding the Two Modes

### 5.1 Standard Mode

**What it does**:
- Trains XGBoost model on raw timeseries data (3 features)
- Uses `etth3.yaml` experiment configuration
- Model expects 24 timesteps of data with 3 features each
- Simpler and faster to deploy

**Model Details**:
- Experiment name: `xgboost`
- Features: `HUFL`, `MUFL`, `mobility_inflow`
- Input chunk length: 24 hours
- Output chunk length: 6 hours (forecasts next 6 hours)

**API Endpoints**:
- `/health` - Check API status
- `/predict` - Single prediction (requires 24 timesteps)
- `/predict/multistep` - Multi-step ahead prediction

### 5.2 Feast Mode

**What it does**:
- Uses Feast feature store for feature engineering
- Adds lag features (24-hour lags) and rolling statistics
- Uses `etth3_feast.yaml` experiment configuration
- Creates 9 features total (3 base + 6 engineered)

**Model Details**:
- Experiment name: `feast_xgboost`
- Base features: `HUFL`, `MUFL`, `mobility_inflow`
- Engineered features: `HUFL_lag24`, `MUFL_lag24`, `HUFL_roll12_mean`, `MUFL_roll12_mean`, `hour_sin`, `hour_cos`
- More complex but potentially more accurate

**API Endpoints**:
- `/health` - Check API status
- `/predict` - Entity-based prediction
- `/predict/feast/batch` - Batch prediction for multiple entities

### 5.3 Key Differences

| Aspect | Standard Mode | Feast Mode |
|--------|--------------|------------|
| Setup Complexity | Simple | Requires Feast initialization |
| Features | 3 raw features | 9 features (3 raw + 6 engineered) |
| Training Time | Faster | Slower |
| Model Accuracy | Good baseline | Potentially better with engineered features |
| Use Case | Learning, prototyping | Production, feature experimentation |

---

## 6. Troubleshooting

### 6.1 Common Issues and Solutions

#### Issue: "ImagePullBackOff" or "ErrImagePull"

**Cause**: Kubernetes can't find your Docker images.

**Solution**:
```bash
# Ensure you're using Minikube's Docker daemon
eval $(minikube docker-env)

# Rebuild images
./docker-build.sh -i train
./docker-build.sh -i serve

# Verify images exist
docker images | grep ml-pipeline
```

#### Issue: Training job completes but no model is registered

**Cause**: The training command in `k8s/job-training-standard.yaml` was commented out (this was a bug that we fixed).

**Solution**: The fix is already in place. The training command should look like:
```yaml
command:
- /bin/bash
- -c
- |
  # [STANDARD] Simple training
  cd {{PROJECT_ROOT}} && python -m mlproject.src.pipeline.dag_run train -e {{PROJECT_ROOT}}/mlproject/configs/experiments/{{EXPERIMENT_CONFIG}} -p {{PROJECT_ROOT}}/mlproject/configs/pipelines/{{PIPELINE_CONFIG}}
```

#### Issue: API returns old predictions or "model not found"

**Cause**: API loaded before training completed, or didn't reload after new model was trained.

**Solution**:
```bash
# Always restart API after training completes
kubectl rollout restart deployment ml-prediction-api-standard -n ml-pipeline
# OR for feast
kubectl rollout restart deployment ml-prediction-api-feast -n ml-pipeline

# Wait for rollout to complete
kubectl rollout status deployment ml-prediction-api-standard -n ml-pipeline
```

#### Issue: "Connection refused" when testing API

**Cause**: Port forwarding is not active.

**Solution**:
```bash
# Kill any existing port forwards
pkill -f "kubectl port-forward"

# Start new port forward
kubectl port-forward -n ml-pipeline service/ml-prediction-service 8000:80 > /dev/null 2>&1 &

# Wait and test
sleep 5
curl http://localhost:8000/health
```

#### Issue: Minikube fails to start with Docker errors

**Cause**: Docker daemon not running.

**Solution**:
```bash
# For WSL2 or systems without systemd
sudo service docker start

# For systems with systemd
sudo systemctl start docker

# Verify Docker is running
docker ps

# Then start Minikube
minikube start --driver=docker --cpus=4 --memory=8192
```

#### Issue: Training job stays in "Pending" state

**Check pod status**:
```bash
kubectl get pods -n ml-pipeline
kubectl describe pod -n ml-pipeline <pod-name>
```

Common causes:
- Insufficient resources: Increase Minikube resources
- ImagePullBackOff: See solution above
- Volume mount issues: Ensure you're running from project root

#### Issue: "Serving config not found" or API fails to start

**Cause**: Serving configuration was not generated for your pipeline.

**Solution**: The deployment script now auto-generates serving configs, but if you encounter issues:
```bash
# Manually generate serving config
python -c "
from mlproject.src.generator.orchestrator import ConfigGenerator

generator = ConfigGenerator(
    train_config_path='mlproject/configs/pipelines/your_pipeline.yaml',
    experiment_config_path='mlproject/configs/experiments/your_experiment.yaml'
)
generator.generate_all('mlproject/configs/generated', alias='latest')
"

# Verify it was created
ls mlproject/configs/generated/*_serve.yaml
```

### 6.2 Debugging Commands

```bash
# Check cluster status
minikube status
kubectl cluster-info

# View all resources in namespace
kubectl get all -n ml-pipeline

# Describe a stuck pod
kubectl describe pod <pod-name> -n ml-pipeline

# View pod logs
kubectl logs <pod-name> -n ml-pipeline

# Follow logs in real-time
kubectl logs -f <pod-name> -n ml-pipeline

# Execute command in running pod
kubectl exec -it <pod-name> -n ml-pipeline -- bash

# Check events
kubectl get events -n ml-pipeline --sort-by='.lastTimestamp'
```

### 6.3 Complete Reset

If things are completely broken, start fresh:

```bash
# Delete namespace (removes all resources)
kubectl delete namespace ml-pipeline

# Wait a moment
sleep 10

# Redeploy from scratch
./deploy.sh -m standard -e etth3.yaml
```

---

## 7. Advanced Topics

### 7.1 Switching Between Modes

To switch from standard to feast (or vice versa):

```bash
# Current mode pods will be deleted automatically
./deploy.sh -m feast -e etth3_feast.yaml

# Wait for training
kubectl wait --for=condition=complete job/training-job-feast-etth1 -n ml-pipeline --timeout=600s

# Restart API
kubectl rollout restart deployment ml-prediction-api-feast -n ml-pipeline
```

### 7.2 Using Custom Configurations

You can specify custom pipeline and experiment configs. The deployment script automatically generates serving configurations if they don't exist.

#### Example: Using Different Pipelines

```bash
# Deploy with nested sub-pipeline
./deploy.sh -m standard -p nested_suppipeline.yaml -e etth3.yaml

# Deploy with parallel ensemble
./deploy.sh -m standard -p parallel_ensemble.yaml -e etth3.yaml

# Deploy with conditional branching
./deploy.sh -m standard -p conditional_branch.yaml -e etth3.yaml

# Dry run (generate manifests without deploying)
./deploy.sh -m standard -e etth3.yaml -d
```

#### How Serving Config Auto-Generation Works

The deployment script automatically:
1. **Checks** if `mlproject/configs/generated/<pipeline>_serve.yaml` exists
2. **Generates** it if missing using `ConfigGenerator`
3. **Uses** the generated config for the API deployment

**Example**:
```bash
# If parallel_ensemble_serve.yaml doesn't exist, it will be auto-generated
./deploy.sh -m standard -p parallel_ensemble.yaml -e etth3.yaml
```

Output:
```
[Pre-Deploy] Generating serving config: parallel_ensemble_serve.yaml
[ConfigGenerator] Successfully generated: mlproject/configs/generated/parallel_ensemble_serve.yaml
```

**Manual Generation** (if you want to inspect before deploying):
```bash
python -c "
from mlproject.src.generator.orchestrator import ConfigGenerator

generator = ConfigGenerator(
    train_config_path='mlproject/configs/pipelines/your_pipeline.yaml',
    experiment_config_path='mlproject/configs/experiments/your_experiment.yaml'
)
paths = generator.generate_all('mlproject/configs/generated', alias='latest')
print(f'Generated configs: {paths}')
"
```

Generated files will be in `mlproject/configs/generated/`:
- `<pipeline>_serve.yaml` - For serving/inference
- `<pipeline>_eval.yaml` - For evaluation
- `<pipeline>_tune.yaml` - For hyperparameter tuning (if include_tune=True)

### 7.3 Accessing MLflow UI

```bash
# Forward MLflow port
kubectl port-forward -n ml-pipeline service/mlflow-service 5000:5000 &

# Open in browser
# http://localhost:5000
```

You can view:
- All training runs
- Model metrics and parameters
- Registered models and versions
- Artifacts

### 7.4 Retraining Models

To retrain with updated data or configuration:

```bash
# 1. Update your data or configs
# 2. Redeploy (old training jobs are automatically deleted)
./deploy.sh -m standard -e etth3.yaml

# 3. Wait for training
kubectl wait --for=condition=complete job/training-job-standard-etth1 -n ml-pipeline --timeout=600s

# 4. Restart API to load new model
kubectl rollout restart deployment ml-prediction-api-standard -n ml-pipeline
```

### 7.5 Scaling the API

```bash
# Scale to more replicas for higher load
kubectl scale deployment ml-prediction-api-standard --replicas=4 -n ml-pipeline

# Verify
kubectl get deployment ml-prediction-api-standard -n ml-pipeline
```

### 7.6 Cleanup

To remove all resources:

```bash
# Delete the namespace (removes everything)
kubectl delete namespace ml-pipeline

# Stop Minikube
minikube stop

# Delete Minikube cluster (complete reset)
minikube delete
```

---

## Quick Reference - Common Commands

```bash
# Build images
eval $(minikube docker-env)
./docker-build.sh -i train
./docker-build.sh -i serve

# Deploy standard mode
./deploy.sh -m standard -e etth3.yaml

# Deploy feast mode
./deploy.sh -m feast -e etth3_feast.yaml

# Wait for training
kubectl wait --for=condition=complete job/training-job-standard-etth1 -n ml-pipeline --timeout=600s

# Restart API
kubectl rollout restart deployment ml-prediction-api-standard -n ml-pipeline

# Port forward
kubectl port-forward -n ml-pipeline service/ml-prediction-service 8000:80 &

# Test
./test_api.sh

# View pods
kubectl get pods -n ml-pipeline

# View logs
kubectl logs -n ml-pipeline job/training-job-standard-etth1

# Cleanup
kubectl delete namespace ml-pipeline
```

---

## Summary Checklist

Before deploying:
- [ ] Docker is installed and running
- [ ] kubectl is installed
- [ ] Minikube is installed and started
- [ ] You've run `eval $(minikube docker-env)`
- [ ] You've built both Docker images
- [ ] You're in the project root directory

For successful deployment:
- [ ] Training job completes (check with kubectl wait)
- [ ] Model is registered in MLflow (check logs)
- [ ] API is restarted after training
- [ ] Port forwarding is active
- [ ] Tests pass with `./test_api.sh`

If something fails:
- [ ] Check pod status: `kubectl get pods -n ml-pipeline`
- [ ] Check logs: `kubectl logs -n ml-pipeline <pod-name>`
- [ ] Verify images: `docker images | grep ml-pipeline`
- [ ] Try complete reset: `kubectl delete namespace ml-pipeline`
