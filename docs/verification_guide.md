# Verification and Testing Guide

This document describes the comprehensive verification scripts available for testing all pipeline configurations, frameworks, and integrations.

## Overview

The project includes six main verification scripts that test different aspects of the system:

1. **verify_all_pipelines.sh** - Complete pipeline verification across all configurations
2. **verify_legacy_run.sh** - Backward compatibility testing for v1.run interface
3. **verify_full_lifecycle.sh** - End-to-end ML lifecycle testing
4. **verify_feast_integration.sh** - Comprehensive Feast feature store testing
5. **verify_config_generation.sh** - Config generation and execution validation
6. **verify_all_deployments.sh** - Kubernetes deployment verification for all modes

## Prerequisites

Ensure your environment is properly configured:

```bash
# Activate virtual environment
source mlproject_env/bin/activate

# Set PYTHONPATH
export PYTHONPATH=.

# Verify installations
python -m mlproject.src.pipeline.dag_run --help
```

---

## 1. Complete Pipeline Verification

**Script:** `verify_all_pipelines.sh`

Tests all pipeline configurations with both FastAPI and Ray Serve frameworks.

### Usage

```bash
./verify_all_pipelines.sh
```

### What It Tests

For each pipeline configuration in `mlproject/configs/pipelines/`:

1. **Training Phase**
   - Detects experiment type (tabular, timeseries, or Feast)
   - Executes training with appropriate experiment config
   - Validates training completion

2. **Config Generation**
   - Generates serve configuration from training pipeline
   - Verifies generated file existence

3. **Serving Phase** (both FastAPI and Ray)
   - Starts API server on port 8082
   - Waits for health check endpoint
   - Validates JSON response structure

4. **API Verification**
   - **Tabular**: Batch prediction with 5 samples
   - **Timeseries**: Multi-step prediction (18 steps)
   - **Feast**: Single and batch entity predictions

### Output

```
=======================================================
   VERIFYING ALL PIPELINES IN mlproject/configs/pipelines
=======================================================

-------------------------------------------------------
>>> Processing Pipeline: standard_train.yaml
-------------------------------------------------------
Detected Type: TIMESERIES
--- [1/4] Training ---
Training PASS
--- [2/4] Generating Serve Config ---
Generation PASS
--- [3/4] Serving (fastapi) ---
Serving Health Check PASS (fastapi)
--- [4/4] Verifying API (fastapi) ---
>>> Timeseries Multistep (fastapi) VERIFICATION PASSED
--- [3/4] Serving (ray) ---
Serving Health Check PASS (ray)
--- [4/4] Verifying API (ray) ---
>>> Timeseries Multistep (ray) VERIFICATION PASSED

=======================================================
   ALL PIPELINES VERIFIED SUCCESSFULLY
=======================================================
```

### Logs

All logs are stored in `tmplogs/`:
- `{pipeline}_train.log`
- `{pipeline}_gen.log`
- `{pipeline}_fastapi_serve.log`
- `{pipeline}_ray_serve.log`

---

## 2. Legacy Compatibility Verification

**Script:** `verify_legacy_run.sh`

Ensures backward compatibility with the legacy v1.run interface.

### Usage

```bash
./verify_legacy_run.sh
```

### What It Tests

1. **Legacy Train**
   ```bash
   python -m mlproject.src.pipeline.compat.v1.run train --config etth1.yaml
   ```

2. **Legacy Eval**
   ```bash
   python -m mlproject.src.pipeline.compat.v1.run eval --config etth1.yaml --alias latest
   ```

3. **Legacy Serve**
   ```bash
   python -m mlproject.src.pipeline.compat.v1.run serve --config etth1.yaml --input ETTh1.csv --alias latest
   ```

### Use Case

Run this when making changes to ensure existing pipelines using the v1 interface continue to work.

---

## 3. Full Lifecycle Verification

**Script:** `verify_full_lifecycle.sh`

Complete end-to-end testing of the ML lifecycle including nested pipelines.

### Usage

```bash
./verify_full_lifecycle.sh
```

### What It Tests

**Phase 1-5: Standard Pipeline**

1. Training with `etth3.yaml` + `standard_train.yaml`
2. Config generation (eval, serve, tune)
3. Evaluation execution
4. Hyperparameter tuning (1 trial)
5. Serving API health check and predictions

**Phase 6: Nested Pipeline**

1. Training with `nested_suppipeline.yaml`
2. Serve config generation
3. API deployment and health check
4. Prediction with multiple model outputs

### API Testing

**Timeseries Standard Prediction:**
```bash
curl -X POST http://localhost:8082/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "date": ["2020-01-01 00:00:00", ...],
      "HUFL": [5.827, 5.8, ...],
      "MUFL": [1.599, 1.492, ...],
      "mobility_inflow": [1.234, 1.456, ...]
    }
  }'
```

**Timeseries Multi-step Prediction (18 steps):**
```bash
curl -X POST http://localhost:8082/predict/multistep \
  -H "Content-Type: application/json" \
  -d '{
    "data": { ... },
    "steps_ahead": 18
  }'
```

**Tabular Batch Prediction:**
```bash
curl -X POST http://localhost:8082/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "Pclass": [1, 2, 3, ...],
      "Age": [22.0, 38.0, ...]
    },
    "return_probabilities": false
  }'
```

---

## 4. Feast Integration Verification

**Script:** `verify_feast_integration.sh`

Comprehensive testing of Feast feature store integration across all pipeline types.

### Usage

```bash
./verify_feast_integration.sh
```

### Test Matrix

The script tests 9 combinations of pipelines and experiments:

| Pipeline | Experiment | Type | Frameworks |
|----------|-----------|------|------------|
| standard_train | etth3_feast | Timeseries + Feast | FastAPI, Ray |
| standard_train | feast_tabular | Tabular + Feast | FastAPI, Ray |
| kmeans_then_xgboost | etth3_feast | Clustering + TS | FastAPI, Ray |
| kmeans_then_xgboost | feast_tabular | Clustering + Tab | FastAPI, Ray |
| parallel_ensemble | etth3_feast | Multi-model TS | FastAPI, Ray |
| parallel_ensemble | feast_tabular | Multi-model Tab | FastAPI, Ray |
| nested_suppipeline | etth3_feast | Nested TS | FastAPI, Ray |
| dynamic_adapter | etth3_feast | Dynamic TS | FastAPI, Ray |
| dynamic_adapter | feast_tabular | Dynamic Tab | FastAPI, Ray |

### Feast API Endpoints

**Single Entity Prediction:**
```bash
curl -X POST http://localhost:8082/predict/feast \
  -H "Content-Type: application/json" \
  -d '{
    "time_point": "2024-01-09T00:00:00",
    "entities": [1]
  }'
```

**Batch Entity Prediction:**
```bash
curl -X POST http://localhost:8082/predict/feast/batch \
  -H "Content-Type: application/json" \
  -d '{
    "time_point": "2024-01-09T00:00:00",
    "entities": [1, 2, 3],
    "entity_key": "location_id"
  }'
```

### Output Location

Logs: `tmplogs/feast_verify/{pipeline}_{experiment}_{train|gen|fastapi_serve|ray_serve}.log`

---

## 5. Config Generation Verification

**Script:** `verify_config_generation.sh`

Tests configuration generation and execution for all pipeline types.

### Usage

```bash
./verify_config_generation.sh
```

### What It Tests

For each pipeline configuration:

1. **Training** - Execute training pipeline
2. **Generation** - Generate eval, serve, and tune configs using `--type all`
3. **Eval Execution** - Run generated eval config
4. **Tune Execution** - Run generated tune config (1 trial)
5. **Serve Execution** - Run generated serve config with DAG runner

### Config Generation Types

```bash
# Generate all config types
python -m mlproject.src.pipeline.dag_run generate \
  -t mlproject/configs/pipelines/standard_train.yaml \
  -e mlproject/configs/experiments/etth3.yaml \
  --type all

# Generated files:
# - mlproject/configs/generated/standard_train_eval.yaml
# - mlproject/configs/generated/standard_train_serve.yaml
# - mlproject/configs/generated/standard_train_tune.yaml
```

---

## 6. Kubernetes Deployment Verification

**Script:** `verify_all_deployments.sh`

Comprehensive end-to-end testing of Kubernetes deployments across all modes and data types.

### Usage

```bash
./verify_all_deployments.sh
```

### What It Tests

This script performs complete deployment verification for 4 scenarios:

1. **Feast Tabular Deployment**
   - Deploys with `feast_tabular.yaml` experiment config
   - Uses Titanic dataset with tabular features
   - Tests ConfigMap creation and Feast integration
   - Verifies entity-based predictions

2. **Feast Time Series Deployment**
   - Deploys with `etth3_feast.yaml` experiment config
   - On-the-fly feature ingestion via init container
   - Tests 9 engineered features (lag, rolling, time)
   - Verifies time series predictions with Feast features

3. **Standard Tabular Deployment**
   - Deploys with `tabular.yaml` experiment config
   - Simple tabular prediction without Feast
   - Tests basic classification endpoints

4. **Standard Time Series Deployment**
   - Deploys with `etth3.yaml` experiment config
   - Time series forecasting without feature engineering
   - Tests multi-step prediction endpoints

### Test Workflow

For each scenario, the script performs:

1. **Deployment** - Runs `deploy.sh` with appropriate flags
2. **Training Verification** - Waits for training job completion (max 10 minutes)
3. **Model Registration Check** - Verifies model logged to MLflow
4. **ConfigMap Creation** - Creates Feast ConfigMap (Feast mode only)
5. **API Restart** - Restarts API pods to load new model
6. **Port Forwarding** - Sets up local access on port 8000
7. **API Testing**:
   - Health check endpoint
   - Standard prediction endpoints
   - Feast-specific endpoints (`/predict/feast`, `/predict/feast/batch`)

### Output

```bash
====================================================
   Comprehensive ML Pipeline Deployment Tests
====================================================

Checking prerequisites...
[PASS] Kubernetes cluster accessible
[PASS] Docker images found

Starting deployment tests...

---------------------------------------
         FEAST MODE TESTS
---------------------------------------

==================================================
Test 1: feast_feast_tabular.yaml
  Mode: feast
  Experiment: feast_tabular.yaml
  Data Type: tabular
==================================================
[1/6] Deploying...
[PASS] Deployment submitted
[2/6] Waiting for training job: training-job-feast-titanic
[PASS] Training completed
[PASS] Model registered: Created version '42' of model 'xgboost_train_model'.
[3/6] Updating Feast ConfigMap...
[PASS] ConfigMap updated
[4/6] Restarting API pods...
[PASS] API pods ready
[5/6] Setting up port forward...
[PASS] Port forward ready
[6/6] Running API tests...
  Testing /health endpoint...
  [PASS] Health check PASS
  Testing /predict (tabular single)...
  [PASS] Single prediction PASS: {'train_model_predictions': [0]}
  Testing /predict/feast endpoint...
  [PASS] /predict/feast PASS: 3 entities
  Testing /predict/feast/batch endpoint...
  [PASS] /predict/feast/batch PASS: 5 entities
[PASS] All API tests PASSED

[... Tests 2-4 ...]

====================================================
              TEST SUMMARY
====================================================

[PASS] feast_feast_tabular.yaml - ALL TESTS PASSED
[PASS] feast_etth3_feast.yaml - ALL TESTS PASSED
[PASS] standard_tabular.yaml - ALL TESTS PASSED
[PASS] standard_etth3.yaml - ALL TESTS PASSED

Total Tests:   4
Passed:        4
Failed:        0
Skipped:       0

====================================================

      [PASS] ALL TESTS PASSED SUCCESSFULLY!

====================================================
```

### Test Details

**Feast Tabular Test Endpoints:**
```bash
# Health check - expects 7 features
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"Pclass": [3], "Age": [25.0], ...}}'

# Entity-based prediction
curl -X POST http://localhost:8000/predict/feast \
  -H "Content-Type: application/json" \
  -d '{"entities": [1], "time_point": "now"}'

# Batch entity prediction
curl -X POST http://localhost:8000/predict/feast/batch \
  -H "Content-Type: application/json" \
  -d '{"entities": [1, 2, 3, 4, 5], "time_point": "now"}'
```

**Feast Time Series Test Endpoints:**
```bash
# Health check - expects 9 features
curl http://localhost:8000/health

# Time series prediction with Feast features
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"date": [...], "HUFL": [...], "MUFL": [...], ...}}'
```

**Standard Mode Test Endpoints:**
```bash
# Health check - expects 3 features (time series) or 7 features (tabular)
curl http://localhost:8000/health

# Standard prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {...}}'
```

### Prerequisites

Before running the script:

```bash
# Kubernetes cluster must be running
minikube status  # or kubectl cluster-info

# Docker images must be built
docker images | grep ml-pipeline
# Should show: ml-pipeline-train:latest and ml-pipeline-serve:latest

# Build if missing
eval $(minikube docker-env)
./docker-build.sh -i train
./docker-build.sh -i serve

# Activate Python environment
source mlproject_env/bin/activate
export PYTHONPATH=.
```

### Logs and Debugging

Deployment logs are saved to `/tmp/deploy_{test_name}.log`:

```bash
# View deployment logs for specific test
cat /tmp/deploy_feast_feast_tabular.yaml.log
cat /tmp/deploy_feast_etth3_feast.yaml.log
cat /tmp/deploy_standard_tabular.yaml.log
cat /tmp/deploy_standard_etth3.yaml.log
```

Port forwarding logs:
```bash
cat /tmp/port-forward.log
```

Check live pod status:
```bash
kubectl get pods -n ml-pipeline
kubectl logs -n ml-pipeline <pod-name>
kubectl describe pod -n ml-pipeline <pod-name>
```

### Use Cases

**Run before commits:**
```bash
# Verify all deployment scenarios still work after code changes
./verify_all_deployments.sh
```

**Run after infrastructure changes:**
```bash
# Test deployment after K8s manifest updates
./verify_all_deployments.sh
```

**Run for CI/CD:**
```bash
# Automated testing in CI pipeline
./verify_all_deployments.sh
if [ $? -eq 0 ]; then
  echo "All deployments verified, proceeding with release"
else
  echo "Deployment verification failed, blocking release"
  exit 1
fi
```

### Cleanup

The script automatically cleans up port forwards on exit. To fully clean up deployments:

```bash
# Remove all K8s resources
kubectl delete namespace ml-pipeline

# OR use cleanup script
./cleanup.sh
```

---

## Troubleshooting

### Port Already in Use

**Error:** `Address already in use: Port 8082`

**Solution:**
```bash
# Find and kill process using port 8082
lsof -ti:8082 | xargs kill -9

# Alternative (Linux)
fuser -k 8082/tcp
```

### Ray Not Stopping

**Error:** Ray serve won't stop between tests

**Solution:**
```bash
# Stop Ray runtime
ray stop

# Force kill all Ray processes
pkill -9 ray
```

### Health Check Timeout

**Error:** Server health check fails after 45 retries

**Diagnosis:**
```bash
# Check server logs
tail -f tmplogs/{pipeline}_fastapi_serve.log
tail -f tmplogs/{pipeline}_ray_serve.log

# Check if server started
ps aux | grep python | grep serve

# Check port status
netstat -an | grep 8082
```

### Verification Script Hangs

**Cause:** Previous server process not cleaned up

**Solution:**
```bash
# Clean up all resources
fuser -k 8082/tcp || true
ray stop || true
pkill -f "serve_api.sh" || true

# Run cleanup trap manually
bash -c "trap 'fuser -k 8082/tcp; ray stop' EXIT; ./verify_all_pipelines.sh"
```

### Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'mlproject'`

**Solution:**
```bash
# Reinstall in editable mode
pip install -e .

# Verify PYTHONPATH
export PYTHONPATH=.
echo $PYTHONPATH
```

---

## CI/CD Integration

### GitHub Actions

Add to `.github/workflows/test.yml`:

```yaml
name: Verification Tests
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  verify-pipelines:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python 3.10
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m venv mlproject_env
          source mlproject_env/bin/activate
          pip install -r requirements/prod.txt
          pip install -e .

      - name: Run Full Lifecycle
        run: |
          source mlproject_env/bin/activate
          ./verify_full_lifecycle.sh

      - name: Run Config Generation
        run: |
          source mlproject_env/bin/activate
          ./verify_config_generation.sh

      - name: Upload Logs
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: verification-logs
          path: tmplogs/
```

### GitLab CI

Add to `.gitlab-ci.yml`:

```yaml
stages:
  - test

verify:all:
  stage: test
  image: python:3.10
  script:
    - python -m venv mlproject_env
    - source mlproject_env/bin/activate
    - pip install -r requirements/prod.txt
    - pip install -e .
    - ./verify_all_pipelines.sh
  artifacts:
    when: on_failure
    paths:
      - tmplogs/
```

---

## Log Analysis

### Viewing Logs

```bash
# List all logs
ls -lh tmplogs/

# View specific log
cat tmplogs/standard_train_train.log

# Follow live log
tail -f tmplogs/standard_train_fastapi_serve.log

# Search for errors
grep -i error tmplogs/*.log
grep -i fail tmplogs/*.log
```

### Log Patterns

**Successful Training:**
```
[RUN] Training COMPLETE
[RUN] Model artifacts saved to: ...
```

**Successful Generation:**
```
[ConfigGenerator] Successfully generated: ...
```

**Successful Serving:**
```
{"status":"healthy","model_loaded":true}
```

---

## Performance Benchmarks

Typical execution times on standard hardware:

| Script | Duration | Pipelines Tested |
|--------|----------|------------------|
| verify_all_pipelines.sh | 30-45 min | All (~10 pipelines x 2 frameworks) |
| verify_legacy_run.sh | 5-10 min | 3 commands (train, eval, serve) |
| verify_full_lifecycle.sh | 15-20 min | 6 phases (standard + nested) |
| verify_feast_integration.sh | 45-60 min | 9 combos x 2 frameworks |
| verify_config_generation.sh | 25-35 min | All pipelines x 4 configs |
| verify_all_deployments.sh | 20-30 min | 4 K8s deployment scenarios |

**Total comprehensive verification:** ~2.5 hours

---

## Quick Reference

```bash
# Run all verifications (full test suite)
./verify_all_pipelines.sh           # Local pipeline verification
./verify_legacy_run.sh
./verify_full_lifecycle.sh
./verify_feast_integration.sh
./verify_config_generation.sh
./verify_all_deployments.sh         # K8s deployment verification

# Run specific tests
./verify_full_lifecycle.sh          # E2E only
./verify_feast_integration.sh       # Feast only
./verify_all_deployments.sh         # K8s only

# Check logs
ls tmplogs/                         # List all logs
tail -f tmplogs/*.log               # Follow all logs

# Cleanup
rm -rf tmplogs/                     # Remove logs
fuser -k 8082/tcp                   # Kill port 8082
ray stop                            # Stop Ray
```

---

## Best Practices

1. **Run Before Commits**
   - Always run `verify_full_lifecycle.sh` before major commits
   - Run relevant verification script when modifying specific components

2. **Log Management**
   - Review logs after failures: `cat tmplogs/{pipeline}_train.log`
   - Keep logs for debugging: `cp -r tmplogs/ tmplogs_backup/`

3. **Resource Cleanup**
   - Scripts include cleanup traps, but verify manually if needed
   - Check port 8082 and Ray status before running

4. **Incremental Testing**
   - Test individual pipelines during development
   - Run full suite before releases

5. **CI Integration**
   - Include verification in pre-merge pipelines
   - Upload logs as artifacts for failed runs

---

## Related Documentation

- [API Generation Guide](./api_generation_guide.md) - How to generate and run APIs
- [Generating Configs](./pipeline_orchestration.md) - DAG pipeline system
- [Feast Integration](./feast_integration.md) - Feature store usage
- [Architecture](./architecture.md) - System architecture overview
