# Microsoft Fabric Deployment Guide — ml_pipeline_fw

This guide explains how to run **ml_pipeline_fw** on [Microsoft Fabric](https://learn.microsoft.com/en-us/fabric/get-started/microsoft-fabric-overview). It covers workspace setup, Lakehouse configuration, training pipelines, model serving, and CI/CD integration.

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Workspace and Lakehouse Setup](#3-workspace-and-lakehouse-setup)
4. [Installing the Package on Fabric](#4-installing-the-package-on-fabric)
5. [Configuring MLflow Tracking](#5-configuring-mlflow-tracking)
6. [Running the Training Pipeline](#6-running-the-training-pipeline)
7. [Running Evaluation](#7-running-evaluation)
8. [Model Serving Options](#8-model-serving-options)
9. [Hyperparameter Tuning](#9-hyperparameter-tuning)
10. [Using Feast with Fabric](#10-using-feast-with-fabric)
11. [CI/CD Integration](#11-cicd-integration)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Architecture Overview

The diagram below maps each local component to its Fabric equivalent.

| Local Component | Microsoft Fabric Equivalent | Notes |
|---|---|---|
| `mlruns/` + `mlflow.db` | **ML Experiments** (built-in MLflow) | Auto-configured in notebooks |
| `artifacts/models/` | **Lakehouse Files** → `Files/models/` | OneLake path |
| `artifacts/preprocessors/` | **Lakehouse Files** → `Files/preprocessors/` | OneLake path |
| CSV data files | **Lakehouse Delta Tables** | Use Spark to load |
| `feast_store/` (offline) | **Lakehouse Delta Tables** | Parquet-compatible |
| FastAPI / Ray Serve | **Fabric Inference Endpoint** *or* Azure Container Apps | See §8 |
| `python -m mlproject.src.pipeline.dag_run train` | **Fabric Notebook** (Spark compute) | No Docker needed |
| Docker training images | **Fabric Environment** (custom pool) | Optional |
| `sqlite:///mlflow.db` (Optuna) | **In-memory Optuna storage** | `storage: null` in config |

### Key design principle

The core pipeline code (`dag_run.py`, `FlexiblePipeline`, `MLflowManager`, model wrappers) is **not modified**. Compatibility is achieved through:

1. A thin `FabricEnv` helper (`mlproject/src/compat/fabric.py`) that auto-detects the Fabric runtime.
2. A YAML config overlay (`mlproject/configs/environments/fabric.yaml`) that overrides paths and tracking URI.

---

## 2. Prerequisites

### Azure / Fabric side

| Requirement | Details |
|---|---|
| Microsoft Fabric capacity | F2 or higher recommended for ML workloads |
| Fabric workspace | Any license SKU; F-SKU preferred for native endpoints |
| Azure App Registration | Needed only for CI/CD automation |
| Storage access | OneLake is automatically available inside notebooks |

### Local side (for initial upload)

```bash
# Build the wheel package locally
pip install build
python -m build --wheel
# Produces: dist/mlproject-<version>-py3-none-any.whl
```

---

## 3. Workspace and Lakehouse Setup

### 3.1 Create a Workspace

1. Sign in to [app.fabric.microsoft.com](https://app.fabric.microsoft.com).
2. Click **Workspaces** → **New workspace**.
3. Name it (e.g., `ml-pipeline-prod`) and assign a Fabric capacity.

### 3.2 Create a Lakehouse

1. Inside the workspace, click **New item** → **Lakehouse**.
2. Name it `ml_artifacts`.
3. The OneLake mount path inside every attached notebook is:
   ```
   /lakehouse/default/Files/
   ```

### 3.3 Create the Required Folder Structure

Inside the Lakehouse, create the following folders under **Files**:

```
Files/
├── models/           ← trained model artifacts (joblib / pyfunc)
├── preprocessors/    ← fitted TransformManager artifacts
├── configs/
│   ├── experiments/  ← copy of mlproject/configs/experiments/
│   ├── pipelines/    ← copy of mlproject/configs/pipelines/
│   └── environments/ ← mlproject/configs/environments/fabric.yaml
├── data/             ← input CSV / Parquet files
└── requirements/
    └── fabric.txt    ← requirements/fabric.txt
```

You can create folders via the Lakehouse UI or the following PySpark snippet in a notebook:

```python
for folder in ["models", "preprocessors", "configs/experiments",
               "configs/pipelines", "configs/environments", "data"]:
    dbutils.fs.mkdirs(f"Files/{folder}")
```

### 3.4 Upload Project Files

#### Option A — Fabric UI (easiest)
Drag-and-drop files into the Lakehouse **Files** explorer.

#### Option B — OneLake REST API (for automation)

```bash
# Authenticate with Azure CLI
az login

TOKEN=$(az account get-access-token \
  --resource https://storage.azure.com \
  --query accessToken -o tsv)

WORKSPACE_ID="<your-workspace-id>"
LAKEHOUSE_ID="<your-lakehouse-id>"
ONELAKE_URL="https://onelake.dfs.fabric.microsoft.com/${WORKSPACE_ID}/${LAKEHOUSE_ID}.Lakehouse"

# Upload the wheel
WHL=$(ls dist/*.whl | head -1)
curl -X PUT "${ONELAKE_URL}/Files/$(basename $WHL)" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @$WHL

# Upload configs (repeat for each file)
curl -X PUT "${ONELAKE_URL}/Files/configs/environments/fabric.yaml" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @mlproject/configs/environments/fabric.yaml
```

---

## 4. Installing the Package on Fabric

### 4.1 Install from Wheel (Recommended)

At the top of every training notebook, run:

```python
# Cell 1 — Install the project wheel and extra dependencies
%pip install -q /lakehouse/default/Files/mlproject-*.whl
%pip install -q -r /lakehouse/default/Files/requirements/fabric.txt
```

> **Note:** Fabric Spark environments already include `pandas`, `numpy`, `scikit-learn`, `torch`,
> `pyarrow`, and `mlflow`. The `requirements/fabric.txt` file only lists packages
> that are **not** pre-installed (e.g., `xgboost`, `catboost`, `omegaconf`, `optuna`).

### 4.2 Install from Source (Development)

If you prefer working directly from source code uploaded to the Lakehouse:

```python
import sys
sys.path.insert(0, "/lakehouse/default/Files/")

# Alternatively, install in editable mode
%pip install -e /lakehouse/default/Files/
```

### 4.3 Creating a Custom Fabric Environment (Optional)

For reproducibility across notebooks, create a reusable Fabric Environment:

1. Go to workspace → **New item** → **Environment**.
2. Under **Public libraries**, add the packages from `requirements/fabric.txt`.
3. Under **Custom libraries**, upload `mlproject-*.whl`.
4. Publish the environment and attach it to your notebooks.

---

## 5. Configuring MLflow Tracking

Microsoft Fabric provides a built-in MLflow tracking server. No external server is required.

### 5.1 How It Works

When running inside a Fabric notebook, the MLflow tracking URI is automatically set to `mlflow` which routes to the workspace's built-in tracking server. All experiments and runs appear in **Workspace → ML Experiments**.

### 5.2 Applying the Fabric Config Overlay

The project includes a ready-made config overlay at `mlproject/configs/environments/fabric.yaml`:

```yaml
# mlproject/configs/environments/fabric.yaml
mlflow:
  tracking_uri: "mlflow"                  # Fabric built-in tracking
  experiment_name: "ml_pipeline_fabric"
  enabled: true
  artifacts:
    log_model: true
    log_config: true

training:
  artifacts_dir: /lakehouse/default/Files/models

preprocessing:
  artifacts_dir: /lakehouse/default/Files/preprocessors

tuning:
  storage: null                           # In-memory (no SQLite on Fabric)
```

### 5.3 Using the FabricEnv Helper

The `FabricEnv` class auto-detects whether you are running on Fabric and applies the correct overrides:

```python
from mlproject.src.compat.fabric import FabricEnv

fabric = FabricEnv.detect()
fabric.apply_env_vars()  # Sets MLFLOW_TRACKING_URI=mlflow on Fabric

print(f"Running on Fabric : {fabric.enabled}")
print(f"Artifacts dir     : {fabric.artifacts_dir('models')}")
# On Fabric  → /lakehouse/default/Files/models
# Locally    → artifacts/models
```

---

## 6. Running the Training Pipeline

### 6.1 Minimal Notebook

Copy the following cells into a Fabric notebook and attach the `ml_artifacts` Lakehouse.

```python
# ── Cell 1: Install dependencies ──────────────────────────────────────────
%pip install -q /lakehouse/default/Files/mlproject-*.whl
%pip install -q -r /lakehouse/default/Files/requirements/fabric.txt
```

```python
# ── Cell 2: Detect Fabric runtime ─────────────────────────────────────────
from mlproject.src.compat.fabric import FabricEnv
from omegaconf import OmegaConf

fabric = FabricEnv.detect()
fabric.apply_env_vars()

# Load the Fabric environment overlay
fabric_overlay = OmegaConf.load(
    "/lakehouse/default/Files/configs/environments/fabric.yaml"
)
```

```python
# ── Cell 3: Merge configs and run training ─────────────────────────────────
import uuid, tempfile
from pathlib import Path
from omegaconf import OmegaConf
from mlproject.src.pipeline.steps.core.utils import ConfigMerger
from mlproject.src.pipeline.pipeline import FlexiblePipeline

EXPERIMENT_PATH = "/lakehouse/default/Files/configs/experiments/tabular.yaml"
PIPELINE_PATH   = "/lakehouse/default/Files/configs/pipelines/standard_train.yaml"

# Deep-merge experiment config with Fabric overlay
merged = ConfigMerger.merge(EXPERIMENT_PATH, PIPELINE_PATH, mode="train")
merged = OmegaConf.merge(merged, fabric_overlay)

# Write to a temporary file (required by FlexiblePipeline)
tmp = Path(tempfile.gettempdir()) / f"fabric_{uuid.uuid4().hex}.yaml"
OmegaConf.save(merged, tmp)

try:
    pipeline = FlexiblePipeline(str(tmp))
    context  = pipeline.run_exp()
    print("Training complete. Context keys:", list(context.keys()))
finally:
    tmp.unlink(missing_ok=True)
```

### 6.2 Using the High-Level `run_training` Helper

If you prefer the simplified CLI-compatible helper (no manual merging):

```python
# Set Fabric-specific env vars first
import os
os.environ["MLFLOW_TRACKING_URI"] = "mlflow"
os.environ["ARTIFACTS_DIR"] = "/lakehouse/default/Files/models"

# Then call run_training directly
from mlproject.src.pipeline.dag_run import run_training

run_training(
    experiment_path="/lakehouse/default/Files/configs/experiments/tabular.yaml",
    pipeline_path="/lakehouse/default/Files/configs/pipelines/standard_train.yaml",
)
```

> **When to use each approach:**
> - **Cell 3 approach** — full control, merges the Fabric overlay cleanly.
> - **run_training approach** — simpler, relies on env vars to override paths.

---

## 7. Running Evaluation

After training, run evaluation to compute metrics on the hold-out set:

```python
# ── Evaluation cell ────────────────────────────────────────────────────────
from mlproject.src.pipeline.dag_run import run_eval

run_eval(
    experiment_path="/lakehouse/default/Files/configs/experiments/tabular.yaml",
    pipeline_path="/lakehouse/default/Files/configs/pipelines/standard_train.yaml",
    alias="latest",   # or a specific version string, e.g. "champion"
)
```

Evaluation results (metrics, confusion matrix, etc.) are automatically logged to the MLflow run. View them in **Workspace → ML Experiments → your experiment name**.

---

## 8. Model Serving Options

### Option A — Fabric Real-time Inference Endpoint (Recommended)

Fabric natively supports deploying registered MLflow models as managed endpoints.

**Via UI:**
1. Go to **ML Experiments** → select your experiment.
2. Find the run and click the registered model.
3. In **ML Models**, click **Deploy** → **Real-time endpoint**.
4. Fabric handles environment, scaling, and TLS automatically.

**Via Python SDK:**
```python
import mlflow

# Load the registered model for batch or online inference
model_uri = "models:/fabric_model/latest"
model     = mlflow.pyfunc.load_model(model_uri)

import pandas as pd
df    = pd.read_csv("/lakehouse/default/Files/data/input.csv")
preds = model.predict(df)
print(preds[:5])
```

> **Important:** Real-time endpoint support is available on Fabric F-SKU (Power BI Premium)
> capacities. If your workspace is on a trial or Pro license, use batch inference or
> Option B below.

### Option B — Azure Container Apps (Keep FastAPI / Ray Serve)

If you want to preserve the existing FastAPI or Ray Serve stack, deploy the pre-built Docker image to Azure Container Apps (ACA) and point its `MLFLOW_TRACKING_URI` at the Fabric workspace.

```bash
# 1. Build and push the serving image
docker build -f Dockerfile.serve -t ml-pipeline-serve:latest .

ACR="<your-acr-name>"
az acr login --name $ACR
docker tag ml-pipeline-serve:latest ${ACR}.azurecr.io/ml-pipeline-serve:latest
docker push ${ACR}.azurecr.io/ml-pipeline-serve:latest

# 2. Create the Container App
az containerapp create \
  --name ml-pipeline-serve \
  --resource-group <resource-group> \
  --environment <container-app-env> \
  --image ${ACR}.azurecr.io/ml-pipeline-serve:latest \
  --target-port 8000 \
  --ingress external \
  --env-vars \
    MLFLOW_TRACKING_URI=<fabric-mlflow-uri> \
    MODEL_ALIAS=latest
```

### Option C — Batch Scoring with PySpark (Fabric)

For large-scale, non-real-time inference, run batch scoring directly on Fabric Spark:

```python
from pyspark.sql import SparkSession
import mlflow.spark

spark = SparkSession.builder.getOrCreate()

# Load input data from Delta Table
df_spark = spark.read.format("delta").load(
    "abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>.Lakehouse/Tables/inference_input"
)

# Load and apply MLflow model (SynapseML PREDICT or pyfunc UDF)
model_uri = "models:/fabric_model/latest"

predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="double")
df_result   = df_spark.withColumn("prediction", predict_udf(*df_spark.columns))

# Write predictions back to Lakehouse Delta Table
df_result.write.format("delta").mode("overwrite").saveAsTable("predictions_output")
```

---

## 9. Hyperparameter Tuning

Optuna tuning works on Fabric with one config change: replace the local SQLite storage with in-memory storage (already set in `configs/environments/fabric.yaml`).

```python
# The fabric.yaml overlay already sets: tuning.storage: null
# This tells Optuna to use in-memory storage (no SQLite file)

from mlproject.src.pipeline.dag_run import run_tune

run_tune(
    experiment_path="/lakehouse/default/Files/configs/experiments/tabular.yaml",
    pipeline_path="/lakehouse/default/Files/configs/pipelines/standard_train.yaml",
    n_trials=50,
)
```

All trial metrics are logged to MLflow automatically by `optuna-integration[mlflow]`.

---

## 10. Using Feast with Fabric

> **Current limitation:** Feast does not natively integrate with Microsoft Fabric.
> The offline store and online store must be configured as external Azure services.

### Recommended architecture for Feast on Fabric

| Feast Component | Azure Service |
|---|---|
| Offline store | Azure Data Lake Storage Gen2 (same OneLake endpoint) |
| Online store | Azure Redis Cache |
| Feature registry | Azure Blob Storage |

### Configuration

```yaml
# feast_store/feature_store.yaml (external, not inside Fabric)
project: ml_pipeline
provider: azure
registry:
  path: az://feast-registry/registry.db
offline_store:
  type: file
  path: /lakehouse/default/Files/feast/
online_store:
  type: redis
  connection_string: "<your-redis>.redis.cache.windows.net:6380,password=...,ssl=True"
```

Feast ingestion can still be triggered from a Fabric Notebook:

```python
# Run Feast materialization from inside a Fabric notebook
%pip install -q feast[azure]

from feast import FeatureStore
store = FeatureStore(repo_path="/lakehouse/default/Files/feast_store/")
store.materialize_incremental(end_date=datetime.utcnow())
```

---

## 11. CI/CD Integration

Extend `.github/workflows/ci.yml` with a deploy job that uploads the wheel and config files to the Fabric Lakehouse after tests pass.

### 11.1 Required GitHub Secrets

| Secret | Description | Where to find |
|---|---|---|
| `FABRIC_CLIENT_ID` | Azure App Registration Client ID | Azure Portal → App Registrations |
| `FABRIC_CLIENT_SECRET` | App Registration Secret | Azure Portal → Certificates & Secrets |
| `FABRIC_TENANT_ID` | Azure AD Tenant ID | Azure Portal → Overview |
| `FABRIC_WORKSPACE_ID` | Fabric Workspace ID | Fabric URL (after `/groups/`) |
| `FABRIC_LAKEHOUSE_ID` | Lakehouse item ID | Lakehouse URL |

### 11.2 Deploy Job

Add the following job to `.github/workflows/ci.yml`:

```yaml
  deploy-to-fabric:
    runs-on: ubuntu-latest
    needs: test-and-lint          # only deploy when tests pass
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Build wheel package
      run: |
        pip install build
        python -m build --wheel

    - name: Authenticate with Azure
      run: |
        TOKEN=$(curl -s -X POST \
          "https://login.microsoftonline.com/${{ secrets.FABRIC_TENANT_ID }}/oauth2/v2.0/token" \
          -d "client_id=${{ secrets.FABRIC_CLIENT_ID }}" \
          -d "client_secret=${{ secrets.FABRIC_CLIENT_SECRET }}" \
          -d "scope=https://storage.azure.com/.default" \
          -d "grant_type=client_credentials" | jq -r '.access_token')
        echo "ONELAKE_TOKEN=$TOKEN" >> $GITHUB_ENV

    - name: Upload wheel to Lakehouse
      run: |
        WHL=$(ls dist/*.whl | head -1)
        BASE="https://onelake.dfs.fabric.microsoft.com/${{ secrets.FABRIC_WORKSPACE_ID }}/${{ secrets.FABRIC_LAKEHOUSE_ID }}.Lakehouse/Files"

        curl -s -X PUT "${BASE}/$(basename $WHL)" \
          -H "Authorization: Bearer $ONELAKE_TOKEN" \
          --data-binary @$WHL
        echo "Uploaded: $(basename $WHL)"

    - name: Upload configs to Lakehouse
      run: |
        BASE="https://onelake.dfs.fabric.microsoft.com/${{ secrets.FABRIC_WORKSPACE_ID }}/${{ secrets.FABRIC_LAKEHOUSE_ID }}.Lakehouse/Files"
        CONFIGS=(
          "mlproject/configs/environments/fabric.yaml:configs/environments/fabric.yaml"
          "requirements/fabric.txt:requirements/fabric.txt"
        )
        for pair in "${CONFIGS[@]}"; do
          src="${pair%%:*}"
          dest="${pair##*:}"
          curl -s -X PUT "${BASE}/${dest}" \
            -H "Authorization: Bearer $ONELAKE_TOKEN" \
            --data-binary @$src
          echo "Uploaded: $dest"
        done
```

### 11.3 Trigger a Training Run via Fabric REST API (Optional)

After uploading files, you can programmatically trigger a notebook run:

```yaml
    - name: Trigger Fabric Notebook run
      run: |
        # Get Fabric API token (different scope from OneLake)
        FABRIC_TOKEN=$(curl -s -X POST \
          "https://login.microsoftonline.com/${{ secrets.FABRIC_TENANT_ID }}/oauth2/v2.0/token" \
          -d "client_id=${{ secrets.FABRIC_CLIENT_ID }}" \
          -d "client_secret=${{ secrets.FABRIC_CLIENT_SECRET }}" \
          -d "scope=https://api.fabric.microsoft.com/.default" \
          -d "grant_type=client_credentials" | jq -r '.access_token')

        # Trigger notebook job (replace NOTEBOOK_ID with your notebook's item ID)
        curl -s -X POST \
          "https://api.fabric.microsoft.com/v1/workspaces/${{ secrets.FABRIC_WORKSPACE_ID }}/items/${{ secrets.FABRIC_NOTEBOOK_ID }}/jobs/instances?jobType=RunNotebook" \
          -H "Authorization: Bearer $FABRIC_TOKEN" \
          -H "Content-Type: application/json" \
          -d '{}'
        echo "Notebook job triggered."
```

---

## 12. Troubleshooting

### MLflow runs not appearing in Fabric ML Experiments

**Symptoms:** Training completes but no runs appear in the workspace.

**Check:**
```python
import mlflow
print(mlflow.get_tracking_uri())   # Should print "mlflow" or a Fabric URL
```

**Fix:** Ensure `FabricEnv.apply_env_vars()` is called before importing any pipeline module, or explicitly set:
```python
import os
os.environ["MLFLOW_TRACKING_URI"] = "mlflow"
import mlflow
mlflow.set_experiment("ml_pipeline_fabric")
```

---

### `ModuleNotFoundError` for mlproject

**Cause:** The wheel was not installed or not found.

**Fix:**
```python
# Verify the wheel is in the Lakehouse
import os
files = os.listdir("/lakehouse/default/Files/")
print([f for f in files if f.endswith(".whl")])

# Reinstall
%pip install -q /lakehouse/default/Files/mlproject-*.whl --force-reinstall
```

---

### Artifacts not saved to Lakehouse

**Symptoms:** Training succeeds but no files appear under `Files/models/`.

**Check the config:**
```python
from omegaconf import OmegaConf
from mlproject.src.pipeline.steps.core.utils import ConfigMerger

merged = ConfigMerger.merge(EXPERIMENT_PATH, PIPELINE_PATH, mode="train")
print(OmegaConf.to_yaml(merged.get("training", {})))
# Expected: artifacts_dir should be /lakehouse/default/Files/models
```

**Fix:** Make sure the Fabric overlay is merged **after** the experiment config:
```python
merged = OmegaConf.merge(experiment_cfg, fabric_overlay)   # ✅ correct order
merged = OmegaConf.merge(fabric_overlay, experiment_cfg)   # ❌ experiment overrides Fabric
```

---

### Optuna raises SQLite errors

**Cause:** Optuna defaults to SQLite storage, which requires a writable filesystem path.

**Fix:** Use the Fabric overlay which sets `tuning.storage: null` (in-memory):
```python
# Verify the setting was applied
print(merged.get("tuning", {}).get("storage"))   # Should print None
```

---

### Ray Serve startup errors inside Fabric notebook

**Cause:** Ray Serve requires multi-process management that conflicts with Fabric's Spark runtime.

**Fix:** Do not use Ray Serve inside Fabric notebooks. Instead:
- Use **Fabric Inference Endpoints** for real-time serving (see §8 Option A).
- Deploy the serving Docker image to **Azure Container Apps** (see §8 Option B).

---

### Package version conflicts

**Symptoms:** `ImportError` or unexpected behavior after install.

**Check installed versions:**
```python
import xgboost, catboost, sklearn, mlflow
print(f"xgboost  : {xgboost.__version__}")
print(f"catboost : {catboost.__version__}")
print(f"sklearn  : {sklearn.__version__}")
print(f"mlflow   : {mlflow.__version__}")
```

**Fix:** Pin versions in `requirements/fabric.txt` if conflicts arise:
```
xgboost==2.0.3
catboost==1.2.5
omegaconf==2.3.0
```

---

## Quick Reference

```bash
# Build and prepare wheel locally
python -m build --wheel

# Upload wheel to Lakehouse (replace placeholders)
TOKEN=$(az account get-access-token --resource https://storage.azure.com --query accessToken -o tsv)
curl -X PUT "https://onelake.dfs.fabric.microsoft.com/<ws>/<lh>.Lakehouse/Files/mlproject.whl" \
  -H "Authorization: Bearer $TOKEN" --data-binary @dist/mlproject-*.whl
```

```python
# Minimal Fabric notebook setup (paste in Cell 1)
%pip install -q /lakehouse/default/Files/mlproject-*.whl
%pip install -q -r /lakehouse/default/Files/requirements/fabric.txt

from mlproject.src.compat.fabric import FabricEnv
fabric = FabricEnv.detect()
fabric.apply_env_vars()
```

```python
# Run training
from mlproject.src.pipeline.dag_run import run_training
run_training(
    experiment_path="/lakehouse/default/Files/configs/experiments/tabular.yaml",
    pipeline_path="/lakehouse/default/Files/configs/pipelines/standard_train.yaml",
)
```

```python
# Batch inference from registered model
import mlflow, pandas as pd
model = mlflow.pyfunc.load_model("models:/fabric_model/latest")
preds = model.predict(pd.read_csv("/lakehouse/default/Files/data/input.csv"))
```

---

## Deployment Checklist

**Initial setup:**
- [ ] Fabric workspace created with a capacity (F2 or higher)
- [ ] Lakehouse `ml_artifacts` created
- [ ] Lakehouse folder structure created (`models/`, `preprocessors/`, `configs/`, `data/`)
- [ ] Wheel built locally (`python -m build --wheel`)
- [ ] Wheel uploaded to `Files/`
- [ ] `requirements/fabric.txt` uploaded to `Files/requirements/`
- [ ] Config files uploaded to `Files/configs/`

**Training:**
- [ ] Fabric notebook created with Lakehouse `ml_artifacts` attached
- [ ] `%pip install` cell runs without errors
- [ ] `FabricEnv.detect()` returns `enabled=True`
- [ ] Pipeline completes without errors
- [ ] MLflow experiment visible in **Workspace → ML Experiments**
- [ ] Artifacts visible under `Files/models/` in the Lakehouse

**Serving:**
- [ ] Model registered in **Workspace → ML Models**
- [ ] Inference endpoint deployed *or* Docker image pushed to Azure Container Apps
- [ ] Health check / predict endpoint responds correctly

**CI/CD:**
- [ ] GitHub secrets `FABRIC_CLIENT_ID`, `FABRIC_CLIENT_SECRET`, `FABRIC_TENANT_ID`, `FABRIC_WORKSPACE_ID`, `FABRIC_LAKEHOUSE_ID` set
- [ ] Deploy job in `ci.yml` uploads wheel and configs on push to `main`
