# Separated Docker Architecture

## Overview

The project now uses **separated Docker images** for different use cases:

- **Training Image** (`Dockerfile.train`): Full ML environment for in-house use
- **Serving Image** (`Dockerfile.serve`): Minimal API for customer deployment

This separation provides:
- **Smaller serving images** for customers (~600MB vs ~2GB)
- **Better security** for customer deployments (no training code/data)
- **Clear boundaries** between in-house and customer environments
- **Shared MLflow** connection for model versioning

---

## Architecture Diagram

```text
+-------------------------------------------------------------+
|                    Shared Infrastructure                     |
+-------------------------------------------------------------+
|  MLflow Server (Port 5000)                                  |
|  - Stores trained models                                     |
|  - Manages model versions and aliases                        |
|                                                              |
|  Feast Registry (Port 6566) [Optional]                      |
|  - Online feature store                                      |
|  - For real-time feature retrieval                          |
+-------------------------------------------------------------+
         v Shared Connection v              v Shared Connection v
+------------------------------+  +------------------------------+
|  TRAINING (In-house)         |  |  SERVING (Customer)          |
+------------------------------+  +------------------------------+
| Dockerfile.train             |  | Dockerfile.serve             |
|                              |  |                              |
| Components:                  |  | Components:                  |
| - Full ML training           |  | - FastAPI + Uvicorn          |
| - Data preprocessing         |  | - MLflow client (skinny)     |
| - Feature engineering        |  | - Feast client (optional)    |
| - Model evaluation           |  | - Minimal dependencies       |
| - Hyperparameter tuning      |  |                              |
| - Feast ingestion            |  | Size: ~600-800MB             |
| - All ML libraries           |  |                              |
| - Development tools          |  | Security:                    |
|                              |  | - No training code           |
| Size: ~1.5-2GB               |  | - No source data             |
|                              |  | - Read-only model access     |
| Usage:                       |  |                              |
| - Train models               |  | Usage:                       |
| - Evaluate models            |  | - Serve predictions          |
| - Tune hyperparameters       |  | - Load models from MLflow    |
| - Ingest features to Feast   |  | - (Optional) Fetch features  |
+------------------------------+  +------------------------------+
```

---

## Image Comparison

| Feature | Training (In-house) | Serving (Customer) |
|---------|--------------------|--------------------|
| **Dockerfile** | `Dockerfile.train` | `Dockerfile.serve` |
| **Base Image** | python:3.11-slim | python:3.11-slim |
| **Size** | ~1.5-2GB | ~600-800MB |
| **ML Libraries** | All (XGBoost, TensorFlow, etc.) | None (uses serialized models) |
| **Training Code** | Full pipeline code | Not included |
| **Serving Code** | Included | Only serving module |
| **Data** | Mounted for training | Not needed |
| **MLflow** | Client + Server connection | Client only (model loading) |
| **Feast** | Full SDK + ingestion | Online client only |
| **Dev Tools** | git, vim, pytest | Not included |
| **Security** | Internal use | Customer-safe |
| **Use Case** | Train, eval, tune, experiment | Production inference only |

---

## Quick Start

### 1. Build Images

```bash
# Build serving image (for customers)
./docker-build.sh -i serve

# Build training image (in-house)
./docker-build.sh -i train

# Build both
./docker-build.sh -i serve && ./docker-build.sh -i train
```

### 2. Run Services

**In-house Training:**

```bash
# Start MLflow first
docker-compose up -d mlflow

# Run training
docker-compose --profile train run --rm train

# Run evaluation
docker-compose --profile eval run --rm evaluate

# Run tuning
docker-compose --profile tune run --rm tune
```

**Customer Serving:**

```bash
# Start serving API (connects to MLflow for models)
docker-compose up -d api mlflow

# Test API
curl http://localhost:8000/health
```

**With Feast Online Features:**

```bash
# Start with Feast registry
docker-compose --profile feast up -d api-feast mlflow feast-registry

# API available at http://localhost:8001
```

---

## Detailed Usage

### Training Workflow (In-house)

**1. Build training image:**

```bash
./docker-build.sh -i train
```

**2. Start infrastructure:**

```bash
docker-compose up -d mlflow
```

**3. Run training:**

```bash
docker-compose --profile train run --rm train

# Or with custom config
docker-compose run --rm train \
  python -m mlproject.src.pipeline.dag_run train \
  -e /app/mlproject/configs/experiments/custom.yaml \
  -p /app/mlproject/configs/pipelines/custom_train.yaml
```

**4. Evaluate model:**

```bash
docker-compose --profile eval run --rm evaluate
```

**5. Tune hyperparameters:**

```bash
docker-compose --profile tune run --rm tune
```

**6. Development mode (interactive):**

```bash
docker-compose --profile dev run --rm train-dev
# Opens bash shell with full environment
```

### Serving Workflow (Customer)

**1. Build serving image:**

```bash
./docker-build.sh -i serve
```

**2. Start serving (standalone):**

```bash
docker-compose up -d api mlflow

# API: http://localhost:8000
# MLflow: http://localhost:5000
```

**3. Test API:**

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "feature1": [1.0, 2.0, 3.0],
      "feature2": [4.0, 5.0, 6.0]
    }
  }'
```

**4. With Feast online features:**

```bash
# Start with Feast enabled
docker-compose --profile feast up -d api-feast mlflow feast-registry

# API with Feast: http://localhost:8001
```

**5. Deploy to customer:**

```bash
# Tag for customer registry
docker tag ml-pipeline-serve:latest customer-registry.com/ml-api:v1.0.0

# Push to customer
docker push customer-registry.com/ml-api:v1.0.0
```

---

## Environment Variables

### Shared (Both Images)

| Variable | Default | Description |
|----------|---------|-------------|
| MLFLOW_TRACKING_URI | http://mlflow:5000 | MLflow server URL |
| PYTHONUNBUFFERED | 1 | Unbuffered Python output |

### Training Only

| Variable | Default | Description |
|----------|---------|-------------|
| FEAST_ONLINE_ENABLED | false | Enable Feast online store |

### Serving Only

| Variable | Default | Description |
|----------|---------|-------------|
| FEAST_ONLINE_ENABLED | false | Enable Feast for serving |
| FEAST_REGISTRY_URI | - | Feast registry server URL |

---

## Volume Mounts

### Training Image

```yaml
volumes:
  - ./mlruns:/app/mlruns              # MLflow artifacts
  - ./artifacts:/app/artifacts        # Training artifacts
  - ./mlproject/data:/app/mlproject/data  # Training data
  - ./feast_store:/app/feast_store    # Feast offline store
  - ./logs:/app/logs                  # Application logs
```

### Serving Image

```yaml
volumes:
  - ./logs:/app/logs:rw               # Application logs only
  # Optional for Feast:
  - ./mlproject/configs/feast:/app/.feast:ro
```

**Note**: Serving image does NOT mount source code or training data for security.

---

## MLflow Integration

### Training Side (In-house)

1. Train model and log to MLflow
2. Set model alias (e.g., `production`, `staging`)
3. MLflow stores model artifacts

```bash
# Training automatically logs to MLflow
docker-compose --profile train run --rm train

# Check MLflow UI
open http://localhost:5000
```

### Serving Side (Customer)

1. API connects to MLflow server
2. Loads model by alias (`production`, `staging`, `latest`)
3. Serves predictions using loaded model

```python
# In serving API code
import mlflow

# Load model from MLflow
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/xgboost_model@{alias}"
)
```

---

## Feast Integration

### Training Side (In-house)

**Ingest features to Feast offline store:**

```bash
docker-compose run --rm train \
  python -m mlproject.src.features.examples.ingest_forecast
```

**Train with Feast features:**

```bash
docker-compose --profile train run --rm train \
  python -m mlproject.src.pipeline.dag_run train \
  -e /app/mlproject/configs/experiments/etth3_feast.yaml \
  -p /app/mlproject/configs/pipelines/standard_train.yaml
```

### Serving Side (Customer)

**Option 1: Without Feast (serving from CSV/DataFrame)**

```bash
# Standard serving
docker-compose up -d api mlflow
```

**Option 2: With Feast online store**

```bash
# Start Feast registry
docker-compose --profile feast up -d api-feast mlflow feast-registry

# API fetches features from Feast online store
curl -X POST http://localhost:8001/predict/feast \
  -H "Content-Type: application/json" \
  -d '{
    "time_point": "2024-01-09T00:00:00",
    "entities": [1, 2, 3]
  }'
```

---

## Security Considerations

### Training Image (In-house)

- Full access to source code
- Access to training data
- Development tools included
- Run in secure internal network

### Serving Image (Customer)

**Security Features:**
- No training source code
- No training data
- No development tools
- Read-only model access via MLflow
- Runs as non-root user (UID 1000)
- Minimal attack surface (~600MB image)

**Not Included:**
- Training pipeline code
- Data preprocessing source
- Hyperparameter tuning logic
- Internal utilities and scripts

**Deployment Checklist:**
- [ ] Use specific version tags (not `latest`)
- [ ] Scan image for vulnerabilities (`docker scan`)
- [ ] Configure resource limits (CPU, memory)
- [ ] Use secrets for MLflow credentials
- [ ] Enable TLS for API endpoints
- [ ] Monitor with health checks
- [ ] Set up logging and alerting

---

## Building for Different Platforms

### For x86 servers (from M1/M2 Mac)

```bash
# Training image
./docker-build.sh -i train -p linux/amd64

# Serving image
./docker-build.sh -i serve -p linux/amd64
```

### Multi-architecture build

```bash
# Create buildx builder
docker buildx create --name multiarch --use

# Build for both architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.serve \
  -t ml-pipeline-serve:latest \
  --push \
  .
```

---

## Troubleshooting

### Training Issues

**Problem: Training container exits immediately**

```bash
# Check logs
docker-compose logs train

# Run interactively
docker-compose --profile dev run --rm train-dev
```

**Problem: MLflow connection error**

```bash
# Verify MLflow is running
docker-compose ps mlflow
curl http://localhost:5000/health

# Check network
docker network inspect ml_pipeline_template_ml-network
```

### Serving Issues

**Problem: Model not found**

```bash
# Verify model exists in MLflow
curl http://localhost:5000/api/2.0/mlflow/registered-models/list

# Check alias is set
curl http://localhost:5000/api/2.0/mlflow/registered-models/get-model-version-by-alias?name=xgboost_model&alias=production
```

**Problem: Serving image too large**

```bash
# Check actual size
docker images ml-pipeline-serve:latest

# If larger than expected, verify Dockerfile.serve is being used
docker history ml-pipeline-serve:latest
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build-train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build training image
        run: ./docker-build.sh -i train -t ${{ github.sha }}

  build-serve:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build serving image
        run: ./docker-build.sh -i serve -t ${{ github.sha }}
      - name: Push to customer registry
        run: |
          docker tag ml-pipeline-serve:${{ github.sha }} \
            customer-registry.com/ml-api:${{ github.sha }}
          docker push customer-registry.com/ml-api:${{ github.sha }}
```

---

## Makefile Commands

```bash
# Build images
make docker-build-train       # Build training image
make docker-build-serve       # Build serving image

# Run services
make docker-train             # Run training job
make docker-eval              # Run evaluation job
make docker-serve             # Start serving API

# Development
make docker-train-dev         # Interactive training environment
make docker-shell-train       # Shell in training container
make docker-shell-serve       # Shell in serving container

# Utilities
make docker-clean             # Clean up all containers and images
make docker-size              # Show image sizes
```

---

## Best Practices

### For In-house Training

1. **Version everything**: Tag images with git SHA or semantic version
2. **Use volumes**: Mount data and artifacts, don't copy into image
3. **Log to MLflow**: Always log models with clear aliases
4. **Test locally**: Run training locally before production
5. **Monitor resources**: Set CPU/memory limits

### For Customer Serving

1. **Minimal image**: Only include necessary dependencies
2. **Security first**: No source code, no training data
3. **Versioned models**: Use specific model aliases, not `latest`
4. **Health checks**: Always include health check endpoints
5. **Resource limits**: Set appropriate CPU/memory constraints
6. **Secrets management**: Use environment variables for credentials
7. **TLS/SSL**: Enable HTTPS for production APIs
8. **Monitoring**: Set up logging and alerting
9. **Documentation**: Provide clear API documentation for customers
10. **Support**: Include contact information for issues

---

## Related Documentation

- [README.Docker.md](README.Docker.md) - Original Docker setup
- [Deployment Guide](docs/deployment_guide.md) - Kubernetes deployment
- [API Generation Guide](docs/api_generation_guide.md) - API generation
- [Verification Guide](docs/verification_guide.md) - Testing

---

---

## Deployment to Kubernetes

While the above commands use `docker-compose` for local development, professional deployment to Kubernetes is handled via the automated `deploy.sh` script.

```bash
# Deploy to K8s with dynamic config
./deploy.sh feast  # or standard
```

For a complete step-by-step walkthrough of the Kubernetes lifecycle, including verification and cleanup, see the **[Zero-to-Hero Deployment Guide](docs/deployment_guide.md)**.

## Cleanup

```bash
# Cleanup K8s resources
./cleanup.sh
```

---

## Support

For issues:
- In-house training: Check logs in `./logs/` directory
- Customer serving: Check MLflow model registry and API logs
- Network issues: Verify docker network `ml-network` exists
- Model loading: Verify MLflow tracking URI is accessible
