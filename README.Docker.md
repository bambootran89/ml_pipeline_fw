# Docker Setup Guide (Legacy/All-in-One)

> [!NOTE]
> This guide covers the monolithic multi-stage Docker build. For the optimized **Separated Architecture** (Train vs. Serve), please refer to **[README.Docker.Separated.md](README.Docker.Separated.md)**.

This document provides instructions for building, testing, and running the ML pipeline using Docker.

## Table of Contents

- [Quick Start](#quick-start)
- [Building Images](#building-images)
- [Running Services](#running-services)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Using Make (Recommended)

```bash
# Build and run everything
make docker-all

# Access services:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
```

### Using Docker Compose

```bash
# Start API and MLflow
docker-compose up -d api mlflow

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Building Images

### Multi-Stage Build Architecture

The Dockerfile uses multi-stage builds for optimal image size:

1. **Builder Stage**: Compiles dependencies and creates wheels
2. **Runtime Stage**: Minimal production image (default)
3. **Development Stage**: Includes dev tools and hot reload

### Build Commands

**Production Image (Optimized):**

```bash
# Using build script (recommended)
./docker-build.sh

# Using Make
make docker-build-prod

# Manual build
docker build --target runtime -t ml-pipeline-template:latest .
```

**Development Image:**

```bash
# Using build script
./docker-build.sh -s development -t dev

# Using Make
make docker-build-dev
```

**Platform-Specific Build:**

```bash
# For M1/M2 Macs deploying to x86 servers
./docker-build.sh -p linux/amd64

# Using Make
make docker-build-amd64
```

**Custom Tag:**

```bash
./docker-build.sh -t v1.0.0
```

### Build Script Options

```bash
./docker-build.sh [OPTIONS]

Options:
  -t, --tag TAG         Docker image tag (default: latest)
  -s, --stage STAGE     Build stage: runtime|development (default: runtime)
  -p, --platform ARCH   Target platform (e.g., linux/amd64, linux/arm64)
  -h, --help            Show help message
```

---

## Running Services

### Docker Compose Services

The `docker-compose.yml` defines several services:

| Service | Port | Description | Profile |
|---------|------|-------------|---------|
| mlflow | 5000 | MLflow tracking server | default |
| api | 8000 | Production API | default |
| api-dev | 8001 | Development API (hot reload) | dev |
| train | - | Training job (on-demand) | train |
| eval | - | Evaluation job (on-demand) | eval |

### Start Services

**Production (API + MLflow):**

```bash
docker-compose up -d api mlflow

# Or using Make
make docker-compose-up
```

**Development Mode (with hot reload):**

```bash
docker-compose --profile dev up -d api-dev mlflow

# Or using Make
make docker-compose-dev
```

**Run Training Job:**

```bash
docker-compose --profile train run --rm train

# Or using Make
make docker-compose-train
```

**Run Evaluation:**

```bash
docker-compose --profile eval run --rm eval

# Or using Make
make docker-compose-eval
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Using Make
make docker-compose-logs
```

### Stop Services

```bash
docker-compose down

# Or using Make
make docker-compose-down

# With volume cleanup
docker-compose down -v
```

---

## Development Workflow

### Hot Reload Development

The development image supports hot reloading for rapid iteration:

```bash
# Start dev environment
make docker-compose-dev

# Edit code locally - changes auto-reload in container
# API available at http://localhost:8001
```

### Interactive Shell

**Production Container:**

```bash
# Using Make
make docker-shell

# Manual
docker run -it --rm \
  -v $(pwd)/mlruns:/app/mlruns \
  ml-pipeline-template:latest /bin/bash
```

**Development Container:**

```bash
# Using Make
make docker-shell-dev

# Manual
docker run -it --rm \
  -v $(pwd):/app \
  ml-pipeline-template:dev /bin/bash
```

### Run Commands in Container

**Training:**

```bash
docker run --rm \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/mlproject/data:/app/mlproject/data \
  ml-pipeline-template:latest \
  python -m mlproject.src.pipeline.dag_run train \
    -e /app/mlproject/configs/experiments/etth3.yaml \
    -p /app/mlproject/configs/pipelines/standard_train.yaml
```

**Evaluation:**

```bash
docker run --rm \
  -v $(pwd)/mlruns:/app/mlruns \
  ml-pipeline-template:latest \
  python -m mlproject.src.pipeline.dag_run eval \
    -e /app/mlproject/configs/experiments/etth3.yaml \
    -p /app/mlproject/configs/generated/standard_train_eval.yaml \
    -a latest
```

---

## Testing

### Automated Testing

**Test Docker Image:**

```bash
# Using test script (comprehensive)
./docker-test.sh

# Using Make
make docker-test
```

The test script validates:
1. Image existence
2. Python version
3. Package installation
4. Dependencies
5. Non-root user
6. Container health
7. API endpoints

### Manual Testing

**Health Check:**

```bash
curl http://localhost:8000/health
```

**Prediction Test:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "HUFL": [1.2, 1.5, 1.8],
      "MUFL": [0.8, 0.9, 1.0]
    }
  }'
```

### Run Tests in Docker

**Linting:**

```bash
make docker-lint
```

**Pytest:**

```bash
make docker-pytest
```

---

## Deployment

### Pushing to Container Registry

**Docker Hub:**

```bash
# Tag and push
docker tag ml-pipeline-template:latest yourusername/ml-pipeline-template:latest
docker push yourusername/ml-pipeline-template:latest

# Using Make
make docker-push REGISTRY=yourusername
```

**AWS ECR:**

```bash
# Login
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag ml-pipeline-template:latest \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-pipeline-template:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-pipeline-template:latest
```

**Google Container Registry:**

```bash
# Configure Docker
gcloud auth configure-docker

# Tag and push
docker tag ml-pipeline-template:latest gcr.io/your-project-id/ml-pipeline-template:latest
docker push gcr.io/your-project-id/ml-pipeline-template:latest
```

### Security Scanning

**Using Trivy:**

```bash
# Install Trivy first
# macOS: brew install aquasecurity/trivy/trivy
# Linux: See https://trivy.dev

# Scan image
make docker-scan

# Or manually
trivy image ml-pipeline-template:latest
```

**Using Docker Scan:**

```bash
docker scan ml-pipeline-template:latest
```

---

## Troubleshooting

### Build Issues

**Problem: Build takes too long**

Solution: The multi-stage build is optimized. First build may take 5-10 minutes, subsequent builds use cache and take 1-2 minutes.

**Problem: Dependency installation fails**

```bash
# Clear build cache and rebuild
docker builder prune -f
./docker-build.sh
```

### Runtime Issues

**Problem: Container exits immediately**

```bash
# Check logs
docker logs <container-id>

# Run with interactive shell
docker run -it --rm ml-pipeline-template:latest /bin/bash
```

**Problem: Health check failing**

```bash
# Check if API is starting
docker logs -f <container-id>

# Increase health check timeout in docker-compose.yml
healthcheck:
  start_period: 60s  # Increase from 40s
```

**Problem: Permission denied errors**

The container runs as non-root user (UID 1000). Ensure mounted volumes have correct permissions:

```bash
# Fix permissions
sudo chown -R 1000:1000 mlruns/ artifacts/ logs/
```

### Resource Issues

**Problem: Out of memory**

```bash
# Increase Docker memory limit (Docker Desktop)
# Settings > Resources > Memory: 8GB+

# Or add resource limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

**Problem: Disk space**

```bash
# Clean up Docker resources
make docker-clean

# Or manually
docker system prune -a
docker volume prune
```

### Network Issues

**Problem: Cannot connect to MLflow**

```bash
# Verify MLflow is running
docker-compose ps

# Check network
docker network inspect ml_pipeline_template_ml-network

# Restart services
docker-compose down && docker-compose up -d
```

---

## Image Size Optimization

The multi-stage build produces optimized images:

```bash
# Check image sizes
make docker-size

# Expected sizes:
# - Production (runtime): ~1.2GB
# - Development: ~1.5GB
# - Builder (not kept): ~2GB
```

**Further optimization:**

1. **Minimize dependencies**: Review `requirements/prod.txt`
2. **Use slim base images**: Already using `python:3.11-slim`
3. **Remove build artifacts**: Handled in multi-stage build
4. **Compress layers**: Use `docker save/load` with compression

---

## Environment Variables

Key environment variables in containers:

| Variable | Default | Description |
|----------|---------|-------------|
| PYTHONUNBUFFERED | 1 | Enable unbuffered output |
| PYTHONPATH | /app | Python module search path |
| MLFLOW_TRACKING_URI | http://mlflow:5000 | MLflow server URL |
| PIP_NO_CACHE_DIR | 1 | Disable pip cache |

Override in docker-compose.yml or via `-e` flag:

```bash
docker run -e MLFLOW_TRACKING_URI=http://custom:5000 ml-pipeline-template:latest
```

---

## Best Practices

1. **Use multi-stage builds**: Already implemented for optimal size
2. **Run as non-root**: Container runs as UID 1000 for security
3. **Health checks**: Defined in Dockerfile and docker-compose.yml
4. **Volume mounts**: Use for data, models, and logs (not in image)
5. **Layer caching**: Copy requirements before code for better caching
6. **Security scanning**: Run `make docker-scan` before deployment
7. **Resource limits**: Set CPU/memory limits in production
8. **Logging**: Use structured logging, collect with EFK/Loki
9. **Versioning**: Tag images with git SHA or semantic version
10. **CI/CD**: Automate build/test/push in pipelines

---

## Make Commands Reference

```bash
# Building
make docker-build-prod      # Build production image
make docker-build-dev       # Build development image
make docker-build-amd64     # Build for x86 platforms

# Testing
make docker-test            # Run comprehensive tests
make docker-lint            # Run linting in container
make docker-pytest          # Run pytest in container
make docker-scan            # Security scan with Trivy

# Running
make docker-compose-up      # Start API + MLflow
make docker-compose-dev     # Start dev mode with hot reload
make docker-compose-train   # Run training job
make docker-compose-eval    # Run evaluation job
make docker-compose-down    # Stop all services
make docker-compose-logs    # View logs

# Utilities
make docker-shell           # Interactive shell (production)
make docker-shell-dev       # Interactive shell (development)
make docker-size            # Show image sizes
make docker-clean           # Clean up Docker resources
make docker-push            # Push to registry (requires REGISTRY=)

# Quick Start
make docker-all             # Build, test, and run
make docker-dev-all         # Development workflow
```

---

## Related Documentation

- [Deployment Guide](docs/deployment_guide.md) - Kubernetes deployment
- [API Generation Guide](docs/api_generation_guide.md) - Generate APIs
- [Verification Guide](docs/verification_guide.md) - Testing scripts

---

## Kubernetes Deployment

For professional deployment using these images on Kubernetes, please refer to the automated scripts:

```bash
# High-level deployment wrapper
./deploy.sh [mode]
```

See the **[Zero-to-Hero Deployment Guide](docs/deployment_guide.md)** for full details.

---

## Support

For issues or questions:

1. Check logs: `docker-compose logs -f`
2. Review troubleshooting section above
3. Open issue on GitHub
4. Check Docker documentation: https://docs.docker.com
