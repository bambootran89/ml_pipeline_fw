SHELL = /bin/bash
PYTHON := python
VENV_NAME = mlproject_env
MAIN_FOLDER = mlproject
TEST_FOLDER = tests
# Docker variables
IMAGE_NAME = ml-pipeline-template
TAG = latest
# Environment
venv:
	${PYTHON} -m venv ${VENV_NAME} && \
	source ${VENV_NAME}/bin/activate && \
	${PYTHON} -m pip install pip setuptools wheel && \
	${PYTHON} -m pip install -r requirements/dev.txt && \
	${PYTHON} -m pip install -e .[dev] && \
	pre-commit install

# Style
style:
	black ./${MAIN_FOLDER}/ --line-length 88
	${PYTHON} -m isort -rc ./${MAIN_FOLDER}/
	${PYTHON} -m  autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r mlproject/
	${PYTHON} -m  autopep8 --in-place --aggressive --aggressive --aggressive -r mlproject/
	flake8 ./${MAIN_FOLDER}/ --exclude=*/generated/*
	${PYTHON} -m pylint ./${MAIN_FOLDER}/

test:
	${PYTHON} -m flake8 ./${MAIN_FOLDER}/ --exclude=*/generated/*
	${PYTHON} -m mypy ./${MAIN_FOLDER}/
	CUDA_VISIBLE_DEVICES=""  ${PYTHON} -m pytest -v -s --durations=0 --disable-warnings ${TEST_FOLDER}/
	${PYTHON} -m pylint ./${MAIN_FOLDER}/


docker-build:
	docker build -t $(IMAGE_NAME):$(TAG) .

docker-run-api:
	docker run -p 8000:8000 --name ml-api --rm $(IMAGE_NAME):$(TAG)

docker-run-train:
	# Ví dụ chạy training job bằng docker container
	docker run --rm $(IMAGE_NAME):$(TAG) python -m mlproject.src.pipeline.run_pipeline train --config mlproject/configs/experiments/etth1.yaml

# ============================================================================
# Docker Commands (Multi-stage builds)
# ============================================================================

.PHONY: docker-build-prod docker-build-dev docker-test docker-push \
        docker-clean docker-compose-up docker-compose-down

# Build production image (optimized, multi-stage)
docker-build-prod:
	@echo "Building production image..."
	@./docker-build.sh -t $(TAG) -s runtime

# Build development image (with dev tools)
docker-build-dev:
	@echo "Building development image..."
	@./docker-build.sh -t dev -s development

# Build for specific platform (e.g., for M1/M2 deploying to x86)
docker-build-amd64:
	@echo "Building for linux/amd64..."
	@./docker-build.sh -t $(TAG) -p linux/amd64

# Test Docker image
docker-test:
	@echo "Testing Docker image..."
	@./docker-test.sh

# Run API with docker-compose
docker-compose-up:
	docker-compose up -d api mlflow

# Run API in development mode
docker-compose-dev:
	docker-compose --profile dev up -d api-dev mlflow

# Run training job with docker-compose
docker-compose-train:
	docker-compose --profile train run --rm train

# Run evaluation job with docker-compose
docker-compose-eval:
	docker-compose --profile eval run --rm eval

# Stop all services
docker-compose-down:
	docker-compose down

# View logs
docker-compose-logs:
	docker-compose logs -f

# Push to registry (requires REGISTRY variable)
docker-push:
	@if [ -z "$(REGISTRY)" ]; then \
		echo "Error: REGISTRY variable not set"; \
		echo "Usage: make docker-push REGISTRY=yourusername"; \
		exit 1; \
	fi
	@echo "Tagging and pushing to $(REGISTRY)/$(IMAGE_NAME):$(TAG)..."
	@docker tag $(IMAGE_NAME):$(TAG) $(REGISTRY)/$(IMAGE_NAME):$(TAG)
	@docker push $(REGISTRY)/$(IMAGE_NAME):$(TAG)

# Clean up Docker resources
docker-clean:
	@echo "Cleaning up Docker resources..."
	@docker-compose down -v
	@docker rmi $(IMAGE_NAME):$(TAG) $(IMAGE_NAME):dev 2>/dev/null || true
	@docker system prune -f

# Show Docker image size
docker-size:
	@docker images $(IMAGE_NAME) --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# ============================================================================
# Docker Helper Commands
# ============================================================================

# Interactive shell in production container
docker-shell:
	docker run -it --rm \
		-v $(PWD)/mlruns:/app/mlruns \
		$(IMAGE_NAME):$(TAG) /bin/bash

# Interactive shell in development container
docker-shell-dev:
	docker run -it --rm \
		-v $(PWD):/app \
		$(IMAGE_NAME):dev /bin/bash

# Run linting in Docker
docker-lint:
	docker run --rm \
		-v $(PWD):/app \
		$(IMAGE_NAME):dev \
		sh -c "flake8 mlproject/ && mypy mlproject/"

# Run tests in Docker
docker-pytest:
	docker run --rm \
		-v $(PWD):/app \
		$(IMAGE_NAME):dev \
		pytest -v tests/

# Security scan with Trivy (requires trivy installed)
docker-scan:
	@command -v trivy >/dev/null 2>&1 || { echo "Trivy not installed. Install from https://trivy.dev"; exit 1; }
	@trivy image $(IMAGE_NAME):$(TAG)

# ============================================================================
# Quick Start Commands
# ============================================================================

# Full Docker workflow: build, test, and run
docker-all: docker-build-prod docker-test docker-compose-up
	@echo "Docker setup complete!"
	@echo "API running at http://localhost:8000"
	@echo "MLflow UI at http://localhost:5000"

# Development workflow
docker-dev-all: docker-build-dev docker-compose-dev
	@echo "Development environment ready!"
	@echo "API (hot reload) at http://localhost:8001"
	@echo "MLflow UI at http://localhost:5000"

# ============================================================================
# Separated Docker Commands (Training vs Serving)
# ============================================================================

.PHONY: docker-build-train docker-build-serve docker-train docker-eval \
        docker-serve docker-serve-feast docker-train-dev

# Build training image (in-house)
docker-build-train:
	@echo "Building training image..."
	@./docker-build.sh -i train -t $(TAG)

# Build serving image (customer)
docker-build-serve:
	@echo "Building serving image..."
	@./docker-build.sh -i serve -t $(TAG)

# Build both images
docker-build-all: docker-build-train docker-build-serve

# Run training job
docker-train:
	docker-compose --profile train run --rm train

# Run evaluation job
docker-eval:
	docker-compose --profile eval run --rm evaluate

# Run tuning job
docker-tune:
	docker-compose --profile tune run --rm tune

# Start serving API (customer deployment)
docker-serve:
	docker-compose up -d api mlflow

# Start serving with Feast
docker-serve-feast:
	docker-compose --profile feast up -d api-feast mlflow feast-registry

# Training development environment
docker-train-dev:
	docker-compose --profile dev run --rm train-dev

# Stop all services
docker-stop:
	docker-compose down

# Show image sizes (separated)
docker-size-all:
	@echo "Training images:"
	@docker images ml-pipeline-train --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
	@echo ""
	@echo "Serving images:"
	@docker images ml-pipeline-serve --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
