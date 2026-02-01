#!/bin/bash
# deploy.sh - Automated deployment script for Feast or Standard mode with Dynamic Config

set -e

# Default Values
MODE="standard"
PIPELINE_CONFIG="standard_train.yaml"
EXPERIMENT_CONFIG=""
NAMESPACE="ml-pipeline"
GENERATED_DIR="k8s/generated"
DRY_RUN=false

# Help Function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -m MODE       Deployment mode: 'standard' or 'feast' (default: standard)"
    echo "  -p CONFIG     Pipeline config file (default: standard_train.yaml)"
    echo "  -e CONFIG     Experiment config file (default: derived from mode)"
    echo "  -n NAMESPACE  Kubernetes namespace (default: ml-pipeline)"
    echo "  -d            Dry run (generate manifests only, do not deploy)"
    echo "  -h            Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -m feast -p conditional_branch_feast.yaml"
    exit 1
}

# Parse Arguments
while getopts "m:p:e:n:dh" opt; do
    case $opt in
        m) MODE="$OPTARG" ;;
        p) PIPELINE_CONFIG="$OPTARG" ;;
        e) EXPERIMENT_CONFIG="$OPTARG" ;;
        n) NAMESPACE="$OPTARG" ;;
        d) DRY_RUN=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Default Experiment Config based on Mode if not set
if [ -z "$EXPERIMENT_CONFIG" ]; then
    if [ "$MODE" == "feast" ]; then
        EXPERIMENT_CONFIG="etth3_feast.yaml"
    else
        EXPERIMENT_CONFIG="etth3.yaml"
    fi
fi

# Derive Serving Config from Pipeline Config
# e.g. standard_train.yaml -> standard_train_serve.yaml
SERVING_CONFIG="${PIPELINE_CONFIG%.yaml}_serve.yaml"

echo "=================================================="
echo " Starting Deployment"
echo " Mode:              [$MODE]"
echo " Pipeline Config:   [$PIPELINE_CONFIG]"
echo " Experiment Config: [$EXPERIMENT_CONFIG]"
echo " Serving Config:    [$SERVING_CONFIG]"
echo " Namespace:         [$NAMESPACE]"
echo " Generated Dir:     [$GENERATED_DIR]"
echo " Dry Run:           [$DRY_RUN]"
echo "=================================================="

# 1. Validation
if [[ "$MODE" != "feast" && "$MODE" != "standard" ]]; then
    echo "Error: Invalid mode '$MODE'. Use 'feast' or 'standard'."
    exit 1
fi

# Check if pipeline config exists (optional strict check)
if [ ! -f "mlproject/configs/pipelines/$PIPELINE_CONFIG" ]; then
    echo "Warning: Config 'mlproject/configs/pipelines/$PIPELINE_CONFIG' not found locally."
fi

# Create Generated Directory
mkdir -p "$GENERATED_DIR"
mkdir -p "mlproject/configs/generated"

# 1.5. Generate Serving Config if not exists
SERVE_CONFIG_PATH="mlproject/configs/generated/$SERVING_CONFIG"
if [ ! -f "$SERVE_CONFIG_PATH" ]; then
    echo "[Pre-Deploy] Generating serving config: $SERVING_CONFIG"
    python -c "
from mlproject.src.generator.orchestrator import ConfigGenerator

try:
    # Generate serve config
    generator = ConfigGenerator(
        train_config_path='mlproject/configs/pipelines/$PIPELINE_CONFIG',
        experiment_config_path='mlproject/configs/experiments/$EXPERIMENT_CONFIG'
    )
    paths = generator.generate_all('mlproject/configs/generated', alias='latest', include_tune=False)
    print(f'[Pre-Deploy] Generated: {paths.get(\"serve\", \"N/A\")}')
except Exception as e:
    print(f'[Pre-Deploy] Warning: Failed to generate serve config: {e}')
    print('[Pre-Deploy] Continuing with existing config if available...')
"
else
    echo "[Pre-Deploy] Using existing serving config: $SERVE_CONFIG_PATH"
fi

# 2. Setup Context
if [ "$DRY_RUN" = false ]; then
    echo "[Setup] Checking cluster connectivity..."
    if ! kubectl cluster-info >/dev/null 2>&1; then
        echo "Error: Unable to connect to Kubernetes cluster."
        echo "Tip: Check if Minikube is running ('minikube status') and Docker is active."
        echo "     If on WSL2 and Docker is stopped, verify with: 'sudo service docker status'"
        echo "     and start with: 'sudo service docker start'"
        exit 1
    fi

    echo "[Setup] Creating namespace $NAMESPACE if not exists..."
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    kubectl config set-context --current --namespace=$NAMESPACE
else
    echo "[Run] Skipping namespace creation (Dry Run)"
fi

# 3. Cleanup Old Resources
if [ "$DRY_RUN" = false ]; then
    echo "[Cleanup] Removing existing jobs and deployments..."
    kubectl delete job training-job-fast-etth1 --ignore-not-found
    kubectl delete job training-job-standard-etth1 --ignore-not-found
    kubectl delete deployment ml-prediction-api-feast --ignore-not-found
    kubectl delete deployment ml-prediction-api-standard --ignore-not-found
else
    echo "[Run] Skipping cleanup (Dry Run)"
fi

# Get Project Root (for hostPath usage)
PROJECT_ROOT=$(pwd)
echo " Project Root:      [$PROJECT_ROOT]"

# 4. Deploy MLflow Server (Shared)
echo "[Deploy] Preparing MLflow Server Manifest..."
MLFLOW_MANIFEST="$GENERATED_DIR/mlflow-server.yaml"
sed -e "s|{{PROJECT_ROOT}}|$PROJECT_ROOT|g" k8s/mlflow-server.yaml > "$MLFLOW_MANIFEST"

if [ "$DRY_RUN" = false ]; then
    echo "[Deploy] Submitting MLflow Server: $MLFLOW_MANIFEST"
    kubectl apply -f "$MLFLOW_MANIFEST"

    echo "[Monitor] Waiting for MLflow to be ready..."
    kubectl wait --for=condition=ready pod -l app=mlflow --timeout=60s || {
        echo "Warning: MLflow pod not ready yet. Check logs."
    }
else
    echo "[Dry Run] Validated MLflow Manifest at: $MLFLOW_MANIFEST"
fi

# 5. Deploy Training Job with Config Injection
echo "[Deploy] Preparing Job Manifest..."
# Create a temporary manifest with the injected config names
JOB_MANIFEST="$GENERATED_DIR/job-training-$MODE.yaml"
# Use | as delimiter for paths
sed -e "s|{{PIPELINE_CONFIG}}|$PIPELINE_CONFIG|g" \
    -e "s|{{EXPERIMENT_CONFIG}}|$EXPERIMENT_CONFIG|g" \
    -e "s|{{PROJECT_ROOT}}|$PROJECT_ROOT|g" \
    k8s/job-training-$MODE.yaml > "$JOB_MANIFEST"

if [ "$DRY_RUN" = false ]; then
    echo "[Deploy] Submitting Training Job: $JOB_MANIFEST"
    kubectl apply -f "$JOB_MANIFEST"

    echo "[Monitor] Waiting for training job to start..."
    sleep 5
    kubectl get jobs
else
    echo "[Dry Run] Validated Job Manifest at: $JOB_MANIFEST"
fi

# 5. Deploy API Service with Config Injection
echo "[Deploy] Preparing API Deployment Manifest..."
API_MANIFEST="$GENERATED_DIR/deployment-api-$MODE.yaml"
sed -e "s|{{PIPELINE_CONFIG}}|$PIPELINE_CONFIG|g" \
    -e "s|{{EXPERIMENT_CONFIG}}|$EXPERIMENT_CONFIG|g" \
    -e "s|{{SERVING_CONFIG}}|$SERVING_CONFIG|g" \
    -e "s|{{PROJECT_ROOT}}|$PROJECT_ROOT|g" \
    k8s/deployment-api-$MODE.yaml > "$API_MANIFEST"

if [ "$DRY_RUN" = false ]; then
    echo "[Deploy] Submitting API Deployment: $API_MANIFEST"
    kubectl apply -f "$API_MANIFEST"

    # 6. Apply Service (Common)
    echo "[Deploy] Ensuring Service exists..."
    kubectl apply -f k8s/service-api.yaml
else
    echo "[Dry Run] Validated API Manifest at: $API_MANIFEST"
fi

echo "=================================================="
if [ "$DRY_RUN" = false ]; then
    echo " Deployment Submitted Successfully!"
    echo " Generated manifests are in '$GENERATED_DIR'"
    echo " Use 'kubectl get pods' to monitor progress."
else
    echo " Dry Run Complete!"
    echo " Manifests generated in '$GENERATED_DIR'"
fi
echo "=================================================="
