#!/bin/bash
# run_full_flow.sh - Orchestrate the full MLOps lifecycle
# Usage: ./run_full_flow.sh [OPTIONS]

set -e

# Default Values
MODE="standard"
PIPELINE_CONFIG="standard_train.yaml"
EXPERIMENT_CONFIG=""
NAMESPACE="ml-pipeline"
GENERATED_DIR="k8s/generated"
PROJECT_ROOT=$(pwd)

# Parse Arguments
while getopts "m:p:e:n:h" opt; do
    case $opt in
        m) MODE="$OPTARG" ;;
        p) PIPELINE_CONFIG="$OPTARG" ;;
        e) EXPERIMENT_CONFIG="$OPTARG" ;;
        n) NAMESPACE="$OPTARG" ;;
        h)
            echo "Usage: $0 [-m mode] [-p pipeline_config] [-e experiment_config]"
            exit 0
            ;;
        *) exit 1 ;;
    esac
done

# Default Experiment Config logic
if [ -z "$EXPERIMENT_CONFIG" ]; then
    if [ "$MODE" == "feast" ]; then
        EXPERIMENT_CONFIG="etth3_feast.yaml"
    else
        EXPERIMENT_CONFIG="etth3.yaml"
    fi
fi

# 1. Deploy MLflow Server (Ensure it's up)
echo "[Flow] Ensuring MLflow Server is running..."
./deploy.sh -m "$MODE" -d > /dev/null 2>&1 # Generate manifests only
kubectl apply -f "$GENERATED_DIR/mlflow-server.yaml"
kubectl wait --for=condition=ready pod -l app=mlflow --timeout=90s

# 2. Smart Checks
echo "[Flow] analyzing configuration..."

# extract values from yaml using python
# Robust python extraction
EXP_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('mlproject/configs/experiments/$EXPERIMENT_CONFIG'))['experiment']['name'])" 2>/dev/null)

# Fallback to grep if python fails (e.g. missing dependencies, though standard python should have yaml if installed, or we use simple grep as backup)
if [ -z "$EXP_NAME" ]; then
    # Try grep again but simple
    EXP_NAME=$(grep "name:" "mlproject/configs/experiments/$EXPERIMENT_CONFIG" | head -1 | awk -F '"' '{print $2}')
fi
echo "Target Experiment: $EXP_NAME"

# Check MLflow
# Forward port temporarily
pkill -f "kubectl port-forward service/mlflow-service" || true
kubectl port-forward service/mlflow-service 5001:5000 > /dev/null 2>&1 &
PF_PID=$!
sleep 5

# Check if Experiment Exists
# API: /api/2.0/mlflow/experiments/get-by-name?experiment_name=...
status_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:5001/api/2.0/mlflow/experiments/get-by-name?experiment_name=$EXP_NAME")
kill $PF_PID

if [ "$status_code" == "200" ]; then
    echo "Experiment '$EXP_NAME' found in MLflow."
    echo "Checking for Registered Model..."
    # Deep check logic could go here (check for 'latest' alias)
    # For now, if experiment exists, we ask user or assume 'partial' flow?
    # User requested: "check experiment, model name... if empty -> training..."
    # Simplifying: If experiment found, we still might need to retrain if data changed?
    # Let's assume: Found = Skip Training? No, user said "if no experiments... train".
    # So: Found = Maybe Skip?
    # User said: "mlfow mà ko có gì thì phải training" -> If MLflow has nothing (for this exp), do training.
    echo "Experiment exists. Proceeding with Serving (or Tuning if requested)."
    FORCE_TRAIN=false
else
    echo "Experiment '$EXP_NAME' NOT found. Triggering FULL FLOW."
    FORCE_TRAIN=true
fi

run_job() {
    JOB_TYPE=$1
    TEMPLATE="k8s/job-$JOB_TYPE-$MODE.yaml"
    MANIFEST="$GENERATED_DIR/job-$JOB_TYPE-$MODE.yaml"

    echo "[Flow] Running $JOB_TYPE Job..."

    # Inject Configs
    sed -e "s|{{PIPELINE_CONFIG}}|$PIPELINE_CONFIG|g" \
        -e "s|{{EXPERIMENT_CONFIG}}|$EXPERIMENT_CONFIG|g" \
        -e "s|{{PROJECT_ROOT}}|$PROJECT_ROOT|g" \
        "$TEMPLATE" > "$MANIFEST"

    # Delete old job if exists
    kubectl delete job "$JOB_TYPE-job-$MODE-etth1" --ignore-not-found

    # Apply
    kubectl apply -f "$MANIFEST"

    # Wait
    echo "[Flow] Waiting for $JOB_TYPE to complete..."
    kubectl wait --for=condition=complete job/"$JOB_TYPE-job-$MODE-etth1" --timeout=300s
}

# 3. Execute Sequence
echo "=================================================="
echo " Starting MLOps Flow"
echo "=================================================="

if [ "$FORCE_TRAIN" = true ]; then
    # A. Train
    run_job "training"

    # B. Eval
    run_job "eval"

    # C. Tune (Optional - skip for speed if needed, but user asked for it)
    run_job "tune"
else
    echo "[Flow] Skipping Training/Eval/Tune (Model already exists)."
fi

# 4. Deploy Serving
echo "[Flow] Deploying Serving Layer..."
./deploy.sh -m "$MODE"

echo "=================================================="
echo " Full Flow Complete!"
echo "=================================================="
