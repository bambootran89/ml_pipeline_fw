#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Setup environment
export PYTHONPATH=.
if [ -d "mlproject_env" ]; then
    source mlproject_env/bin/activate
fi

LOG_DIR="tmplogs/feast_verify"
mkdir -p "$LOG_DIR"

# Find and kill any process using port 8082 (handles Errno 48)
if lsof -t -i:8082 > /dev/null 2>&1; then
    lsof -t -i:8082 | xargs kill -9 > /dev/null 2>&1 || true
fi

# Clean up function to release port 8082 and stop Ray
cleanup() {
    # Alternative kill method for Linux environments
    fuser -k 8082/tcp > /dev/null 2>&1 || true

    # Stop Ray runtime
    ray stop > /dev/null 2>&1 || true
}

# Ensure cleanup runs on script exit
trap cleanup EXIT

# Verify Response Function
verify_response() {
    local RESPONSE="$1"
    local EXPECTED_COUNT="$2"
    local DESCRIPTION="$3"

    echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'prediction' in data:
        preds = data['prediction']
        count = len(preds) if isinstance(preds, list) else 1
        sample = str(preds)[:100]
    elif 'predictions' in data:
        preds = data['predictions']
        count = len(preds)
        sample = str(preds)[:100]
    else:
        print(f'FAILED: Invalid response structure: {data}')
        sys.exit(1)

    if not preds:
        print('FAILED: Empty predictions')
        sys.exit(1)

    print(f'OK: Found {count} preds. Sample: {sample}...')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"
    if [ $? -eq 0 ]; then
         echo -e "${GREEN}>>> $DESCRIPTION: PASSED${NC}"
    else
         echo -e "${RED}>>> $DESCRIPTION: FAILED${NC}"
         echo "Response: $RESPONSE"
         return 1
    fi
}

# Define Test Cases: "ExperimentConfig|PipelineConfig"
TEST_CASES=(
    "mlproject/configs/experiments/etth3_feast.yaml|mlproject/configs/pipelines/standard_train.yaml"
    "mlproject/configs/experiments/feast_tabular.yaml|mlproject/configs/pipelines/standard_train.yaml"
    "mlproject/configs/experiments/etth3_feast.yaml|mlproject/configs/pipelines/kmeans_then_xgboost.yaml"
    "mlproject/configs/experiments/feast_tabular.yaml|mlproject/configs/pipelines/kmeans_then_xgboost.yaml"
    "mlproject/configs/experiments/etth3_feast.yaml|mlproject/configs/pipelines/parallel_ensemble.yaml"
    "mlproject/configs/experiments/feast_tabular.yaml|mlproject/configs/pipelines/parallel_ensemble.yaml"
    "mlproject/configs/experiments/etth3_feast.yaml|mlproject/configs/pipelines/nested_suppipeline.yaml"
    "mlproject/configs/experiments/etth3_feast.yaml|mlproject/configs/pipelines/dynamic_adapter_train.yaml"
    "mlproject/configs/experiments/feast_tabular.yaml|mlproject/configs/pipelines/dynamic_adapter_train.yaml"
)

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}   STARTING COMPREHENSIVE FEAST VERIFICATION (FASTAPI + RAY)${NC}"
echo -e "${BLUE}=======================================================${NC}"

for CASE in "${TEST_CASES[@]}"; do
    IFS="|" read -r EXP_CONFIG PIPE_CONFIG <<< "$CASE"

    EXP_NAME=$(basename "$EXP_CONFIG" .yaml)
    PIPE_NAME=$(basename "$PIPE_CONFIG" .yaml)
    RUN_ID="${PIPE_NAME}_${EXP_NAME}"

    echo -e "\n${YELLOW}-------------------------------------------------------${NC}"
    echo -e "${YELLOW}Testing: Pipeline=[$PIPE_NAME] Experiment=[$EXP_NAME]${NC}"
    echo -e "${YELLOW}-------------------------------------------------------${NC}"

    # 1. Train
    echo -e "${BLUE}--- [1/3] Training ---${NC}"
    if python -m mlproject.src.pipeline.dag_run train -e "$EXP_CONFIG" \
        -p "$PIPE_CONFIG" > "$LOG_DIR/${RUN_ID}_train.log" 2>&1; then
        echo -e "${GREEN}Training PASS${NC}"
    else
        echo -e "${RED}Training FAILED${NC}"; tail -n 20 "$LOG_DIR/${RUN_ID}_train.log"; continue
    fi

    # 2. Generate Serve Config
    echo -e "${BLUE}--- [2/3] Generation ---${NC}"
    if python -m mlproject.src.pipeline.dag_run generate -t "$PIPE_CONFIG" \
        -e "$EXP_CONFIG" --type serve > "$LOG_DIR/${RUN_ID}_gen.log" 2>&1; then
        echo -e "${GREEN}Generation PASS${NC}"
    else
        echo -e "${RED}Generation FAILED${NC}"; tail -n 20 "$LOG_DIR/${RUN_ID}_gen.log"; continue
    fi

    SERVE_CONFIG="mlproject/configs/generated/${PIPE_NAME}_serve.yaml"

    # 3. Verify Serving (Loop Frameworks)
    for FRAMEWORK in "fastapi" "ray"; do
        echo -e "${BLUE}--- [3/3] Serving ($FRAMEWORK) ---${NC}"
        # Clear port before starting new service
        cleanup

        bash ./mlproject/serve_api.sh -e "$EXP_CONFIG" -a latest -f "$FRAMEWORK" \
            "$SERVE_CONFIG" --port 8082 > "$LOG_DIR/${RUN_ID}_${FRAMEWORK}_serve.log" 2>&1 &

        # Wait for health status
        COUNT=0
        HEALTHY=false
        while [ $COUNT -lt 45 ]; do
            if curl -s http://127.0.0.1:8082/health | grep -q '"status":"healthy"'; then
                HEALTHY=true
                break
            fi
            sleep 2
            COUNT=$((COUNT+1))
        done

        if [ "$HEALTHY" = false ]; then
            echo -e "${RED}Serving Health Check FAILED ($FRAMEWORK)${NC}"
            cat "$LOG_DIR/${RUN_ID}_${FRAMEWORK}_serve.log"
            cleanup
            continue
        fi

        # Verify based on Experiment Type
        if [[ "$EXP_NAME" == *"tabular"* ]]; then
            # Tabular Prediction
            RESP=$(curl -s -X POST http://localhost:8082/predict/feast \
                -H "Content-Type: application/json" -d '{"entities": [1]}')
            verify_response "$RESP" 1 "Tabular Single ($FRAMEWORK)"

            RESP=$(curl -s -X POST http://localhost:8082/predict/feast/batch \
                -H "Content-Type: application/json" \
                -d '{"entities": [1, 2, 3], "entity_key": "passenger_id"}')
            verify_response "$RESP" 3 "Tabular Batch ($FRAMEWORK)"
        else
            # Timeseries Prediction
            RESP=$(curl -s -X POST http://localhost:8082/predict/feast \
                -H "Content-Type: application/json" \
                -d '{"time_point": "2024-01-09T00:00:00", "entities": [1]}')
            verify_response "$RESP" 1 "Timeseries Single ($FRAMEWORK)"

            RESP=$(curl -s -X POST http://localhost:8082/predict/feast/batch \
                -H "Content-Type: application/json" \
                -d '{"time_point": "2024-01-09T00:00:00", "entities": [1], "entity_key": "location_id"}')
            verify_response "$RESP" 1 "Timeseries Batch ($FRAMEWORK)"
        fi

        cleanup
    done
    cleanup
done

echo -e "\n${GREEN}Feature Integration Verification Complete${NC}"
