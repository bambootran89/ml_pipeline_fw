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

PIPELINE_DIR="mlproject/configs/pipelines"
EXP_DIR="mlproject/configs/experiments"
GENERATED_DIR="mlproject/configs/generated"
LOG_DIR="tmplogs"

# Create log directory
mkdir -p "$LOG_DIR"

# Clean up any existing background server
fuser -k 8082/tcp > /dev/null 2>&1 || true
ray stop > /dev/null 2>&1 || true

cleanup() {
    fuser -k 8082/tcp > /dev/null 2>&1 || true
    ray stop > /dev/null 2>&1 || true
    # Find and kill any process using port 8082 (handles Errno 48)
    if lsof -t -i:8082 > /dev/null 2>&1; then
        lsof -t -i:8082 | xargs kill -9 > /dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}   VERIFYING ALL PIPELINES IN $PIPELINE_DIR${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Iterate over all yaml files in the pipeline directory
for PIPE_PATH in "$PIPELINE_DIR"/*.yaml; do
    PIPE_FILENAME=$(basename "$PIPE_PATH")
    PIPE_NAME="${PIPE_FILENAME%.*}"

    echo -e "\n${YELLOW}-------------------------------------------------------${NC}"
    echo -e "${YELLOW}>>> Processing Pipeline: $PIPE_FILENAME${NC}"
    echo -e "${YELLOW}-------------------------------------------------------${NC}"

    # 1. Determine Experiment Config
    if [[ "$PIPE_FILENAME" == *"tabular"* ]]; then
        EXP_CONFIG="$EXP_DIR/tabular.yaml"
        TYPE="tabular"
        echo "Detected Type: TABULAR"
    elif [[ "$PIPE_FILENAME" == *"feast"* ]]; then
        EXP_CONFIG="$EXP_DIR/etth1_feast.yaml"
        TYPE="feast"
        echo "Detected Type: FEAST"
    else
        EXP_CONFIG="$EXP_DIR/etth3.yaml"
        TYPE="timeseries"
        echo "Detected Type: TIMESERIES"
    fi

    # 2. Train
    echo -e "${BLUE}--- [1/4] Training ---${NC}"
    if python -m mlproject.src.pipeline.dag_run train -e "$EXP_CONFIG" -p "$PIPE_PATH" > "$LOG_DIR/${PIPE_NAME}_train.log" 2>&1; then
        echo -e "${GREEN}Training PASS${NC}"
    else
        echo -e "${RED}Training FAILED${NC}"
        tail -n 20 "$LOG_DIR/${PIPE_NAME}_train.log"
        exit 1
    fi

    # 3. Generate Serve Config
    echo -e "${BLUE}--- [2/4] Generating Serve Config ---${NC}"
    if python -m mlproject.src.pipeline.dag_run generate -t "$PIPE_PATH" -e "$EXP_CONFIG" --type serve > "$LOG_DIR/${PIPE_NAME}_gen.log" 2>&1; then
        echo -e "${GREEN}Generation PASS${NC}"
    else
        echo -e "${RED}Generation FAILED${NC}"
        tail -n 20 "$LOG_DIR/${PIPE_NAME}_gen.log"
        exit 1
    fi

    # Determine Generated Config Name
    SERVE_CONFIG="$GENERATED_DIR/${PIPE_NAME}_serve.yaml"

    if [ ! -f "$SERVE_CONFIG" ]; then
        echo -e "${RED}Serve config not found: $SERVE_CONFIG${NC}"
        ls -l "$GENERATED_DIR"
        exit 1
    fi

    # --- Loop through Frameworks ---
    for FRAMEWORK in "fastapi" "ray"; do
        echo -e "${BLUE}--- [3/4] Serving ($FRAMEWORK) ---${NC}"

        # Ensure clean port/process
        cleanup

        bash ./mlproject/serve_api.sh -e "$EXP_CONFIG" -a latest -f "$FRAMEWORK" "$SERVE_CONFIG" --port 8082 > "$LOG_DIR/${PIPE_NAME}_${FRAMEWORK}_serve.log" 2>&1 &
        SERVER_PID=$!

        # Wait for health
        echo "Waiting for $FRAMEWORK server..."
        MAX_RETRIES=45
        COUNT=0
        HEALTHY=false
        while [ $COUNT -lt $MAX_RETRIES ]; do
            if curl -s http://127.0.0.1:8082/health | grep -q '"status":"healthy"'; then
                HEALTHY=true
                break
            fi
            sleep 2
            COUNT=$((COUNT+1))
        done

        if [ "$HEALTHY" = false ]; then
            echo -e "${RED}Serving Health Check FAILED ($FRAMEWORK)${NC}"
            cat "$LOG_DIR/${PIPE_NAME}_${FRAMEWORK}_serve.log"
            cleanup
            exit 1
        fi
        echo -e "${GREEN}Serving Health Check PASS ($FRAMEWORK)${NC}"

# Function to verify JSON response
verify_response() {
    local RESPONSE="$1"
    local EXPECTED_COUNT="$2"
    local DESCRIPTION="$3"

    echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    # Handle direct prediction list (PredictResponse) or dict (MultiPredictResponse)
    if 'prediction' in data:
        preds = data['prediction']
        print(f'Found single output: Count={len(preds) if isinstance(preds, list) else 1}')
    elif 'predictions' in data:
        preds_dict = data['predictions']
        print(f'Found {len(preds_dict)} outputs: {list(preds_dict.keys())}')
        preds = preds_dict
    else:
        print('FAILED: No predictions found')
        print(f'Full Response: {data}')
        sys.exit(1)

    # Validate count/content
    if not preds:
        print('FAILED: Predictions empty')
        sys.exit(1)

    print('OK')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"
    if [ $? -eq 0 ]; then
         echo -e "${GREEN}>>> $DESCRIPTION VERIFICATION PASSED${NC}"
    else
         echo -e "${RED}>>> $DESCRIPTION VERIFICATION FAILED${NC}"
         echo "$RESPONSE"
         cleanup
         exit 1
    fi
}

        # 5. Verify API
        echo -e "${BLUE}--- [4/4] Verifying API ($FRAMEWORK) ---${NC}"

        if [ "$TYPE" == "tabular" ]; then
            # Tabular Batch
            RESPONSE=$(curl -s -X POST http://localhost:8082/predict/batch \
              -H "Content-Type: application/json" \
              -d '{
                "data": {
                  "Pclass": [3, 1, 3, 1, 3],
                  "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
                  "SibSp": [1, 1, 0, 1, 0],
                  "Parch": [0, 0, 0, 0, 0],
                  "Fare": [7.25, 71.28, 7.92, 53.1, 8.05],
                  "Sex": ["male", "female", "female", "female", "male"],
                  "Embarked": ["S", "C", "S", "S", "S"]
                },
                "return_probabilities": true
            }')
            verify_response "$RESPONSE" 5 "Tabular Batch ($FRAMEWORK)"

        elif [ "$TYPE" == "feast" ]; then
            # 1. Single Entity Test
            RESPONSE_SINGLE=$(curl -s -X POST http://localhost:8082/predict/feast \
               -H "Content-Type: application/json" \
               -d '{
                 "time_point": "2024-01-09T00:00:00",
                 "entities": [1]
               }')
            verify_response "$RESPONSE_SINGLE" 1 "Feast Single Entity ($FRAMEWORK)"

            # 2. Batch Entity Test
            RESPONSE_BATCH=$(curl -s -X POST http://localhost:8082/predict/feast/batch \
               -H "Content-Type: application/json" \
               -d '{
                 "time_point": "2024-01-09T00:00:00",
                 "entities": [1],
                 "entity_key": "location_id"
               }')
            verify_response "$RESPONSE_BATCH" 1 "Feast Batch ($FRAMEWORK)"

        else
            # Timeseries Multi-step
            RESPONSE=$(curl -s -X POST http://localhost:8082/predict/multistep \
              -H "Content-Type: application/json" \
              -d '{
              "data": {
                "date": ["2020-01-01 00:00:00","2020-01-01 01:00:00","2020-01-01 02:00:00","2020-01-01 03:00:00","2020-01-01 04:00:00","2020-01-01 05:00:00","2020-01-01 06:00:00","2020-01-01 07:00:00","2020-01-01 08:00:00","2020-01-01 09:00:00","2020-01-01 10:00:00","2020-01-01 11:00:00","2020-01-01 12:00:00","2020-01-01 13:00:00","2020-01-01 14:00:00","2020-01-01 15:00:00","2020-01-01 16:00:00","2020-01-01 17:00:00","2020-01-01 18:00:00","2020-01-01 19:00:00","2020-01-01 20:00:00","2020-01-01 21:00:00","2020-01-01 22:00:00","2020-01-01 23:00:00","2020-01-02 00:00:00","2020-01-02 01:00:00","2020-01-02 02:00:00","2020-01-02 03:00:00","2020-01-02 04:00:00","2020-01-02 05:00:00","2020-01-02 06:00:00","2020-01-02 07:00:00","2020-01-02 08:00:00","2020-01-02 09:00:00","2020-01-02 10:00:00","2020-01-02 11:00:00"],
                "HUFL": [5.827,5.8,5.969,6.372,7.153,7.976,8.715,9.34,9.763,9.986,10.04,9.916,9.609,9.156,8.591,7.97,7.338,6.745,6.233,5.838,5.582,5.465,5.465,5.557,5.607,5.657,5.707,5.757,5.807,5.857,5.907,5.957,6.007,6.057,6.107,6.157],
                "MUFL": [1.599,1.492,1.492,1.492,1.492,1.509,1.582,1.711,1.896,2.113,2.337,2.552,2.742,2.902,3.024,3.104,3.137,3.125,3.067,2.969,2.838,2.683,2.515,2.346,2.366,2.386,2.406,2.426,2.446,2.466,2.486,2.506,2.526,2.546,2.566,2.586],
                "mobility_inflow": [1.234,1.456,1.678,1.89,2.123,2.456,2.789,3.012,3.234,3.456,3.678,3.89,4.012,4.123,4.234,4.345,4.456,4.567,4.678,4.789,4.89,4.901,4.912,4.923,4.953,4.983,5.013,5.043,5.073,5.103,5.133,5.163,5.193,5.223,5.253,5.283]
              },
              "steps_ahead": 18
            }')
            verify_response "$RESPONSE" 1 "Timeseries Multistep ($FRAMEWORK)"
        fi

        # Cleanup after each framework
        cleanup
        echo ""
    done

done

echo -e "${GREEN}=======================================================${NC}"
echo -e "${GREEN}   ALL PIPELINES VERIFIED SUCCESSFULLY${NC}"
echo -e "${GREEN}=======================================================${NC}"
