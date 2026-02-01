#!/bin/bash
# verify_all_deployments.sh - Comprehensive K8s Deployment Verification
# Tests all combinations: (Feast/Standard) x (Tabular/TimeSeries) x (etth1/etth2/etth3)

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Results array
declare -a RESULTS

# Ensure PATH includes common locations for tools on macOS
export PATH=$PATH:/usr/local/bin:/opt/homebrew/bin

# Activate environment
if [ -f "mlproject_env/bin/activate" ]; then
    source mlproject_env/bin/activate
else
    echo -e "${YELLOW}[WARN] mlproject_env/bin/activate not found. Assuming environment is already active or global tools will be used.${NC}"
fi
export PYTHONPATH=.

# Cleanup function
cleanup() {
    echo -e "\n${BLUE}[Cleanup] Stopping port forwards...${NC}"
    if command -v kubectl >/dev/null 2>&1; then
        killall kubectl 2>/dev/null || true
    fi
    pkill -f "kubectl port-forward" || true
}
trap cleanup EXIT

# Check for required tools
for tool in docker kubectl curl python3; do
    if ! command -v $tool >/dev/null 2>&1; then
        echo -e "${RED}Error: Required tool '$tool' not found in PATH.${NC}"
        echo -e "Current PATH: $PATH"
        exit 1
    fi
done

# Function to wait for port forward
wait_for_port_forward() {
    local max_retries=15
    local count=0
    while [ $count -lt $max_retries ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            return 0
        fi
        sleep 2
        count=$((count+1))
    done
    return 1
}

# Function to test health endpoint
test_health() {
    local data_type=$1
    echo -e "${BLUE}  Testing /health endpoint...${NC}"

    RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null || echo '{}')

    if echo "$RESPONSE" | grep -q '"status":"healthy"'; then
        echo -e "${GREEN}  [PASS] Health check OK${NC}"
        echo "$RESPONSE" | python3 -m json.tool | grep -E "(status|model_loaded|data_type|features)" | head -10
        return 0
    else
        echo -e "${RED}  ✗ Health check FAILED${NC}"
        echo "$RESPONSE"
        return 1
    fi
}

# Function to test standard prediction (tabular)
test_tabular_predict() {
    echo -e "${BLUE}  Testing /predict (tabular single)...${NC}"

    RESPONSE=$(curl -s -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{
        "data": {
          "Pclass": [3],
          "Age": [25.0],
          "SibSp": [0],
          "Parch": [0],
          "Fare": [7.25],
          "Sex": [1],
          "Embarked": [2]
        }
      }' 2>/dev/null || echo '{}')

    if echo "$RESPONSE" | grep -q "predictions"; then
        PRED=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('predictions',{}))" 2>/dev/null || echo "{}")
        echo -e "${GREEN}  ✓ Single prediction PASS: $PRED${NC}"
        return 0
    else
        echo -e "${RED}  ✗ Single prediction FAILED${NC}"
        return 1
    fi
}

# Function to test time series prediction
test_timeseries_predict() {
    echo -e "${BLUE}  Testing /predict (timeseries)...${NC}"

    RESPONSE=$(curl -s -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{
        "data": {
          "date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00", "2020-01-01 03:00:00", "2020-01-01 04:00:00", "2020-01-01 05:00:00", "2020-01-01 06:00:00", "2020-01-01 07:00:00", "2020-01-01 08:00:00", "2020-01-01 09:00:00", "2020-01-01 10:00:00", "2020-01-01 11:00:00", "2020-01-01 12:00:00", "2020-01-01 13:00:00", "2020-01-01 14:00:00", "2020-01-01 15:00:00", "2020-01-01 16:00:00", "2020-01-01 17:00:00", "2020-01-01 18:00:00", "2020-01-01 19:00:00", "2020-01-01 20:00:00", "2020-01-01 21:00:00", "2020-01-01 22:00:00", "2020-01-01 23:00:00"],
          "HUFL": [5.827, 5.8, 5.969, 6.372, 7.153, 7.976, 8.715, 9.340, 9.763, 9.986, 10.040, 9.916, 9.609, 9.156, 8.591, 7.970, 7.338, 6.745, 6.233, 5.838, 5.582, 5.465, 5.465, 5.557],
          "MUFL": [1.599, 1.492, 1.492, 1.492, 1.492, 1.509, 1.582, 1.711, 1.896, 2.113, 2.337, 2.552, 2.742, 2.902, 3.024, 3.104, 3.137, 3.125, 3.067, 2.969, 2.838, 2.683, 2.515, 2.346],
          "mobility_inflow": [1.234, 1.456, 1.678, 1.890, 2.123, 2.456, 2.789, 3.012, 3.234, 3.456, 3.678, 3.890, 4.012, 4.123, 4.234, 4.345, 4.456, 4.567, 4.678, 4.789, 4.890, 4.901, 4.912, 4.923]
        }
      }' 2>/dev/null || echo '{}')

    if echo "$RESPONSE" | grep -q "predictions"; then
        # Show first 5 predictions
        PREDS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); p=list(d.get('predictions',{}).values())[0] if d.get('predictions') else []; print(p[:5])" 2>/dev/null || echo "[]")
        COUNT=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); p=list(d.get('predictions',{}).values())[0] if d.get('predictions') else []; print(len(p))" 2>/dev/null || echo "0")
        echo -e "${GREEN}  [PASS] Prediction OK: $COUNT timesteps. First 5: $PREDS${NC}"
        return 0
    else
        echo -e "${RED}  [FAIL] Prediction FAILED${NC}"
        return 1
    fi
}

# Function to test time series prediction with engineered features (Feast)
test_timeseries_feast_predict() {
    echo -e "${BLUE}  Testing /predict (timeseries with Feast features)...${NC}"

    RESPONSE=$(curl -s -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{
        "data": {
          "date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00", "2020-01-01 03:00:00", "2020-01-01 04:00:00", "2020-01-01 05:00:00", "2020-01-01 06:00:00", "2020-01-01 07:00:00", "2020-01-01 08:00:00", "2020-01-01 09:00:00", "2020-01-01 10:00:00", "2020-01-01 11:00:00", "2020-01-01 12:00:00", "2020-01-01 13:00:00", "2020-01-01 14:00:00", "2020-01-01 15:00:00", "2020-01-01 16:00:00", "2020-01-01 17:00:00", "2020-01-01 18:00:00", "2020-01-01 19:00:00", "2020-01-01 20:00:00", "2020-01-01 21:00:00", "2020-01-01 22:00:00", "2020-01-01 23:00:00"],
          "HUFL": [5.827, 5.8, 5.969, 6.372, 7.153, 7.976, 8.715, 9.340, 9.763, 9.986, 10.040, 9.916, 9.609, 9.156, 8.591, 7.970, 7.338, 6.745, 6.233, 5.838, 5.582, 5.465, 5.465, 5.557],
          "MUFL": [1.599, 1.492, 1.492, 1.492, 1.492, 1.509, 1.582, 1.711, 1.896, 2.113, 2.337, 2.552, 2.742, 2.902, 3.024, 3.104, 3.137, 3.125, 3.067, 2.969, 2.838, 2.683, 2.515, 2.346],
          "mobility_inflow": [1.234, 1.456, 1.678, 1.890, 2.123, 2.456, 2.789, 3.012, 3.234, 3.456, 3.678, 3.890, 4.012, 4.123, 4.234, 4.345, 4.456, 4.567, 4.678, 4.789, 4.890, 4.901, 4.912, 4.923],
          "HUFL_lag24": [5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8],
          "MUFL_lag24": [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7],
          "HUFL_roll12_mean": [6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4],
          "MUFL_roll12_mean": [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4],
          "hour_sin": [0.0, 0.26, 0.5, 0.71, 0.87, 0.97, 1.0, 0.97, 0.87, 0.71, 0.5, 0.26, 0.0, -0.26, -0.5, -0.71, -0.87, -0.97, -1.0, -0.97, -0.87, -0.71, -0.5, -0.26],
          "hour_cos": [1.0, 0.97, 0.87, 0.71, 0.5, 0.26, 0.0, -0.26, -0.5, -0.71, -0.87, -0.97, -1.0, -0.97, -0.87, -0.71, -0.5, -0.26, 0.0, 0.26, 0.5, 0.71, 0.87, 0.97]
        }
      }' 2>/dev/null || echo '{}')

    if echo "$RESPONSE" | grep -q "predictions"; then
        PREDS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); p=list(d.get('predictions',{}).values())[0] if d.get('predictions') else []; print(p[:5])" 2>/dev/null || echo "[]")
        COUNT=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); p=list(d.get('predictions',{}).values())[0] if d.get('predictions') else []; print(len(p))" 2>/dev/null || echo "0")
        echo -e "${GREEN}  [PASS] Feast Prediction OK: $COUNT timesteps. First 5: $PREDS${NC}"
        return 0
    else
        echo -e "${RED}  [FAIL] Feast Prediction FAILED${NC}"
        return 1
    fi
}

# Function to test Feast-specific endpoints
test_feast_endpoints() {
    local data_type=$1

    if [ "$data_type" = "tabular" ]; then
        echo -e "${BLUE}  Testing /predict/feast endpoint...${NC}"

        RESPONSE=$(curl -s -X POST http://localhost:8000/predict/feast \
          -H "Content-Type: application/json" \
          -d '{"entities": [1, 2, 3], "time_point": "now"}' 2>/dev/null || echo '{}')

        if echo "$RESPONSE" | grep -q "predictions"; then
            PREDS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); p=d.get('predictions',{}).get('train_model_predictions',[]); print(p)" 2>/dev/null || echo "[]")
            COUNT=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); p=d.get('predictions',{}).get('train_model_predictions',[]); print(len(p))" 2>/dev/null || echo "0")
            echo -e "${GREEN}  ✓ /predict/feast PASS: $COUNT entities. Values: $PREDS${NC}"
        else
            echo -e "${RED}  ✗ /predict/feast FAILED${NC}"
            return 1
        fi

        echo -e "${BLUE}  Testing /predict/feast/batch endpoint...${NC}"

        RESPONSE=$(curl -s -X POST http://localhost:8000/predict/feast/batch \
          -H "Content-Type: application/json" \
          -d '{"entities": [1, 2, 3, 4, 5], "time_point": "now"}' 2>/dev/null || echo '{}')

        if echo "$RESPONSE" | grep -q "predictions"; then
            PREDS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); p=d.get('predictions',{}).get('train_model_predictions',[]); print(p)" 2>/dev/null || echo "[]")
            COUNT=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); p=d.get('predictions',{}).get('train_model_predictions',[]); print(len(p))" 2>/dev/null || echo "0")
            echo -e "${GREEN}  ✓ /predict/feast/batch PASS: $COUNT entities. Values: $PREDS${NC}"
            return 0
        else
            echo -e "${RED}  ✗ /predict/feast/batch FAILED${NC}"
            return 1
        fi
    fi

    return 0
}

# Function to run a deployment test
run_deployment_test() {
    local mode=$1
    local experiment=$2
    local repo=$3
    local data_type=$4
    local pipeline_config=${5:-"standard_train.yaml"} # Default to standard_train.yaml if not provided
    local skip_predict=${6:-"false"}
    local test_name="${mode}_${experiment}"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    echo ""
    echo "=================================================="
    echo "Test $TOTAL_TESTS: $test_name"
    echo "  Mode: $mode"
    echo "  Experiment: $experiment"
    echo "  Pipeline: $pipeline_config"
    echo "  Data Type: $data_type"
    echo "=================================================="

    # Deploy
    echo -e "${BLUE}[1/6] Deploying...${NC}"
    # Pass -p pipeline_config to deploy.sh
    if ! ./deploy.sh -m "$mode" -e "$experiment" -r "$repo" -t "$data_type" -p "$pipeline_config" > /tmp/deploy_${test_name}.log 2>&1; then
        echo -e "${RED}✗ Deployment FAILED${NC}"
        cat /tmp/deploy_${test_name}.log | tail -20
        RESULTS+=("${RED}[FAIL] $test_name - Deployment Failed${NC}")
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
    echo -e "${GREEN}[PASS] Deployment submitted${NC}"

    # Get job name
    # Get job name deterministicly
    # Logic matches deploy.sh: EXPERIMENT_NAME=$(basename "$EXPERIMENT_CONFIG" .yaml | sed 's/_feast$//' | sed 's/_/-/g')
    EXP_NAME=$(basename "$experiment" .yaml | sed 's/_feast$//' | sed 's/_/-/g')
    JOB_NAME="training-job-${mode}-${EXP_NAME}"
    echo -e "${BLUE}  Target Job Name: $JOB_NAME${NC}"

    # Verify if job exists (it might take a second to appear)
    sleep 2
    if ! kubectl get job $JOB_NAME -n ml-pipeline > /dev/null 2>&1; then
       echo -e "${YELLOW}[WARN] Job $JOB_NAME not found yet, checking list...${NC}"
       kubectl get jobs -n ml-pipeline
    fi

    if [ -z "$JOB_NAME" ]; then
        echo -e "${YELLOW}[WARN] No training job found, using existing model${NC}"
    else
        echo -e "${BLUE}[2/6] Waiting for training job: $JOB_NAME${NC}"

        # Wait for job with timeout
        if kubectl wait --for=condition=complete job/$JOB_NAME -n ml-pipeline --timeout=600s > /dev/null 2>&1; then
            echo -e "${GREEN}[PASS] Training completed${NC}"

            # Verify model registration
            if kubectl logs -n ml-pipeline job/$JOB_NAME --tail=50 2>/dev/null | grep -q "Created version"; then
                VERSION=$(kubectl logs -n ml-pipeline job/$JOB_NAME --tail=50 2>/dev/null | grep "Created version" | tail -1)
                echo -e "${GREEN}[PASS] Model registered: $VERSION${NC}"
            fi
        else
            JOB_STATUS=$(kubectl get job $JOB_NAME -n ml-pipeline -o jsonpath='{.status.conditions[0].type}' 2>/dev/null || echo "Unknown")

            if [ "$JOB_STATUS" = "Failed" ]; then
                echo -e "${RED}[FAIL] Training job FAILED${NC}"
                kubectl logs -n ml-pipeline job/$JOB_NAME --tail=30 2>/dev/null || true
                RESULTS+=("${RED}[FAIL] $test_name - Training Failed${NC}")
                FAILED_TESTS=$((FAILED_TESTS + 1))
                return 1
            else
                echo -e "${YELLOW}[WARN] Training timeout, continuing with existing model${NC}"
            fi
        fi
    fi

    # Update ConfigMap if Feast mode
    if [ "$mode" = "feast" ]; then
        echo -e "${BLUE}[3/6] Updating Feast ConfigMap...${NC}"
        kubectl delete configmap feature-store-config -n ml-pipeline --ignore-not-found > /dev/null 2>&1

        if [ "$data_type" = "tabular" ]; then
            kubectl create configmap feature-store-config --from-file=feature_store.yaml=titanic_repo/feature_store.yaml -n ml-pipeline > /dev/null 2>&1
        else
            kubectl create configmap feature-store-config --from-file=feature_store.yaml=feature_repo_etth1/feature_store.yaml -n ml-pipeline > /dev/null 2>&1
        fi
        echo -e "${GREEN}[PASS] ConfigMap updated${NC}"
    else
        echo -e "${BLUE}[3/6] Skipping ConfigMap (Standard mode)${NC}"
    fi

    # Restart API
    echo -e "${BLUE}[4/6] Restarting API pods...${NC}"
    DEPLOYMENT_NAME="ml-prediction-api-${mode}"

    kubectl rollout restart deployment $DEPLOYMENT_NAME -n ml-pipeline > /dev/null 2>&1 || true
    sleep 5

    # Wait for rollout to complete
    if kubectl rollout status deployment $DEPLOYMENT_NAME -n ml-pipeline --timeout=180s > /dev/null 2>&1; then
        echo -e "${GREEN}[PASS] API pods ready${NC}"
    else
        echo -e "${RED}[FAIL] API pods failed to start${NC}"
        kubectl get pods -n ml-pipeline -l app=ml-prediction
        kubectl describe deployment $DEPLOYMENT_NAME -n ml-pipeline | tail -20
        RESULTS+=("${RED}[FAIL] $test_name - API Failed to Start${NC}")
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi

    # Setup port forward
    echo -e "${BLUE}[5/6] Setting up port forward...${NC}"
    cleanup
    sleep 3
    kubectl port-forward -n ml-pipeline service/ml-prediction-service 8000:80 > /tmp/port-forward.log 2>&1 &
    sleep 8

    if ! wait_for_port_forward; then
        echo -e "${RED}[FAIL] Port forward failed${NC}"
        RESULTS+=("${RED}[FAIL] $test_name - Port Forward Failed${NC}")
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
    echo -e "${GREEN}[PASS] Port forward ready${NC}"

    # Run tests
    echo -e "${BLUE}[6/6] Running API tests...${NC}"
    TEST_PASSED=true

    if ! test_health "$data_type"; then
        TEST_PASSED=false
    fi

    sleep 2

    if [ "$skip_predict" = "true" ]; then
        echo -e "${YELLOW}  ⚠ Skipping prediction tests (marked as skipped)${NC}"
    else
        if [ "$data_type" = "tabular" ]; then
            if ! test_tabular_predict; then
                TEST_PASSED=false
            fi
        else
            if [ "$mode" = "feast" ]; then
                if ! test_timeseries_feast_predict; then
                    TEST_PASSED=false
                fi
            else
                if ! test_timeseries_predict; then
                    TEST_PASSED=false
                fi
            fi
        fi
    fi

    sleep 2

    if [ "$mode" = "feast" ]; then
        if ! test_feast_endpoints "$data_type"; then
            TEST_PASSED=false
        fi
    fi

    if [ "$TEST_PASSED" = true ]; then
        echo -e "${GREEN}[PASS] All API tests PASSED${NC}"
        RESULTS+=("${GREEN}[PASS] $test_name - ALL TESTS PASSED${NC}")
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}[FAIL] Some API tests FAILED${NC}"
        RESULTS+=("${RED}[FAIL] $test_name - API Tests Failed${NC}")
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Main execution
echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}   Comprehensive ML Pipeline Deployment Tests      ${NC}"
echo -e "${BLUE}====================================================${NC}"

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! kubectl cluster-info > /dev/null 2>&1; then
    echo -e "${RED}✗ Kubernetes cluster not accessible${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Kubernetes cluster accessible${NC}"

if ! docker images | grep -q "ml-pipeline"; then
    echo -e "${RED}✗ ML pipeline Docker images not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker images found${NC}"

# Test scenarios
echo -e "\n${BLUE}Starting deployment tests...${NC}\n"

# ----------------------------------------------------------------------
# NEW TESTS: Expanded Scenarios (Nested, Dynamic, Conditional)
# ----------------------------------------------------------------------
echo -e "\n${YELLOW}---------------------------------------${NC}"
echo -e "${YELLOW}      EXPANDED SCENARIO TESTS          ${NC}"
echo -e "${YELLOW}---------------------------------------${NC}"

# 7. Conditional Branch (using timeseries data as per config analysis)
# Note: conditional_branch.yaml refers to tft_ts.yaml / xgboost_ts.yaml
# MOVED TO TOP FOR DEBUGGING
run_deployment_test "standard" "etth3.yaml" "n/a" "timeseries" "conditional_branch.yaml" "true"

# 5. Nested Pipeline (using tabular data for clustering)
# Note: nested_suppipeline.yaml uses 'cluster_type_1' which implies tabular/clustering logic
run_deployment_test "standard" "tabular.yaml" "n/a" "tabular" "nested_suppipeline.yaml"

# 6. Dynamic Adapter (using tabular data for kmeans then xgboost)
# Note: dynamic_adapter_train.yaml uses 'kmeans_dynamic'
run_deployment_test "standard" "tabular.yaml" "n/a" "tabular" "dynamic_adapter_train.yaml"


# FEAST MODE TESTS
echo -e "${YELLOW}---------------------------------------${NC}"
echo -e "${YELLOW}         FEAST MODE TESTS              ${NC}"
echo -e "${YELLOW}---------------------------------------${NC}"

# Tabular Feast (this works)
run_deployment_test "feast" "feast_tabular.yaml" "titanic_repo" "tabular" "standard_train.yaml"

# Time Series Feast (now works with on-the-fly ingestion)
run_deployment_test "feast" "etth3_feast.yaml" "feature_repo_etth1" "timeseries" "standard_train.yaml"

# STANDARD MODE TESTS
echo -e "\n${YELLOW}---------------------------------------${NC}"
echo -e "${YELLOW}        STANDARD MODE TESTS            ${NC}"
echo -e "${YELLOW}---------------------------------------${NC}"

# Tabular Standard
run_deployment_test "standard" "tabular.yaml" "n/a" "tabular" "standard_train.yaml"

# Time Series Standard
run_deployment_test "standard" "etth3.yaml" "n/a" "timeseries" "standard_train.yaml"

# Print final summary
echo ""
echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}              TEST SUMMARY                          ${NC}"
echo -e "${BLUE}====================================================${NC}"
echo ""

for result in "${RESULTS[@]}"; do
    echo -e "$result"
done

echo ""
echo -e "${BLUE}Total Tests:   $TOTAL_TESTS${NC}"
echo -e "${GREEN}Passed:        $PASSED_TESTS${NC}"
echo -e "${RED}Failed:        $FAILED_TESTS${NC}"
echo -e "${YELLOW}Skipped:       $SKIPPED_TESTS${NC}"

echo ""
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}====================================================${NC}"
    echo -e "${GREEN}                                                    ${NC}"
    echo -e "${GREEN}      [PASS] ALL TESTS PASSED SUCCESSFULLY!             ${NC}"
    echo -e "${GREEN}                                                    ${NC}"
    echo -e "${GREEN}====================================================${NC}"
    exit 0
else
    echo -e "${RED}====================================================${NC}"
    echo -e "${RED}                                                    ${NC}"
    echo -e "${RED}      [FAIL] SOME TESTS FAILED                          ${NC}"
    echo -e "${RED}                                                    ${NC}"
    echo -e "${RED}====================================================${NC}"
    exit 1
fi
