#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Setup environment
export PYTHONPATH=.
if [ -d "mlproject_env" ]; then
    source mlproject_env/bin/activate
fi

CONFIG_FILE="mlproject/configs/experiments/etth1.yaml"
# Use actual data file as input for serving test
INPUT_CSV="mlproject/data/ETTh1.csv"

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}   VERIFYING LEGACY COMPATIBILITY (v1.run)${NC}"
echo -e "${BLUE}=======================================================${NC}"

# 1. Run Legacy TRAIN
echo -e "${BLUE}--- [1/3] Running Legacy TRAIN ---${NC}"
CMD="python -m mlproject.src.pipeline.compat.v1.run train --config $CONFIG_FILE"
echo "Executing: $CMD"
if $CMD > legacy_train.log 2>&1; then
    echo -e "${GREEN}Legacy TRAIN PASS${NC}"
else
    echo -e "${RED}Legacy TRAIN FAILED${NC}"
    tail -n 20 legacy_train.log
    exit 1
fi

# 2. Run Legacy EVAL
echo -e "${BLUE}--- [2/3] Running Legacy EVAL ---${NC}"
CMD="python -m mlproject.src.pipeline.compat.v1.run eval --config $CONFIG_FILE --alias latest"
echo "Executing: $CMD"
if $CMD > legacy_eval.log 2>&1; then
    echo -e "${GREEN}Legacy EVAL PASS${NC}"
else
    echo -e "${RED}Legacy EVAL FAILED${NC}"
    tail -n 20 legacy_eval.log
    exit 1
fi

# 3. Run Legacy SERVE
echo -e "${BLUE}--- [3/3] Running Legacy SERVE ---${NC}"
CMD="python -m mlproject.src.pipeline.compat.v1.run serve --config $CONFIG_FILE --input $INPUT_CSV --alias latest"
echo "Executing: $CMD"
if $CMD > legacy_serve.log 2>&1; then
    echo -e "${GREEN}Legacy SERVE PASS${NC}"
else
    echo -e "${RED}Legacy SERVE FAILED${NC}"
    tail -n 20 legacy_serve.log
    exit 1
fi

echo -e "${GREEN}=======================================================${NC}"
echo -e "${GREEN}   LEGACY COMPATIBILITY VERIFIED${NC}"
echo -e "${GREEN}=======================================================${NC}"
