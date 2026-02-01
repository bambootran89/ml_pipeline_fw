#!/bin/bash
# Quick launcher for serve APIs

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Serve API Quick Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Default values
FRAMEWORK="fastapi"
PORT=8000
HOST="0.0.0.0"
EXPERIMENT_CONFIG=""
ALIAS="production"

# Parse arguments
SERVE_CONFIG=""

show_help() {
    echo "Usage: ./serve_api.sh [OPTIONS] <serve_config.yaml>"
    echo ""
    echo "Options:"
    echo "  -e, --experiment EXPERIMENT Experiment config (e.g., etth1.yaml)"
    echo "  -f, --framework FRAMEWORK   Framework: fastapi or ray (default: fastapi)"
    echo "  -p, --port PORT            Port number (default: 8000)"
    echo "  -h, --host HOST            Host address (default: 0.0.0.0)"
    echo "  -a, --alias ALIAS          MLflow alias (default: production)"
    echo "  --help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  # FastAPI on port 8000 with latest alias"
    echo "  ./serve_api.sh -e mlproject/configs/experiments/etth1.yaml -a latest mlproject/configs/generated/standard_train_serve.yaml"
    echo ""
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--experiment)
            EXPERIMENT_CONFIG="$2"
            shift 2
            ;;
        -f|--framework)
            FRAMEWORK="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -a|--alias)
            ALIAS="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        -*)
            echo "Error: Unknown option $1"
            show_help
            ;;
        *)
            SERVE_CONFIG="$1"
            shift
            ;;
    esac
done

# Check if serve config provided
if [ -z "$SERVE_CONFIG" ]; then
    echo -e "${YELLOW}Error: Serve config not provided${NC}"
    echo ""
    show_help
fi

# Check if experiment config provided
if [ -z "$EXPERIMENT_CONFIG" ]; then
    echo -e "${YELLOW}Error: Experiment config not provided${NC}"
    echo -e "${YELLOW}Use -e to specify experiment config (e.g., -e mlproject/configs/experiments/etth1.yaml)${NC}"
    echo ""
    show_help
fi



if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo -e "${YELLOW}Error: Experiment config not found: $EXPERIMENT_CONFIG${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Experiment config: $EXPERIMENT_CONFIG"
echo "  Serve config: $SERVE_CONFIG"
echo "  Framework: $FRAMEWORK"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Alias: $ALIAS"
echo ""

# Auto-generate serve config if missing
if [ ! -f "$SERVE_CONFIG" ]; then
    echo -e "${YELLOW}Serve config not found. Generating from experiment...${NC}"
    # Infer pipeline config from serve config name (standard_train_serve.yaml -> standard_train.yaml)
    # This is a heuristic; properly we should pass the pipeline config as an arg, but for now we infer or use defaults
    # Re-using the standard training dag_run generate command

    # We need the training config path to generate.
    # Assumption: The input serve config path follows pattern configs/generated/<pipeline>_serve.yaml
    # We try to reverse engineer the pipeline config path

    BASE_NAME=$(basename "$SERVE_CONFIG" _serve.yaml)
    PIPELINE_CONFIG="mlproject/configs/pipelines/${BASE_NAME}.yaml"

    if [ -f "$PIPELINE_CONFIG" ]; then
         echo "Detected pipeline config: $PIPELINE_CONFIG"
         python -m mlproject.src.pipeline.dag_run generate \
            -t "$PIPELINE_CONFIG" \
            -e "$EXPERIMENT_CONFIG" \
            --type serve \
            --alias "$ALIAS"

         if [ -f "$SERVE_CONFIG" ]; then
             echo -e "${GREEN}Successfully generated $SERVE_CONFIG${NC}"
         else
             echo -e "${RED}Failed to generate $SERVE_CONFIG${NC}"
             exit 1
         fi
    else
         echo -e "${RED}Could not infer pipeline config path from $SERVE_CONFIG. Cannot auto-generate.${NC}"
         exit 1
    fi
fi

# Run the Python command
echo -e "${BLUE}Starting server...${NC}"
echo ""

python -m mlproject.serve.run_generated_api \
    --serve-config "$SERVE_CONFIG" \
    --experiment-config "$EXPERIMENT_CONFIG" \
    --framework "$FRAMEWORK" \
    --host "$HOST" \
    --port "$PORT" \
    --alias "$ALIAS"
