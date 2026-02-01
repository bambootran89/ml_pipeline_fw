#!/bin/bash
# Build script for separated Docker images (training vs serving)

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Default values
IMAGE_TYPE="serve"  # serve or train
TAG="latest"
STAGE=""
PLATFORM=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE_TYPE="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -s|--stage)
            STAGE="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="--platform $2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -i, --image TYPE      Image type: serve|train (default: serve)"
            echo "  -t, --tag TAG         Docker image tag (default: latest)"
            echo "  -s, --stage STAGE     Build stage (train: training|training-dev, serve: serving)"
            echo "  -p, --platform ARCH   Target platform (e.g., linux/amd64, linux/arm64)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Build serving image (for customers)"
            echo "  $0 -i serve"
            echo ""
            echo "  # Build training image (in-house)"
            echo "  $0 -i train"
            echo ""
            echo "  # Build training dev environment"
            echo "  $0 -i train -s training-dev"
            echo ""
            echo "  # Build for specific platform"
            echo "  $0 -i serve -p linux/amd64"
            echo ""
            echo "  # Build with custom tag"
            echo "  $0 -i serve -t v1.0.0"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Set defaults based on image type
if [ "$IMAGE_TYPE" = "train" ]; then
    DOCKERFILE="Dockerfile.train"
    IMAGE_NAME="ml-pipeline-train"
    if [ -z "$STAGE" ]; then
        STAGE="training"
    fi
elif [ "$IMAGE_TYPE" = "serve" ]; then
    DOCKERFILE="Dockerfile.serve"
    IMAGE_NAME="ml-pipeline-serve"
    if [ -z "$STAGE" ]; then
        STAGE="runtime"
    fi
else
    echo -e "${RED}Error: Invalid image type '$IMAGE_TYPE'${NC}"
    echo "Must be 'serve' or 'train'"
    exit 1
fi

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Building Docker Image${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "Type:  ${GREEN}${IMAGE_TYPE}${NC} (${DOCKERFILE})"
echo -e "Image: ${GREEN}${IMAGE_NAME}:${TAG}${NC}"
echo -e "Stage: ${GREEN}${STAGE}${NC}"
if [ -n "$PLATFORM" ]; then
    echo -e "Platform: ${GREEN}${PLATFORM#--platform }${NC}"
fi
echo ""

# Build the image
echo -e "${YELLOW}Building image...${NC}"
docker build \
    $PLATFORM \
    --target $STAGE \
    -t ${IMAGE_NAME}:${TAG} \
    -t ${IMAGE_NAME}:${STAGE}-${TAG} \
    -f ${DOCKERFILE} \
    .

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Build Complete${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "Images created:"
echo -e "  - ${IMAGE_NAME}:${TAG}"
echo -e "  - ${IMAGE_NAME}:${STAGE}-${TAG}"
echo ""

# Show image size
echo -e "${BLUE}Image details:${NC}"
docker images ${IMAGE_NAME}:${TAG} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
echo ""

# Suggest next steps based on image type
if [ "$IMAGE_TYPE" = "serve" ]; then
    echo -e "${YELLOW}Next steps (SERVING):${NC}"
    echo "  Test locally:"
    echo "    docker-compose up -d api mlflow"
    echo "    curl http://localhost:8000/health"
    echo ""
    echo "  Deploy to customer:"
    echo "    docker tag ${IMAGE_NAME}:${TAG} registry.customer.com/${IMAGE_NAME}:${TAG}"
    echo "    docker push registry.customer.com/${IMAGE_NAME}:${TAG}"
else
    echo -e "${YELLOW}Next steps (TRAINING):${NC}"
    echo "  Run training:"
    echo "    docker-compose --profile train run --rm train"
    echo ""
    echo "  Run evaluation:"
    echo "    docker-compose --profile eval run --rm evaluate"
    echo ""
    echo "  Interactive shell:"
    echo "    docker-compose --profile dev run --rm train-dev"
fi
echo ""

# Show comparison
echo -e "${BLUE}Image sizes comparison:${NC}"
echo "Expected sizes:"
if [ "$IMAGE_TYPE" = "serve" ]; then
    echo "  - Serving: ~600-800MB (minimal)"
else
    echo "  - Training: ~1.5-2GB (full environment)"
fi
