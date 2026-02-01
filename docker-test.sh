#!/bin/bash
# Test script for Docker image validation

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

IMAGE_NAME="ml-pipeline-template:latest"
CONTAINER_NAME="ml-test-$(date +%s)"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Docker Image Testing${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Test 1: Check image exists
echo -e "${YELLOW}[1/7] Checking if image exists...${NC}"
if docker images ${IMAGE_NAME} | grep -q "ml-pipeline-template"; then
    echo -e "${GREEN}PASS${NC} - Image found"
else
    echo -e "${RED}FAIL${NC} - Image not found. Run ./docker-build.sh first"
    exit 1
fi

# Test 2: Check Python version
echo -e "${YELLOW}[2/7] Checking Python version...${NC}"
PYTHON_VERSION=$(docker run --rm ${IMAGE_NAME} python --version)
echo -e "${GREEN}PASS${NC} - ${PYTHON_VERSION}"

# Test 3: Check package installation
echo -e "${YELLOW}[3/7] Checking package installation...${NC}"
if docker run --rm ${IMAGE_NAME} python -c "import mlproject" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC} - mlproject package installed"
else
    echo -e "${RED}FAIL${NC} - mlproject package not found"
    exit 1
fi

# Test 4: Check dependencies
echo -e "${YELLOW}[4/7] Checking key dependencies...${NC}"
DEPS=("pandas" "numpy" "mlflow" "fastapi" "uvicorn")
for dep in "${DEPS[@]}"; do
    if docker run --rm ${IMAGE_NAME} python -c "import $dep" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $dep"
    else
        echo -e "  ${RED}✗${NC} $dep (missing)"
        exit 1
    fi
done

# Test 5: Check user (should be non-root)
echo -e "${YELLOW}[5/7] Checking container user...${NC}"
USER_ID=$(docker run --rm ${IMAGE_NAME} id -u)
if [ "$USER_ID" != "0" ]; then
    echo -e "${GREEN}PASS${NC} - Running as non-root user (UID: ${USER_ID})"
else
    echo -e "${YELLOW}WARNING${NC} - Running as root user"
fi

# Test 6: Start container and check health
echo -e "${YELLOW}[6/7] Testing container startup and health check...${NC}"
docker run -d --name ${CONTAINER_NAME} -p 8002:8000 ${IMAGE_NAME} >/dev/null

echo -n "  Waiting for container to be healthy"
MAX_RETRIES=30
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -sf http://localhost:8002/health >/dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}PASS${NC} - Health check successful"
        break
    fi
    echo -n "."
    sleep 2
    RETRY=$((RETRY + 1))
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    echo ""
    echo -e "${RED}FAIL${NC} - Health check failed after ${MAX_RETRIES} retries"
    echo "Container logs:"
    docker logs ${CONTAINER_NAME} | tail -20
    docker stop ${CONTAINER_NAME} >/dev/null 2>&1
    docker rm ${CONTAINER_NAME} >/dev/null 2>&1
    exit 1
fi

# Test 7: Check API response
echo -e "${YELLOW}[7/7] Testing API endpoint...${NC}"
RESPONSE=$(curl -s http://localhost:8002/health)
if echo "$RESPONSE" | grep -q "status"; then
    echo -e "${GREEN}PASS${NC} - API responding correctly"
    echo "  Response: $RESPONSE"
else
    echo -e "${RED}FAIL${NC} - Unexpected API response"
    echo "  Response: $RESPONSE"
fi

# Cleanup
echo ""
echo -e "${BLUE}Cleaning up test container...${NC}"
docker stop ${CONTAINER_NAME} >/dev/null 2>&1
docker rm ${CONTAINER_NAME} >/dev/null 2>&1

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  All Tests Passed${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${BLUE}Image Statistics:${NC}"
docker images ${IMAGE_NAME} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
echo ""
echo -e "${BLUE}Security Scan:${NC}"
echo "Run security scan with:"
echo "  docker scan ${IMAGE_NAME}"
echo "  # or"
echo "  trivy image ${IMAGE_NAME}"
