#!/bin/bash
# Docker Build and Test Script for MONAI Aegis

set -e  # Exit on error

echo "=== MONAI Aegis Docker Test Script ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="monai_aegis"
IMAGE_TAG="test"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Step 1: Build Docker Image
echo -e "${YELLOW}Step 1: Building Docker image...${NC}"
docker build -t ${FULL_IMAGE} . || {
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ Docker image built successfully${NC}"
echo ""

# Step 2: Run Unit Tests
echo -e "${YELLOW}Step 2: Running unit tests...${NC}"
docker run --rm ${FULL_IMAGE} \
    python -m unittest discover /app/tests/unit -v || {
    echo -e "${RED}✗ Unit tests failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ Unit tests passed${NC}"
echo ""

# Step 3: Run Integration Tests
echo -e "${YELLOW}Step 3: Running integration tests...${NC}"
docker run --rm ${FULL_IMAGE} \
    python -m unittest discover /app/tests/integration -v || {
    echo -e "${RED}✗ Integration tests failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ Integration tests passed${NC}"
echo ""

# Step 4: Run Pipeline on Test Data (if present)
if [ -d "staging_input" ] && [ "$(ls -A staging_input)" ]; then
    echo -e "${YELLOW}Step 4: Running pipeline on staging_input/...${NC}"
    
    # Create output directory if it doesn't exist
    mkdir -p staging_output
    
    docker run --rm \
        -v "$(pwd)/staging_input:/app/staging_input:ro" \
        -v "$(pwd)/staging_output:/app/staging_output" \
        ${FULL_IMAGE} || {
        echo -e "${RED}✗ Pipeline execution failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Pipeline executed successfully${NC}"
    echo -e "${GREEN}  Output files in: staging_output/${NC}"
else
    echo -e "${YELLOW}Step 4: Skipped (no files in staging_input/)${NC}"
fi
echo ""

# Summary
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}✓ All Docker tests passed!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo "Docker image: ${FULL_IMAGE}"
echo "To run manually:"
echo "  docker run --rm \\"
echo "    -v \"\$(pwd)/staging_input:/app/staging_input:ro\" \\"
echo "    -v \"\$(pwd)/staging_output:/app/staging_output\" \\"
echo "    ${FULL_IMAGE}"
