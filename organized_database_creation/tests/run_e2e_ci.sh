#!/bin/bash
# Continuous Integration Script for E2E Testing
# Runs curator → embeddings → RAG retrieval pipeline test

set -e  # Exit on error

echo "========================================"
echo "CI: End-to-End Pipeline Test"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_OUTPUT="$PROJECT_ROOT/test_output_e2e"

echo -e "${YELLOW}Project root: $PROJECT_ROOT${NC}"
echo -e "${YELLOW}Test output: $TEST_OUTPUT${NC}"

# Step 1: Check Python environment
echo ""
echo "Step 1: Checking Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Python found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python3 not found${NC}"
    exit 1
fi

# Step 2: Check required dependencies
echo ""
echo "Step 2: Checking dependencies..."
REQUIRED_PACKAGES=("requests" "beautifulsoup4" "sentence-transformers" "faiss-cpu" "numpy")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import ${package//-/_}" 2>/dev/null; then
        echo -e "${GREEN}✓ $package${NC}"
    else
        echo -e "${YELLOW}! $package (missing)${NC}"
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing missing packages...${NC}"
    pip3 install "${MISSING_PACKAGES[@]}" || {
        echo -e "${RED}✗ Failed to install dependencies${NC}"
        exit 1
    }
fi

# Step 3: Clean previous test output
echo ""
echo "Step 3: Cleaning previous test output..."
if [ -d "$TEST_OUTPUT" ]; then
    rm -rf "$TEST_OUTPUT"
    echo -e "${GREEN}✓ Cleaned test output directory${NC}"
else
    echo -e "${GREEN}✓ No previous test output${NC}"
fi

# Step 4: Run E2E test
echo ""
echo "Step 4: Running end-to-end pipeline test..."
echo "========================================"

cd "$PROJECT_ROOT/tests"
if python3 test_e2e_pipeline.py; then
    echo ""
    echo -e "${GREEN}✓ E2E pipeline test PASSED${NC}"
    TEST_RESULT=0
else
    echo ""
    echo -e "${RED}✗ E2E pipeline test FAILED${NC}"
    TEST_RESULT=1
fi

# Step 5: Check test results
echo ""
echo "Step 5: Analyzing test results..."
RESULTS_FILE="$TEST_OUTPUT/e2e_test_results.json"

if [ -f "$RESULTS_FILE" ]; then
    echo -e "${GREEN}✓ Test results file found${NC}"
    echo ""
    echo "Test Results Summary:"
    echo "--------------------"
    cat "$RESULTS_FILE" | python3 -m json.tool || cat "$RESULTS_FILE"
    echo ""
else
    echo -e "${RED}✗ Test results file not found${NC}"
    TEST_RESULT=1
fi

# Step 6: Verify outputs
echo ""
echo "Step 6: Verifying test outputs..."
TEST_JSONL="$TEST_OUTPUT/test_data.jsonl"
TEST_INDEX="$TEST_OUTPUT/embeddings/faiss_index.bin"
TEST_METADATA="$TEST_OUTPUT/embeddings/metadata.json"

if [ -f "$TEST_JSONL" ]; then
    JSONL_LINES=$(wc -l < "$TEST_JSONL")
    echo -e "${GREEN}✓ JSONL file: $JSONL_LINES entries${NC}"
else
    echo -e "${RED}✗ JSONL file not found${NC}"
    TEST_RESULT=1
fi

if [ -f "$TEST_INDEX" ]; then
    INDEX_SIZE=$(du -h "$TEST_INDEX" | cut -f1)
    echo -e "${GREEN}✓ FAISS index: $INDEX_SIZE${NC}"
else
    echo -e "${RED}✗ FAISS index not found${NC}"
    TEST_RESULT=1
fi

if [ -f "$TEST_METADATA" ]; then
    METADATA_SIZE=$(du -h "$TEST_METADATA" | cut -f1)
    echo -e "${GREEN}✓ Metadata file: $METADATA_SIZE${NC}"
else
    echo -e "${RED}✗ Metadata file not found${NC}"
    TEST_RESULT=1
fi

# Final summary
echo ""
echo "========================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓✓✓ CI TEST PASSED ✓✓✓${NC}"
    echo "All pipeline stages completed successfully"
else
    echo -e "${RED}✗✗✗ CI TEST FAILED ✗✗✗${NC}"
    echo "One or more pipeline stages failed"
fi
echo "========================================"

exit $TEST_RESULT
