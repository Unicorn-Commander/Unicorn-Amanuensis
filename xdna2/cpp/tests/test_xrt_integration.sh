#!/bin/bash
#
# XRT NPU Integration Test Runner
# =================================
#
# Runs the C++ XRT integration test and validates NPU performance.
#
# Expected Results:
# - Accuracy: 100% exact match
# - Speedup: 1211.3x (same as Python)
# - Latency: <0.3ms (vs Python's 2ms overhead)
#

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}XRT NPU Integration Test${NC}"
echo "========================================"
echo

# Check if test binary exists
TEST_BINARY="./build/tests/test_xrt_npu_integration"
if [ ! -f "$TEST_BINARY" ]; then
    echo -e "${RED}ERROR: Test binary not found: $TEST_BINARY${NC}"
    echo "Please build the project first:"
    echo "  cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp"
    echo "  mkdir -p build && cd build"
    echo "  cmake .."
    echo "  make test_xrt_npu_integration"
    exit 1
fi

# Check if kernel files exist
KERNEL_DIR="$HOME/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
XCLBIN_PATH="$KERNEL_DIR/final_512x512x512_64x64x64_8c.xclbin"
INSTS_PATH="$KERNEL_DIR/insts_512x512x512_64x64x64_8c.txt"

if [ ! -f "$XCLBIN_PATH" ]; then
    echo -e "${RED}ERROR: Kernel not found: $XCLBIN_PATH${NC}"
    echo "Please build the kernel first:"
    echo "  cd $KERNEL_DIR/.."
    echo "  make"
    exit 1
fi

if [ ! -f "$INSTS_PATH" ]; then
    echo -e "${RED}ERROR: Instructions not found: $INSTS_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Test binary found"
echo -e "${GREEN}✓${NC} Kernel files found"
echo

# Check XRT
if [ ! -d "/opt/xilinx/xrt" ]; then
    echo -e "${RED}ERROR: XRT not found at /opt/xilinx/xrt${NC}"
    echo "Please install XRT first."
    exit 1
fi

echo -e "${GREEN}✓${NC} XRT installation found"
echo

# Set Python path for XRT
export PYTHONPATH="/opt/xilinx/xrt/python:$PYTHONPATH"

# Run test
echo -e "${BLUE}Running test...${NC}"
echo

if $TEST_BINARY; then
    echo
    echo -e "${GREEN}═══════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ TEST PASSED${NC}"
    echo -e "${GREEN}═══════════════════════════════════════${NC}"
    exit 0
else
    echo
    echo -e "${RED}═══════════════════════════════════════${NC}"
    echo -e "${RED}❌ TEST FAILED${NC}"
    echo -e "${RED}═══════════════════════════════════════${NC}"
    exit 1
fi
