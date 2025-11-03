#!/bin/bash
#
# Compile multi-core attention kernel using IRON-generated MLIR
#

set -e

KERNEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$KERNEL_DIR/build_attention_iron"
PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
MLIR_AIE_PYTHON=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie

# Setup environment
export PEANO_INSTALL_DIR
export PYTHONPATH="${MLIR_AIE_PYTHON}:$PYTHONPATH"
export PATH=/opt/xilinx/xrt/bin:${PEANO_INSTALL_DIR}/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "=== Multi-Core Attention Kernel Compilation (IRON) ==="
echo "Build directory: $BUILD_DIR"

# Step 1: Use existing C kernel object file
echo ""
echo "Step 1/3: Using pre-compiled C kernel..."
if [ ! -f "attention_int8_64x64_tiled.o" ]; then
    # Copy from build_attention_64x64
    if [ -f "$KERNEL_DIR/build_attention_64x64/attention_int8_64x64.o" ]; then
        cp "$KERNEL_DIR/build_attention_64x64/attention_int8_64x64.o" attention_int8_64x64_tiled.o
        echo "✓ Copied existing kernel object file"
    else
        echo "✗ Error: No pre-compiled kernel found!"
        echo "  Please compile attention_int8_64x64.c first"
        exit 1
    fi
else
    echo "✓ Using existing attention_int8_64x64_tiled.o"
fi

# Step 2: Use aiecc.py to compile MLIR → XCLBIN
echo ""
echo "Step 2/3: Compiling MLIR to XCLBIN using aiecc.py..."

# Copy MLIR to build dir
cp "$KERNEL_DIR/attention_iron_generated.mlir" .

/home/ucadmin/.local/bin/aiecc.py \
    --sysroot=${PEANO_INSTALL_DIR}/../sysroot \
    --host-target=x86_64-amd-linux-gnu \
    "$KERNEL_DIR/attention_iron_generated.mlir" \
    -o attention_multicore.xclbin \
    --xclbin-kernel-name=MLIR_AIE \
    --peano=${PEANO_INSTALL_DIR}

if [ -f "attention_multicore.xclbin" ]; then
    echo "✓ XCLBIN generated: attention_multicore.xclbin"
    ls -lh attention_multicore.xclbin
else
    echo "✗ XCLBIN generation failed!"
    exit 1
fi

# Step 3: Verify XCLBIN
echo ""
echo "Step 3/3: Verifying XCLBIN..."
if command -v xclbinutil &> /dev/null; then
    xclbinutil --info --input attention_multicore.xclbin
else
    echo "⚠ xclbinutil not found, skipping verification"
fi

echo ""
echo "=== Compilation Complete ==="
echo "✓ XCLBIN: $BUILD_DIR/attention_multicore.xclbin"
echo "✓ Ready for testing with 4-column parallel execution"
echo ""
echo "Next steps:"
echo "  1. Create test script for multi-tile processing"
echo "  2. Benchmark 4× throughput improvement"
echo "  3. Measure realtime factor (target: 27-33×)"
