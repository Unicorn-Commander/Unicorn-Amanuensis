#!/bin/bash
#
# Compile Fixed Matrix Multiply Kernel
# Uses packed input buffer to fix zero-output issue
#

set -e  # Exit on error

echo "=================================================="
echo "Compiling Fixed Matmul Kernel"
echo "=================================================="
echo

# Find Peano compiler
if [ -n "$PEANO_INSTALL_DIR" ]; then
    PEANO=$PEANO_INSTALL_DIR/bin
elif [ -d "/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie" ]; then
    PEANO=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie/bin
elif [ -d "/home/ucadmin/mlir-aie-source/ironenv/lib/python3.13/site-packages/llvm-aie" ]; then
    PEANO=/home/ucadmin/mlir-aie-source/ironenv/lib/python3.13/site-packages/llvm-aie/bin
else
    echo "❌ ERROR: Peano compiler not found!"
    echo "   Please set PEANO_INSTALL_DIR environment variable"
    exit 1
fi

echo "✅ Found Peano compiler: $PEANO"
echo

# Create build directory
mkdir -p build_matmul_fixed
cd build_matmul_fixed

echo "Step 1: Compiling C kernel to object file..."
$PEANO/clang \
    --target=aie2 \
    -I$PEANO/../aie_kernels/aie2/include \
    -c ../matmul_int8.c \
    -o matmul_fixed.o
echo "✅ C kernel compiled: matmul_fixed.o"
echo

echo "Step 2: Compiling MLIR to XCLBIN using aiecc.py..."
/home/ucadmin/.local/bin/aiecc.py \
    --sysroot=$PEANO/../sysroot \
    --host-target=x86_64-amd-linux-gnu \
    ../matmul_fixed.mlir \
    -I$PEANO/../aie_kernels/aie2/include \
    -o matmul_fixed.xclbin \
    --xclbin-kernel-name=MLIR_AIE \
    --peano-install-dir=$PEANO
echo "✅ XCLBIN generated: matmul_fixed.xclbin"
echo

# Check file sizes
echo "Build artifacts:"
ls -lh matmul_fixed.o matmul_fixed.xclbin matmul_lowered.mlir
echo

echo "=================================================="
echo "✅ Fixed Matmul Kernel Compiled Successfully!"
echo "=================================================="
echo
echo "Next step: Run test with fixed kernel"
echo "  python3 test_matmul_fixed.py"
