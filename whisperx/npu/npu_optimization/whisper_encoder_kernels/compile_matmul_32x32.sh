#!/bin/bash
#
# Compile 32x32 Matrix Multiply Kernel
# Scaled up from 16x16 for better performance
#

set -e  # Exit on error

echo "=================================================="
echo "Compiling 32x32 Matmul Kernel"
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
mkdir -p build_matmul_32x32
cd build_matmul_32x32

echo "Step 1: Compiling C kernel to object file..."
$PEANO/clang \
    --target=aie2 \
    -I$PEANO/../aie_kernels/aie2/include \
    -c ../matmul_int8_32x32.c \
    -o matmul_32x32.o
echo "✅ C kernel compiled: matmul_32x32.o"
echo

echo "Step 2: Compiling MLIR to XCLBIN using aiecc.py (Peano-only mode)..."
# Try Peano-only mode (same as working matmul_fixed)
/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --sysroot=$PEANO/../sysroot \
    --host-target=x86_64-amd-linux-gnu \
    ../matmul_32x32.mlir \
    -I$PEANO/../aie_kernels/aie2/include \
    -o matmul_32x32.xclbin \
    --xclbin-kernel-name=MLIR_AIE \
    --peano-install-dir=$PEANO \
    --no-xchesscc \
    --no-xbridge \
    --unified
echo "✅ XCLBIN generated: matmul_32x32.xclbin"
echo

# Check file sizes
echo "Build artifacts:"
ls -lh matmul_32x32.o matmul_32x32.xclbin main_sequence.bin
echo

echo "=================================================="
echo "✅ 32x32 Matmul Kernel Compiled Successfully!"
echo "=================================================="
echo
echo "Next step: Run test with 32x32 kernel"
echo "  python3 test_matmul_32x32.py"
