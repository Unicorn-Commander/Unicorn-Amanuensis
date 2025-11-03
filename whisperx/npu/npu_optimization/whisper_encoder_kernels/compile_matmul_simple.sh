#!/bin/bash
#
# Compile Fixed Matrix Multiply Kernel using aiecc.py
# Simple one-step compilation
#

set -e  # Exit on error

echo "=================================================="
echo "Compiling Fixed Matmul Kernel (Simple Method)"
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

export PEANO_INSTALL_DIR=$(dirname $PEANO)
echo "✅ Found Peano: $PEANO_INSTALL_DIR"
echo

# Create build directory
mkdir -p build_matmul_fixed
cd build_matmul_fixed

echo "Compiling MLIR + C kernel to XCLBIN using aiecc.py..."
echo "This may take 30-60 seconds..."
echo

timeout 120 /home/ucadmin/.local/bin/aiecc.py \
    ../matmul_fixed.mlir \
    -I$PEANO_INSTALL_DIR/aie_kernels/aie2/include \
    --sysroot=$PEANO_INSTALL_DIR/sysroot \
    --host-target=x86_64-amd-linux-gnu \
    --peano-install-dir=$PEANO_INSTALL_DIR \
    --xclbin-kernel-name=MLIR_AIE

echo
echo "✅ Compilation complete!"
echo

# Find generated files
if [ -f "matmul_fixed.xclbin" ]; then
    echo "✅ XCLBIN generated: matmul_fixed.xclbin"
    ls -lh matmul_fixed.xclbin
elif [ -f "final.xclbin" ]; then
    echo "✅ XCLBIN generated: final.xclbin"
    mv final.xclbin matmul_fixed.xclbin
    ls -lh matmul_fixed.xclbin
else
    echo "Looking for generated files..."
    ls -lh *.xclbin 2>/dev/null || echo "No XCLBIN found"
fi

if [ -f "insts.bin" ]; then
    echo "✅ Instructions generated: insts.bin"
    ls -lh insts.bin
fi

echo
echo "=================================================="
echo "✅ Compilation Complete!"
echo "=================================================="
