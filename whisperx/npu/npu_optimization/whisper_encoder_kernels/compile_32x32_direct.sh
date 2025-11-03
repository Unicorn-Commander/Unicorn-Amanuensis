#!/bin/bash
set -e

echo "=================================================="
echo "Compiling 32x32 Matmul Kernel (Direct Method)"
echo "=================================================="
echo

# Set environment
export PEANO_INSTALL_DIR=/home/ucadmin/.local/lib/python3.13/site-packages/llvm-aie
export PATH=/opt/xilinx/xrt/bin:$PATH

cd build_matmul_32x32

echo "Step 1: Lower MLIR with aie-opt..."
/home/ucadmin/.local/bin/aie-opt \
    --aie-canonicalize-device \
    --aie-objectFifo-stateful-transform \
    --aie-create-pathfinder-flows \
    --aie-assign-buffer-addresses \
    --aie-objectFifo-register-process \
    --aie-assign-lock-ids \
    --aie-register-objectFifos \
    --aie-normalize-address-spaces \
    --aie-standard-lowering=tilecol=0 tilerow=2 \
    ../matmul_32x32.mlir -o matmul_32x32_lowered.mlir

echo "✅ MLIR lowered successfully"
echo

echo "Step 2: Generate XCLBIN with aie-translate..."
/home/ucadmin/.local/bin/aie-translate \
    --aie-generate-xclbin \
    --xclbin-name=matmul_32x32.xclbin \
    matmul_32x32_lowered.mlir

echo
echo "=================================================="
echo "✅ 32x32 Matmul Kernel Compiled Successfully!"
echo "=================================================="
echo
echo "Build artifacts:"
ls -lh matmul_32x32.xclbin 2>/dev/null || echo "XCLBIN not generated - check logs"
echo
