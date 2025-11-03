#!/bin/bash
# Direct MLIR-AIE2 compilation (bypassing aiecc.py and Chess!)
# Based on Meeting-Ops solution that achieved 220x speedup

set -e

echo "=========================================="
echo "Direct MLIR-AIE2 Compilation (No Chess!)"
echo "=========================================="

MLIR_FILE="../matmul_32x32.mlir"
BUILD_DIR="build_matmul_32x32_direct"

mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "Step 1: Lower MLIR to AIE dialect..."
/home/ucadmin/.local/bin/aie-opt \
    --aie-canonicalize-device \
    --aie-lower-to-aie \
    --aie-assign-tile-ids \
    --aie-assign-buffer-addresses \
    $MLIR_FILE \
    -o matmul_lowered.mlir

echo "✅ Lowered to AIE dialect"

echo "Step 2: Generate AIE configuration..."
/home/ucadmin/.local/bin/aie-translate \
    --aie-generate-xclbin \
    --aie-generate-npu \
    matmul_lowered.mlir \
    -o matmul_32x32.xclbin

echo "✅ XCLBIN generated!"

ls -lh matmul_32x32.xclbin

echo "=========================================="
echo "SUCCESS! Compiled without Chess compiler!"
echo "=========================================="
