#!/bin/bash
#
# Compile 64x64 Matrix Multiply Kernel
# 16x fewer kernel invocations than 16x16 for 10x speedup
#
# Expected performance improvement:
# - 512x512 matrix: 512 invocations (vs 32,768 for 16x16)
# - Target: 1,200-1,500ms total time (vs 11,485ms with 16x16)
# - Speedup: 8-10x overall improvement
#

set -e  # Exit on error

echo "=================================================="
echo "Compiling 64x64 Matmul Kernel for 10x Speedup"
echo "=================================================="
echo

# Find Peano compiler
PEANO_INSTALL_DIR="/home/ucadmin/.local/lib/python3.13/site-packages/llvm-aie"

if [ ! -d "$PEANO_INSTALL_DIR" ]; then
    echo "❌ ERROR: Peano compiler not found at $PEANO_INSTALL_DIR"
    exit 1
fi

PEANO=$PEANO_INSTALL_DIR/bin
echo "✅ Found Peano compiler: $PEANO"
echo

# Create build directory
mkdir -p build_matmul_64x64
cd build_matmul_64x64

echo "Step 1: Compiling C kernel to object file..."
echo "  Input: matmul_int8_64x64.c"
echo "  Output: matmul_64x64.o"
echo

$PEANO/clang \
    --target=aie2 \
    -I$PEANO/../aie_kernels/aie2/include \
    -c ../matmul_int8_64x64.c \
    -o matmul_64x64.o

if [ ! -f matmul_64x64.o ]; then
    echo "❌ ERROR: Failed to compile C kernel"
    exit 1
fi

echo "✅ C kernel compiled: matmul_64x64.o ($(stat -c%s matmul_64x64.o) bytes)"
echo

echo "Step 2: Lowering MLIR to hardware representation..."
echo "  Input: matmul_64x64.mlir"
echo "  Output: matmul_64x64_lowered.mlir"
echo

/home/ucadmin/.local/bin/aie-opt \
    --aie-canonicalize-device \
    --aie-objectFifo-stateful-transform \
    --aie-create-pathfinder-flows \
    --aie-assign-buffer-addresses \
    ../matmul_64x64.mlir -o matmul_64x64_lowered.mlir

if [ ! -f matmul_64x64_lowered.mlir ]; then
    echo "❌ ERROR: Failed to lower MLIR"
    exit 1
fi

echo "✅ MLIR lowered: matmul_64x64_lowered.mlir ($(stat -c%s matmul_64x64_lowered.mlir) bytes)"
echo

echo "Step 3: Generating XCLBIN binary..."
echo "  Input: matmul_64x64_lowered.mlir + matmul_64x64.o"
echo "  Output: matmul_64x64.xclbin"
echo

/home/ucadmin/.local/bin/aiecc.py \
    --sysroot=$PEANO/../sysroot \
    --host-target=x86_64-amd-linux-gnu \
    ../matmul_64x64.mlir \
    -I$PEANO/../aie_kernels/aie2/include \
    -o matmul_64x64.xclbin \
    --xclbin-kernel-name=MLIR_AIE \
    --peano-install-dir=$PEANO

if [ ! -f matmul_64x64.xclbin ]; then
    echo "❌ ERROR: Failed to generate XCLBIN"
    exit 1
fi

echo "✅ XCLBIN generated: matmul_64x64.xclbin ($(stat -c%s matmul_64x64.xclbin) bytes)"
echo

# Create symlink for easy access
ln -sf build_matmul_64x64/matmul_64x64.xclbin ../matmul_64x64.xclbin 2>/dev/null || true

# Check file sizes
echo "=================================================="
echo "Build Artifacts Summary"
echo "=================================================="
ls -lh matmul_64x64.o matmul_64x64.xclbin matmul_64x64_lowered.mlir 2>/dev/null || true
echo

echo "=================================================="
echo "✅ 64x64 Matmul Kernel Compiled Successfully!"
echo "=================================================="
echo
echo "Memory Usage Analysis:"
echo "  Input buffer:  8,192 bytes (64x64 A + 64x64 B)"
echo "  Output buffer: 4,096 bytes (64x64 C)"
echo "  Accumulator:   16,384 bytes (64x64 int32)"
echo "  Total:         ~28 KB (88% of 32 KB tile memory)"
echo
echo "Performance Expectations:"
echo "  512x512 matrix: 512 kernel calls (vs 32,768 for 16x16)"
echo "  API overhead:   ~154ms (vs 9,830ms for 16x16)"
echo "  Compute time:   ~1,200ms (slightly slower per tile)"
echo "  Total time:     ~1,350ms (vs 11,485ms for 16x16)"
echo "  Expected speedup: 8-10x ✨"
echo
echo "Next steps:"
echo "  1. Update Python wrapper: tile_size=64"
echo "  2. Test with benchmark: python3 test_batched_matmul_benchmark.py --tile-size=64"
echo "  3. Verify 10x speedup on 512x512 matrices"
echo
