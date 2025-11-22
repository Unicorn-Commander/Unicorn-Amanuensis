#!/bin/bash
#
# Compile Streaming Encoder Layer for Phoenix NPU
#
# This script compiles all kernels and generates the XCLBIN for
# a full encoder layer with streaming architecture.
#

set -e  # Exit on error

# Source environment
source setup_env.sh

echo "=== Compiling Streaming Encoder Layer ==="
echo ""

# Working directory
WORK_DIR="$(pwd)/kernels_xdna1"
BUILD_DIR="$(pwd)/build_streaming_encoder"
mkdir -p "$BUILD_DIR"

cd "$WORK_DIR"

# Compiler settings
PEANO=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie/bin/clang
INCLUDES="-I/home/ucadmin/mlir-aie-source/third_party/aie_api/include"
INCLUDES="$INCLUDES -I/home/ucadmin/mlir-aie-source/aie_runtime_lib/AIE2"
INCLUDES="$INCLUDES -I/home/ucadmin/mlir-aie-source/aie_runtime_lib"
CFLAGS="-O2 -std=c++20 --target=aie2-none-unknown-elf"

echo "Step 1: Compiling C++ kernels to object files..."
echo "----------------------------------------------"

# Compile LayerNorm (streaming 512 elements)
echo "Compiling layernorm_streaming_bf16.cc..."
timeout 120 $PEANO $CFLAGS $INCLUDES -c layernorm_streaming_bf16.cc -o layernorm_streaming_bf16.o
echo "✓ LayerNorm compiled ($(ls -lh layernorm_streaming_bf16.o | awk '{print $5}'))"

# Compile Softmax (streaming 1500 elements)
echo "Compiling softmax_streaming_bf16.cc..."
timeout 120 $PEANO $CFLAGS $INCLUDES -c softmax_streaming_bf16.cc -o softmax_streaming_bf16.o
echo "✓ Softmax compiled ($(ls -lh softmax_streaming_bf16.o | awk '{print $5}'))"

# Compile GELU (FFN 2048 elements)
echo "Compiling gelu_ffn_bf16.cc..."
timeout 120 $PEANO $CFLAGS $INCLUDES -c gelu_ffn_bf16.cc -o gelu_ffn_bf16.o
echo "✓ GELU compiled ($(ls -lh gelu_ffn_bf16.o | awk '{print $5}'))"

# Compile MatMul (64×64 tiled)
echo "Compiling matmul_64x64_bf16.cc..."
timeout 120 $PEANO $CFLAGS $INCLUDES -c matmul_64x64_bf16.cc -o matmul_64x64_bf16.o
echo "✓ MatMul 64×64 compiled ($(ls -lh matmul_64x64_bf16.o | awk '{print $5}'))"

echo ""
echo "Step 2: Verifying object files with llvm-nm..."
echo "----------------------------------------------"

for obj in layernorm_streaming_bf16.o softmax_streaming_bf16.o gelu_ffn_bf16.o matmul_64x64_bf16.o; do
    echo "Symbols in $obj:"
    $PEANO_INSTALL_DIR/bin/llvm-nm $obj | grep " T " || echo "  (no exported symbols - checking all symbols)"
    $PEANO_INSTALL_DIR/bin/llvm-nm $obj | head -5
    echo ""
done

echo "Step 3: Generating XCLBIN with aiecc.py..."
echo "----------------------------------------------"

# Copy MLIR file to build directory
cp encoder_streaming_layer.mlir "$BUILD_DIR/"
cp *.o "$BUILD_DIR/"

cd "$BUILD_DIR"

# Generate XCLBIN
echo "Running aiecc.py (this may take 2-5 minutes)..."
timeout 300 /home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --no-xchesscc \
    --no-xbridge \
    encoder_streaming_layer.mlir \
    2>&1 | tee compilation.log

# Check if XCLBIN was created
if [ -f "final.xclbin" ]; then
    echo ""
    echo "=== ✓ COMPILATION SUCCESSFUL ==="
    echo "XCLBIN: $BUILD_DIR/final.xclbin"
    ls -lh final.xclbin
    echo ""
    echo "Size: $(ls -lh final.xclbin | awk '{print $5}')"
    echo "MD5: $(md5sum final.xclbin | awk '{print $1}')"
    echo ""
    echo "You can now test with:"
    echo "  cd $BUILD_DIR"
    echo "  python3 ../../test_streaming_encoder.py"
else
    echo ""
    echo "=== ✗ COMPILATION FAILED ==="
    echo "Check compilation.log for details"
    tail -50 compilation.log
    exit 1
fi
