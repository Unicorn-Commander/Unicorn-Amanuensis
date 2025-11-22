#!/bin/bash
# Compile Parallel Softmax - 4 Tiles Processing Simultaneously
# XDNA1 Phoenix NPU

set -e

echo "======================================================================"
echo "Parallel Softmax Compilation - 4 Tiles for XDNA1"
echo "======================================================================"
echo
echo "Source files:"
echo "  - softmax_bf16_xdna1.cc - Single softmax kernel (reused)"
echo "  - softmax_parallel_4tile.mlir - 4-tile parallel MLIR"
echo
echo "Target: 4 × 1024 bfloat16 softmax operations in parallel"
echo "Expected Performance: ~1.6 ms for 4 frames (vs 5.4 ms batched)"
echo "======================================================================"
echo

# Setup environment
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

# Verify tools
echo "Step 0: Verify toolchain..."
if [ ! -f "$PEANO_INSTALL_DIR/bin/clang" ]; then
    echo "ERROR: Peano clang not found"
    exit 1
fi
echo "Peano clang: $PEANO_INSTALL_DIR/bin/clang"
echo

# Create build directory
WORK_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1
BUILD_DIR=$WORK_DIR/build_softmax_parallel
mkdir -p $BUILD_DIR
cd $WORK_DIR

# Reuse existing softmax kernel
echo "Step 1: Copy existing softmax kernel..."
if [ -f "build_softmax_bf16/softmax_bf16_xdna1.o" ]; then
    cp build_softmax_bf16/softmax_bf16_xdna1.o $BUILD_DIR/
    echo "Reusing existing kernel: $(stat -c%s $BUILD_DIR/softmax_bf16_xdna1.o) bytes"
else
    # Compile if not exists
    echo "Compiling softmax kernel..."
    $PEANO_INSTALL_DIR/bin/clang \
        -O2 \
        -std=c++20 \
        --target=aie2-none-unknown-elf \
        -I/home/ucadmin/mlir-aie-source/third_party/aie_api/include \
        -I/home/ucadmin/mlir-aie-source/aie_runtime_lib/AIE2 \
        -I/home/ucadmin/mlir-aie-source/aie_runtime_lib \
        -I/home/ucadmin/mlir-aie-source/aie_kernels \
        -c softmax_bf16_xdna1.cc \
        -o $BUILD_DIR/softmax_bf16_xdna1.o
    echo "Kernel compiled: $(stat -c%s $BUILD_DIR/softmax_bf16_xdna1.o) bytes"
fi
echo

echo "Step 2: Copy MLIR file..."
cp softmax_parallel_4tile.mlir $BUILD_DIR/softmax_parallel_4tile.mlir
echo "MLIR file prepared"
echo

echo "Step 3: Generate XCLBIN with aiecc.py..."
cd $BUILD_DIR

/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=softmax_parallel_4tile.xclbin \
    --npu-insts-name=insts.bin \
    softmax_parallel_4tile.mlir

cd $WORK_DIR
echo
echo "======================================================================"
echo "COMPILATION COMPLETE!"
echo "======================================================================"
echo

echo "Generated Files:"
ls -lh $BUILD_DIR/softmax_parallel_4tile.xclbin $BUILD_DIR/insts.bin 2>/dev/null || echo "XCLBIN files not found"
echo

echo "Next Step: Test parallel softmax on NPU"
echo "  python3 test_softmax_parallel.py"
echo
