#!/bin/bash
# Compile Batched Softmax BF16 Kernel for XDNA1 Phoenix NPU
# Processes 4 softmax operations per invocation to achieve 3x speedup

set -e  # Exit on error

echo "======================================================================"
echo "Batched Softmax BF16 Kernel Compilation for XDNA1"
echo "======================================================================"
echo
echo "Source files:"
echo "  - softmax_bf16_xdna1_batched.cc - BF16 batched softmax (4 × 1024 elements)"
echo "  - softmax_batched_bf16.mlir - MLIR wrapper with ObjectFIFO pattern"
echo
echo "Target: 4 × 1024 bfloat16 softmax operations (8192 bytes)"
echo "Expected Performance: 0.52ms per-frame (3x speedup from 1.565ms)"
echo "======================================================================"
echo

# Setup environment (same as other kernels)
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

# Verify tools
echo "Step 0: Verify toolchain..."
if [ ! -f "$PEANO_INSTALL_DIR/bin/clang" ]; then
    echo "ERROR: Peano clang not found at $PEANO_INSTALL_DIR/bin/clang"
    exit 1
fi
if ! command -v aiecc.py &> /dev/null; then
    echo "ERROR: aiecc.py not found in PATH"
    exit 1
fi
echo "Peano clang: $PEANO_INSTALL_DIR/bin/clang"
echo "aiecc.py: $(which aiecc.py)"
echo

# Create build directory
WORK_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1
BUILD_DIR=$WORK_DIR/build_softmax_batched
mkdir -p $BUILD_DIR
cd $WORK_DIR

echo "Step 1: Compile batched softmax kernel..."
$PEANO_INSTALL_DIR/bin/clang \
    -O2 \
    -std=c++20 \
    --target=aie2-none-unknown-elf \
    -I/home/ucadmin/mlir-aie-source/third_party/aie_api/include \
    -I/home/ucadmin/mlir-aie-source/aie_runtime_lib/AIE2 \
    -I/home/ucadmin/mlir-aie-source/aie_runtime_lib \
    -I/home/ucadmin/mlir-aie-source/aie_kernels \
    -c softmax_bf16_xdna1_batched.cc \
    -o $BUILD_DIR/softmax_bf16_xdna1_batched.o

echo "Batched softmax kernel compiled: $(stat -c%s $BUILD_DIR/softmax_bf16_xdna1_batched.o) bytes"
echo

echo "Step 2: Create combined object archive..."
$PEANO_INSTALL_DIR/bin/llvm-ar rcs \
    $BUILD_DIR/softmax_bf16_xdna1_batched_combined.o \
    $BUILD_DIR/softmax_bf16_xdna1_batched.o

echo "Combined archive: $(stat -c%s $BUILD_DIR/softmax_bf16_xdna1_batched_combined.o) bytes"
echo

echo "Step 3: Verify symbols in archive..."
$PEANO_INSTALL_DIR/bin/llvm-nm $BUILD_DIR/softmax_bf16_xdna1_batched_combined.o | grep -E "softmax" || true
echo

echo "Step 4: Copy MLIR file..."
cp softmax_batched_bf16.mlir $BUILD_DIR/softmax_batched_bf16.mlir
echo "MLIR file prepared"
echo

echo "Step 5: Generate XCLBIN with aiecc.py..."
cd $BUILD_DIR

/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=softmax_batched_bf16.xclbin \
    --npu-insts-name=insts.bin \
    softmax_batched_bf16.mlir

cd $WORK_DIR
echo
echo "======================================================================"
echo "COMPILATION COMPLETE!"
echo "======================================================================"
echo

echo "Generated Files:"
ls -lh $BUILD_DIR/softmax_batched_bf16.xclbin $BUILD_DIR/insts.bin 2>/dev/null || echo "XCLBIN files not found - check compilation output above"
echo

echo "Object Files:"
ls -lh $BUILD_DIR/softmax_bf16_xdna1_batched.o
ls -lh $BUILD_DIR/softmax_bf16_xdna1_batched_combined.o
echo

echo "Expected Performance:"
echo "  - Batch size: 4"
echo "  - Total time: ~2.1ms for 4 softmax operations"
echo "  - Per-frame: 0.52ms (3x speedup from 1.565ms)"
echo "  - Memory: 16 KB (8 KB input + 8 KB output)"
echo

echo "Next Step: Test Batched Softmax on NPU"
echo "  cd $BUILD_DIR"
echo "  python3 ../test_softmax_batched.py"
echo
