#!/bin/bash
# Compile Softmax BF16 Kernel for XDNA1 Phoenix NPU
# Pattern: Same as working GELU, matmul, and attention kernels

set -e  # Exit on error

echo "======================================================================"
echo "Softmax BF16 Kernel Compilation for XDNA1"
echo "======================================================================"
echo
echo "Source files:"
echo "  - softmax_bf16_xdna1.cc - BF16 softmax activation (1024 elements)"
echo "  - softmax_bf16.mlir - MLIR wrapper with ObjectFIFO pattern"
echo
echo "Target: 1024 bfloat16 softmax (2048 bytes)"
echo "Expected Performance: <1ms for 1024 elements"
echo "======================================================================"
echo

# Setup environment (same as other kernels)
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

# Verify tools
echo "Step 0: Verify toolchain..."
if [ ! -f "$PEANO_INSTALL_DIR/bin/clang" ]; then
    echo "❌ ERROR: Peano clang not found at $PEANO_INSTALL_DIR/bin/clang"
    exit 1
fi
if ! command -v aiecc.py &> /dev/null; then
    echo "❌ ERROR: aiecc.py not found in PATH"
    exit 1
fi
echo "✅ Peano clang: $PEANO_INSTALL_DIR/bin/clang"
echo "✅ aiecc.py: $(which aiecc.py)"
echo

# Create build directory
WORK_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/kernels_xdna1
BUILD_DIR=$WORK_DIR/build_softmax_bf16
mkdir -p $BUILD_DIR
cd $WORK_DIR

echo "Step 1: Compile softmax kernel..."
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

echo "✅ Softmax kernel compiled: $(stat -c%s $BUILD_DIR/softmax_bf16_xdna1.o) bytes"
echo

echo "Step 2: Create combined object archive..."
$PEANO_INSTALL_DIR/bin/llvm-ar rcs \
    $BUILD_DIR/softmax_bf16_xdna1_combined.o \
    $BUILD_DIR/softmax_bf16_xdna1.o

echo "✅ Combined archive: $(stat -c%s $BUILD_DIR/softmax_bf16_xdna1_combined.o) bytes"
echo

echo "Step 3: Verify symbols in archive..."
$PEANO_INSTALL_DIR/bin/llvm-nm $BUILD_DIR/softmax_bf16_xdna1_combined.o | grep -E "softmax" || true
echo

echo "Step 4: Copy MLIR file..."
cp softmax_bf16.mlir $BUILD_DIR/softmax_bf16.mlir
echo "✅ MLIR file prepared"
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
    --xclbin-name=softmax_bf16.xclbin \
    --npu-insts-name=insts.bin \
    softmax_bf16.mlir

cd $WORK_DIR
echo
echo "======================================================================"
echo "✅ COMPILATION COMPLETE!"
echo "======================================================================"
echo

echo "Generated Files:"
ls -lh $BUILD_DIR/softmax_bf16.xclbin $BUILD_DIR/insts.bin 2>/dev/null || echo "⚠️  XCLBIN files not found - check compilation output above"
echo

echo "Object Files:"
ls -lh $BUILD_DIR/softmax_bf16_xdna1.o
ls -lh $BUILD_DIR/softmax_bf16_xdna1_combined.o
echo

echo "Compilation Times:"
echo "  - C++ kernel compilation: ~0.2-0.3s"
echo "  - MLIR lowering: ~0.3-0.5s"
echo "  - Total: <1s"
echo

echo "Expected Performance:"
echo "  - 1024 elements: <1ms (target)"
echo "  - Numerically stable BF16 implementation"
echo

echo "Next Step: Test Softmax on NPU"
echo "  cd $BUILD_DIR"
echo "  python3 ../test_softmax.py"
echo
