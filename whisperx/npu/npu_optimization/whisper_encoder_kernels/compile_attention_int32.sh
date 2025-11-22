#!/bin/bash
# Compile Attention Kernel for Whisper Encoder - 64x64 SCALED VERSION
# Pattern: Same as working matmul and mel kernels

set -e  # Exit on error

echo "======================================================================"
echo "Attention Mechanism Kernel Compilation - INT32 VERSION"
echo "======================================================================"
echo
echo "Source files:"
echo "  - attention_int8_64x64.c - INT8 attention mechanism (64x64 matrices)"
echo "  - attention_64x64.mlir - MLIR wrapper with ObjectFIFO pattern"
echo
echo "Target: 64x64 int8 attention (4096 bytes per Q/K/V matrix)"
echo "Memory: 64x64 int32 accumulator = 16KB (fits in AIE2 32KB)"
echo "Expected Performance: 8-10ms per tile (vs 0.56ms for 16x16)"
echo "======================================================================"
echo

# Setup environment (same as matmul and mel kernel)
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
WORK_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels
BUILD_DIR=$WORK_DIR/build_attention_int32
mkdir -p $BUILD_DIR
cd $WORK_DIR

echo "Step 1: Compile attention kernel (64x64 tiled version)..."
$PEANO_INSTALL_DIR/bin/clang \
    -O2 \
    -std=c11 \
    --target=aie2-none-unknown-elf \
    -c attention_int8_64x64_tiled.c \
    -o $BUILD_DIR/attention_int8_64x64.o

echo "✅ Attention compiled: $(stat -c%s $BUILD_DIR/attention_int8_64x64.o) bytes"
echo

echo "Step 2: Create combined object archive..."
$PEANO_INSTALL_DIR/bin/llvm-ar rcs \
    $BUILD_DIR/attention_combined_64x64.o \
    $BUILD_DIR/attention_int8_64x64.o

echo "✅ Combined archive: $(stat -c%s $BUILD_DIR/attention_combined_64x64.o) bytes"
echo

echo "Step 3: Verify symbols in archive..."
$PEANO_INSTALL_DIR/bin/llvm-nm $BUILD_DIR/attention_combined_64x64.o | grep -E "(attention_64x64|softmax)" || true
echo

echo "Step 4: Copy MLIR file..."
cp attention_64x64.mlir $BUILD_DIR/attention_64x64.mlir
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
    --xclbin-name=attention_64x64.xclbin \
    --npu-insts-name=insts.bin \
    attention_64x64.mlir

cd $WORK_DIR
echo
echo "======================================================================"
echo "✅ COMPILATION COMPLETE!"
echo "======================================================================"
echo

echo "Generated Files:"
ls -lh $BUILD_DIR/attention_64x64.xclbin $BUILD_DIR/insts.bin 2>/dev/null || echo "⚠️  XCLBIN files not found - check compilation output above"
echo

echo "Object Files:"
ls -lh $BUILD_DIR/attention_int8_64x64.o
ls -lh $BUILD_DIR/attention_combined_64x64.o
echo

echo "Next Step: Test attention mechanism on NPU"
echo "  cd $BUILD_DIR"
echo "  python3 ../test_attention_64x64.py"
echo
