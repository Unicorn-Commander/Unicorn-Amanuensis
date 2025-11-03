#!/bin/bash
# Compile Attention Kernel for Whisper Encoder
# Pattern: Same as working matmul and mel kernels

set -e  # Exit on error

echo "======================================================================"
echo "Attention Mechanism Kernel Compilation"
echo "======================================================================"
echo
echo "Source files:"
echo "  - attention_int8.c - INT8 attention mechanism (Q@K^T, softmax, @V)"
echo "  - attention_simple.mlir - MLIR wrapper with ObjectFIFO pattern"
echo
echo "Target: 16x16 int8 attention (256 bytes per Q/K/V matrix)"
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
BUILD_DIR=$WORK_DIR/build_attention
mkdir -p $BUILD_DIR
cd $WORK_DIR

echo "Step 1: Compile attention kernel..."
$PEANO_INSTALL_DIR/bin/clang \
    -O2 \
    -std=c11 \
    --target=aie2-none-unknown-elf \
    -c attention_int8.c \
    -o $BUILD_DIR/attention_int8.o

echo "✅ Attention compiled: $(stat -c%s $BUILD_DIR/attention_int8.o) bytes"
echo

echo "Step 2: Create combined object archive..."
$PEANO_INSTALL_DIR/bin/llvm-ar rcs \
    $BUILD_DIR/attention_combined.o \
    $BUILD_DIR/attention_int8.o

echo "✅ Combined archive: $(stat -c%s $BUILD_DIR/attention_combined.o) bytes"
echo

echo "Step 3: Verify symbols in archive..."
$PEANO_INSTALL_DIR/bin/llvm-nm $BUILD_DIR/attention_combined.o | grep -E "(attention_16x16|softmax)" || true
echo

echo "Step 4: Copy MLIR file..."
cp attention_simple.mlir $BUILD_DIR/attention_simple.mlir
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
    --xclbin-name=attention_simple.xclbin \
    --npu-insts-name=insts.bin \
    attention_simple.mlir

cd $WORK_DIR
echo
echo "======================================================================"
echo "✅ COMPILATION COMPLETE!"
echo "======================================================================"
echo

echo "Generated Files:"
ls -lh $BUILD_DIR/attention_simple.xclbin $BUILD_DIR/insts.bin 2>/dev/null || echo "⚠️  XCLBIN files not found - check compilation output above"
echo

echo "Object Files:"
ls -lh $BUILD_DIR/attention_int8.o
ls -lh $BUILD_DIR/attention_combined.o
echo

echo "Next Step: Test attention mechanism on NPU"
echo "  cd $BUILD_DIR"
echo "  python3 ../test_attention.py"
echo
