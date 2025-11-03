#!/bin/bash
# Compile Multi-Core Attention Kernel for Whisper Encoder
# Uses all 4 columns for 4x throughput

set -e

echo "======================================================================"
echo "Multi-Core Attention Mechanism - 4× PARALLEL PROCESSING"
echo "======================================================================"
echo
echo "Architecture: Uses all 4 Phoenix NPU columns"
echo "  Column 0: Processes tile 0 (Q+K+V → Attention output)"
echo "  Column 1: Processes tile 1 (Q+K+V → Attention output)"
echo "  Column 2: Processes tile 2 (Q+K+V → Attention output)"
echo "  Column 3: Processes tile 3 (Q+K+V → Attention output)"
echo
echo "Performance: 4× throughput vs single-core"
echo "Input: 4 × 12288 bytes = 49152 bytes (4 tiles of Q+K+V)"
echo "Output: 4 × 4096 bytes = 16384 bytes (4 attention outputs)"
echo "======================================================================"
echo

# Setup environment
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

# Verify tools
echo "Step 0: Verify toolchain..."
if [ ! -f "$PEANO_INSTALL_DIR/bin/clang" ]; then
    echo "❌ ERROR: Peano clang not found"
    exit 1
fi
if ! command -v aiecc.py &> /dev/null; then
    echo "❌ ERROR: aiecc.py not found"
    exit 1
fi
echo "✅ Peano clang: $PEANO_INSTALL_DIR/bin/clang"
echo "✅ aiecc.py: $(which aiecc.py)"
echo

WORK_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels
BUILD_DIR=$WORK_DIR/build_attention_multicore
mkdir -p $BUILD_DIR
cd $WORK_DIR

echo "Step 1: Compile attention kernel (same as single-core)..."
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

echo "Step 3: Copy MLIR file..."
cp attention_64x64_multicore.mlir $BUILD_DIR/attention_64x64_multicore.mlir
echo "✅ MLIR file prepared (multi-core design)"
echo

echo "Step 4: Generate XCLBIN with aiecc.py..."
cd $BUILD_DIR

/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=attention_multicore.xclbin \
    --npu-insts-name=insts_multicore.bin \
    attention_64x64_multicore.mlir

cd $WORK_DIR
echo
echo "======================================================================"
echo "✅ MULTI-CORE COMPILATION COMPLETE!"
echo "======================================================================"
echo
echo "Generated Files:"
ls -lh $BUILD_DIR/attention_multicore.xclbin $BUILD_DIR/insts_multicore.bin 2>/dev/null || echo "⚠️  Check compilation output"
echo
echo "Architecture:"
echo "  - 4 parallel compute cores"
echo "  - 4 independent DMA channels"
echo "  - 4× throughput improvement"
echo
echo "Next: Test multi-core kernel"
echo "  cd $BUILD_DIR"
echo "  python3 ../test_attention_multicore.py"
echo
