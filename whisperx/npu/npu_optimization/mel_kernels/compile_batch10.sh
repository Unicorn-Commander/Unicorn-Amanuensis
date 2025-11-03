#!/bin/bash
# Compile MEL kernel with BATCH-10 optimization
# November 1, 2025

set -e  # Exit on error

echo "======================================================================"
echo "MEL Kernel Compilation - BATCH 10 Optimization"
echo "======================================================================"
echo
echo "Batch size: 10 frames per NPU invocation"
echo "Expected speedup: 6-8x vs single-frame processing"
echo "Memory usage: ~18.6 KB (fits in 64 KB AIE tile memory)"
echo
echo "======================================================================"
echo

# Setup environment
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH
export LD_LIBRARY_PATH=/home/ucadmin/tools/vitis_aie_essentials/lib:$LD_LIBRARY_PATH

# Verify tools available
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
echo "✅ xclbinutil: $(which xclbinutil)"
echo

# Create build directory
WORK_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels
BUILD_DIR=$WORK_DIR/build_batch10
mkdir -p $BUILD_DIR
cd $WORK_DIR

echo "Step 1: Verify C kernel object file exists..."
if [ ! -f "mel_fixed_combined.o" ]; then
    echo "❌ ERROR: mel_fixed_combined.o not found"
    echo "   Run build script to create C kernel first"
    exit 1
fi
echo "✅ C kernel object: $(stat -c%s mel_fixed_combined.o) bytes"
echo

echo "Step 2: Copy files to build directory..."
cp mel_fixed_v3_batch10.mlir $BUILD_DIR/
cp mel_fixed_combined.o $BUILD_DIR/
cd $BUILD_DIR
echo "✅ Files copied"
echo

echo "Step 3: Generate XCLBIN with aiecc.py (batch-10 version)..."
time /home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=mel_batch10.xclbin \
    --npu-insts-name=insts_batch10.bin \
    mel_fixed_v3_batch10.mlir \
    2>&1 | tee compilation.log

cd $WORK_DIR
echo
echo "======================================================================"
echo "✅ COMPILATION COMPLETE!"
echo "======================================================================"
echo

if [ -f "$BUILD_DIR/mel_batch10.xclbin" ]; then
    echo "Generated Files:"
    ls -lh $BUILD_DIR/mel_batch10.xclbin $BUILD_DIR/insts_batch10.bin
    echo

    echo "XCLBIN Information:"
    xclbinutil --input $BUILD_DIR/mel_batch10.xclbin --info | head -50
    echo

    echo "======================================================================"
    echo "✅ SUCCESS! Batch-10 kernel ready for NPU execution"
    echo "======================================================================"
    echo
    echo "Performance expectations:"
    echo "  - Batch size: 10 frames per call"
    echo "  - Expected speedup: 6-8x vs single-frame"
    echo "  - For 1h 44m audio: ~16-22 seconds (from 134s)"
    echo
    echo "Next step: Test with Python wrapper"
    echo "  python3 test_mel_batch10.py"
else
    echo "❌ ERROR: XCLBIN generation failed!"
    echo "Check compilation.log for details:"
    tail -50 $BUILD_DIR/compilation.log
    exit 1
fi
