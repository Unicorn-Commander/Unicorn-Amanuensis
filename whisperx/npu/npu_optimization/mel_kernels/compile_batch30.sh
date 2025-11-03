#!/bin/bash
# Compile MEL kernel with BATCH-30 optimization
# November 2, 2025

set -e  # Exit on error

echo "======================================================================"
echo "MEL Kernel Compilation - BATCH 30 Optimization"
echo "======================================================================"
echo
echo "Batch size: 30 frames per NPU invocation"
echo "Expected speedup: 1.5x faster mel preprocessing (45x → 67x realtime)"
echo "Memory usage: ~56.9 KB (88.9% of 64 KB AIE tile memory)"
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
BUILD_DIR=$WORK_DIR/build_batch30
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
cp mel_fixed_v3_batch30.mlir $BUILD_DIR/
cp mel_fixed_combined.o $BUILD_DIR/
cd $BUILD_DIR
echo "✅ Files copied"
echo

echo "Step 3: Generate XCLBIN with aiecc.py (batch-30 version)..."
time /home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=mel_batch30.xclbin \
    --npu-insts-name=insts_batch30.bin \
    mel_fixed_v3_batch30.mlir \
    2>&1 | tee compilation.log

cd $WORK_DIR
echo
echo "======================================================================"
echo "✅ COMPILATION COMPLETE!"
echo "======================================================================"
echo

if [ -f "$BUILD_DIR/mel_batch30.xclbin" ]; then
    echo "Generated Files:"
    ls -lh $BUILD_DIR/mel_batch30.xclbin $BUILD_DIR/insts_batch30.bin
    echo

    echo "XCLBIN Information:"
    xclbinutil --input $BUILD_DIR/mel_batch30.xclbin --info | head -50
    echo

    echo "======================================================================"
    echo "✅ SUCCESS! Batch-30 kernel ready for NPU execution"
    echo "======================================================================"
    echo
    echo "Performance expectations:"
    echo "  - Batch size: 30 frames per call"
    echo "  - Expected speedup: 1.5x vs batch-20"
    echo "  - For 1h 44m audio: ~5-7 seconds (from 8-11s)"
    echo "  - Preprocessing realtime: 67x (from 45x)"
    echo
    echo "Next step: Test with Python wrapper"
    echo "  python3 test_mel_batch30.py"
else
    echo "❌ ERROR: XCLBIN generation failed!"
    echo "Check compilation.log for details:"
    tail -50 $BUILD_DIR/compilation.log
    exit 1
fi
