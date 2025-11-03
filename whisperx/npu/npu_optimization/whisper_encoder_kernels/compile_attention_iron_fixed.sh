#!/bin/bash
# Compile Fixed Multi-Core Attention Kernel (Parallel Execution)
# Fixed: Parallel DMA execution instead of serial

set -e  # Exit on error

echo "======================================================================"
echo "Multi-Core Attention Kernel Compilation - PARALLEL EXECUTION FIX"
echo "======================================================================"
echo
echo "Source files:"
echo "  - attention_int8_64x64_tiled.o - Compiled attention kernel"
echo "  - attention_iron_generated.mlir - FIXED MLIR with parallel DMAs"
echo
echo "Target: 4-tile parallel attention (4 NPU columns)"
echo "Fix: All tiles execute in parallel (not serial)"
echo "Expected Performance: ~2.5ms for 4 tiles (same as single tile)"
echo "======================================================================"
echo

# Setup environment
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

# Set directories
WORK_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels
BUILD_DIR=$WORK_DIR/build_attention_iron
cd $BUILD_DIR

echo "Step 1: MLIR file already fixed (parallel DMA execution)"
echo "✅ Fixed MLIR: attention_iron_generated.mlir"
ls -lh attention_iron_generated.mlir
echo

echo "Step 2: Kernel object already compiled"
echo "✅ Kernel object: attention_int8_64x64_tiled.o"
ls -lh attention_int8_64x64_tiled.o
echo

echo "Step 3: Generate XCLBIN with aiecc.py (parallel fix)..."
echo "This will compile the fixed MLIR with parallel DMA execution"
echo

/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=attention_multicore_fixed.xclbin \
    --npu-insts-name=insts_fixed.bin \
    attention_iron_generated.mlir

echo
echo "======================================================================"
echo "✅ COMPILATION COMPLETE WITH PARALLEL FIX!"
echo "======================================================================"
echo

echo "Generated Files:"
ls -lh attention_multicore_fixed.xclbin insts_fixed.bin 2>/dev/null || echo "⚠️  XCLBIN files not found - check compilation output above"
echo

echo "Comparison:"
echo "  OLD (serial): attention_multicore.xclbin + insts.bin"
echo "  NEW (parallel): attention_multicore_fixed.xclbin + insts_fixed.bin"
echo

echo "Next Step: Test fixed kernel on NPU"
echo "  python3 test_attention_multicore_iron.py"
echo

