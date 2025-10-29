#!/bin/bash
# Compile MEL kernel with FIXED source code (FFT scaling + HTK mel filters)
# October 28, 2025 - Using fixes from 21:06-21:23 UTC

set -e  # Exit on error

echo "======================================================================"
echo "MEL Kernel Recompilation with Fixes"
echo "======================================================================"
echo
echo "Source files:"
echo "  - fft_fixed_point.c (21:06 UTC) - FFT with scaling fix"
echo "  - mel_kernel_fft_fixed.c (21:23 UTC) - HTK mel filterbanks"
echo "  - mel_coeffs_fixed.h (21:21 UTC) - 207 KB coefficient table"
echo
echo "======================================================================"
echo

# Setup environment (same as successful build at 17:03)
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

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
BUILD_DIR=$WORK_DIR/build_fixed_v3
mkdir -p $BUILD_DIR
cd $WORK_DIR

echo "Step 1: Compile FIXED FFT module (with scaling)..."
$PEANO_INSTALL_DIR/bin/clang \
    -O2 \
    -std=c11 \
    --target=aie2-none-unknown-elf \
    -c fft_fixed_point.c \
    -o $BUILD_DIR/fft_fixed_point_v3.o

echo "✅ FFT compiled: $(stat -c%s $BUILD_DIR/fft_fixed_point_v3.o) bytes"
echo

echo "Step 2: Compile FIXED MEL kernel (with HTK filters)..."
$PEANO_INSTALL_DIR/bin/clang++ \
    -O2 \
    -std=c++20 \
    --target=aie2-none-unknown-elf \
    -c mel_kernel_fft_fixed.c \
    -o $BUILD_DIR/mel_kernel_fft_fixed_v3.o

echo "✅ MEL kernel compiled: $(stat -c%s $BUILD_DIR/mel_kernel_fft_fixed_v3.o) bytes"
echo

echo "Step 3: Create combined object archive..."
$PEANO_INSTALL_DIR/bin/llvm-ar rcs \
    $BUILD_DIR/mel_fixed_combined_v3.o \
    $BUILD_DIR/fft_fixed_point_v3.o \
    $BUILD_DIR/mel_kernel_fft_fixed_v3.o

echo "✅ Combined archive: $(stat -c%s $BUILD_DIR/mel_fixed_combined_v3.o) bytes"
echo

echo "Step 4: Verify symbols in archive..."
$PEANO_INSTALL_DIR/bin/llvm-nm $BUILD_DIR/mel_fixed_combined_v3.o | grep -E "(mel_kernel_simple|fft_radix2_512_fixed|apply_mel_filters)" || true
echo

echo "Step 5: Copy MLIR file and update link_with..."
cp mel_fixed_v3.mlir $BUILD_DIR/mel_fixed_v3.mlir
# Update link_with to point to new object file
sed -i 's/link_with = "mel_fixed_combined_v2.o"/link_with = "mel_fixed_combined_v3.o"/' $BUILD_DIR/mel_fixed_v3.mlir
echo "✅ MLIR file prepared"
echo

echo "Step 6: Generate XCLBIN with aiecc.py..."
cd $BUILD_DIR

/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=mel_fixed_v3.xclbin \
    --npu-insts-name=insts_v3.bin \
    mel_fixed_v3.mlir

cd $WORK_DIR
echo
echo "======================================================================"
echo "✅ COMPILATION COMPLETE!"
echo "======================================================================"
echo
echo "Generated Files:"
ls -lh $BUILD_DIR/mel_fixed_v3.xclbin $BUILD_DIR/insts_v3.bin 2>/dev/null || echo "⚠️  XCLBIN files not found - check compilation output above"
echo
echo "Object Files (validated):"
ls -lh $BUILD_DIR/fft_fixed_point_v3.o
ls -lh $BUILD_DIR/mel_kernel_fft_fixed_v3.o
ls -lh $BUILD_DIR/mel_fixed_combined_v3.o
echo
echo "Next Step: Test on NPU hardware"
echo "  python3 quick_correlation_test.py"
echo
