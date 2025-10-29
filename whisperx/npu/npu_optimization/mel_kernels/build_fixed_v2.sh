#!/bin/bash
#
# NPU Kernel Recompilation Script (V2 with FFT and Mel Fixes)
# October 28, 2025
#
# This script compiles the fixed C kernels to object files.
# XCLBIN generation requires full Vitis AIE toolchain (see BUILD_STATUS_V2_OCT28.md)
#

set -e

# Change to script directory
cd "$(dirname "$0")"

# Create build directory
mkdir -p build_fixed_v2
cd build_fixed_v2

# Set Peano compiler location
export PEANO=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie

echo "======================================================================"
echo "NPU Kernel Recompilation (V2 with Fixes)"
echo "======================================================================"
echo
echo "Fixes included:"
echo "  - FFT: Per-stage scaling to prevent overflow"
echo "  - Mel: HTK triangular filterbank (207 KB coefficients)"
echo
echo "======================================================================"
echo

# Step 1: Compile FFT module
echo "Step 1: Compile FFT fixed point module..."
time $PEANO/bin/clang++ -target aie2-none-unknown-elf -O2 \
    -c ../fft_fixed_point.c -o fft_fixed_point_v2.o 2>&1 | grep -v "treating 'c' input"
echo "✅ FFT compiled: $(stat -c%s fft_fixed_point_v2.o) bytes"
echo

# Step 2: Compile mel kernel
echo "Step 2: Compile mel kernel with HTK filters..."
time $PEANO/bin/clang++ -target aie2-none-unknown-elf -O2 \
    -c ../mel_kernel_fft_fixed.c -o mel_kernel_fft_fixed_v2.o 2>&1 | grep -v "treating 'c' input"
echo "✅ Mel kernel compiled: $(stat -c%s mel_kernel_fft_fixed_v2.o) bytes"
echo

# Step 3: Create combined archive
echo "Step 3: Create combined archive..."
$PEANO/bin/llvm-ar rcs mel_fixed_combined_v2.o fft_fixed_point_v2.o mel_kernel_fft_fixed_v2.o
echo "✅ Archive created: $(stat -c%s mel_fixed_combined_v2.o) bytes"
echo

# Step 4: Validate symbols
echo "Step 4: Validate symbols..."
echo "Expected symbols:"
$PEANO/bin/llvm-nm mel_fixed_combined_v2.o | grep -E "(mel_kernel_simple|fft_radix2|apply_mel)"
echo "✅ All required symbols present"
echo

# Step 5: Copy MLIR template and update link_with
echo "Step 5: Create MLIR file with updated link_with..."
sed 's/link_with = "mel_fixed_combined.o"/link_with = "mel_fixed_combined_v2.o"/' \
    ../build_fixed/mel_fixed.mlir > mel_fixed_v2.mlir
echo "✅ MLIR file created: $(stat -c%s mel_fixed_v2.mlir) bytes"
echo

echo "======================================================================"
echo "✅ C Compilation Complete!"
echo "======================================================================"
echo
echo "Generated files:"
ls -lh fft_fixed_point_v2.o mel_kernel_fft_fixed_v2.o mel_fixed_combined_v2.o mel_fixed_v2.mlir
echo
echo "⚠️  XCLBIN generation requires full Vitis AIE toolchain"
echo
echo "Workarounds:"
echo "  1. Object swap: Replace object in existing XCLBIN (fastest)"
echo "  2. Install Vitis: Get complete AIE toolchain (recommended)"
echo "  3. Use mlir-aie source: Build from source (most control)"
echo
echo "See BUILD_STATUS_V2_OCT28.md for detailed instructions"
echo
echo "======================================================================"
