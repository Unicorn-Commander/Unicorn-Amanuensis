#!/bin/bash
# Build MEL kernel with real FFT computation
# October 28, 2025 - FFT Integration Build

set -e  # Exit on error

echo "======================================================================"
echo "MEL Kernel with FFT - Build Pipeline"
echo "======================================================================"
echo

# Setup paths
PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
PEANO=$PEANO_INSTALL_DIR/bin/clang
PEANO_CXX=$PEANO_INSTALL_DIR/bin/clang++
PEANO_AR=$PEANO_INSTALL_DIR/bin/llvm-ar
AIE_OPT=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aie-opt
AIE_TRANSLATE=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aie-translate
XCLBINUTIL=/opt/xilinx/xrt/bin/xclbinutil

WORK_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels
BUILD_DIR=$WORK_DIR/build
TEMPLATE_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/build

cd $WORK_DIR

echo "Step 1: Compile FFT module..."
$PEANO -O2 -std=c11 --target=aie2-none-unknown-elf -c fft_real_simple.c -o fft_real.o
echo "✅ FFT module compiled: $(stat -c%s fft_real.o) bytes"
echo

echo "Step 2: Compile MEL kernel with FFT..."
$PEANO_CXX -O2 -std=c++20 --target=aie2-none-unknown-elf -c mel_kernel_fft.c -o mel_kernel_fft.o
echo "✅ MEL kernel compiled: $(stat -c%s mel_kernel_fft.o) bytes"
echo

echo "Step 3: Create combined object archive..."
$PEANO_AR rcs mel_kernel_combined.o mel_kernel_fft.o fft_real.o
echo "✅ Combined archive created: $(stat -c%s mel_kernel_combined.o) bytes"
echo

echo "Step 4: Compile MLIR to XCLBIN..."
cd $BUILD_DIR

# Use aiecc.py if available, or aie-opt + aie-translate manually
if command -v aiecc.py &> /dev/null; then
    echo "Using aiecc.py for compilation..."
    python3 /home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
        --sysroot=$PEANO_INSTALL_DIR \
        --host-target=aarch64-linux-gnu \
        $WORK_DIR/mel_with_fft.mlir \
        -I$WORK_DIR \
        -o mel_fft.xclbin
    echo "✅ XCLBIN compiled with aiecc.py: $(stat -c%s mel_fft.xclbin) bytes"
else
    echo "Manual compilation (aie-opt + aie-translate)..."

    # Phase 1: Lower MLIR
    $AIE_OPT --aie-canonicalize-device \
        --aie-objectFifo-stateful-transform \
        --aie-create-pathfinder-flows \
        --aie-assign-buffer-addresses \
        $WORK_DIR/mel_with_fft.mlir -o mel_fft_physical.mlir
    echo "✅ MLIR lowered: $(stat -c%s mel_fft_physical.mlir) bytes"

    # Phase 2: Generate NPU instructions
    $AIE_TRANSLATE --aie-generate-xaie mel_fft_physical.mlir -o insts.txt
    echo "✅ Instructions generated"

    # Phase 3: Generate CDO files
    $AIE_TRANSLATE --aie-generate-cdo mel_fft_physical.mlir
    echo "✅ CDO files generated"

    # Note: Full XCLBIN generation requires additional steps
    echo "⚠️  Manual XCLBIN packaging not yet implemented"
    echo "    Use aiecc.py or refer to build_mel_complete.sh for full packaging"
fi

cd $WORK_DIR
echo

echo "======================================================================"
echo "✅ FFT COMPILATION COMPLETE!"
echo "======================================================================"
echo
echo "Compiled Files:"
echo "  - fft_real.o             : $(stat -c%s fft_real.o) bytes - FFT implementation"
echo "  - mel_kernel_fft.o       : $(stat -c%s mel_kernel_fft.o) bytes - MEL kernel"
echo "  - mel_kernel_combined.o  : $(stat -c%s mel_kernel_combined.o) bytes - Combined archive"
echo
echo "Next Steps:"
echo "  1. Test compilation: ./build_mel_with_fft.sh"
echo "  2. If successful, use aiecc.py to generate complete XCLBIN"
echo "  3. Test on NPU hardware"
echo
