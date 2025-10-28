#!/bin/bash
# Build MEL kernel with INFINITE LOOP (event-driven execution)
# This is the correct AIE execution model!

set -e

echo "======================================================================"
echo "MEL Kernel with Infinite Loop - Event-Driven Execution"
echo "======================================================================"
echo

# Setup paths
PEANO=/home/ucadmin/mlir-aie-source/ironenv/lib/python3.13/site-packages/llvm-aie/bin/clang++
AIECC=/home/ucadmin/mlir-aie-source/build/bin/aiecc.py
PYTHON=/home/ucadmin/mlir-aie-source/ironenv/bin/python

WORK_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels
BUILD_DIR=$WORK_DIR/build_loop

cd $WORK_DIR

echo "Step 1: Clean and setup build directory..."
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR
echo "✅ Build directory ready"
echo

echo "Step 2: Compile C kernel (NO main, just pure computation)..."
$PEANO -O2 --target=aie2-none-unknown-elf \
    -c ../mel_kernel_simple.c -o mel_kernel_simple.o
echo "✅ C kernel compiled: $(stat -c%s mel_kernel_simple.o) bytes"
echo

echo "Step 3: Copy MLIR with infinite loop..."
cp ../mel_with_loop.mlir .
echo "✅ MLIR copied"
echo

echo "Step 4: Compile to XCLBIN with aiecc.py..."
echo "   This does: MLIR lowering → CDO generation → PDI → XCLBIN"
$PYTHON $AIECC \
    --no-aiesim \
    --aie-generate-npu-insts \
    --aie-generate-xclbin \
    --no-compile-host \
    --xclbin-name=mel_with_loop.xclbin \
    --npu-insts-name=insts.txt \
    mel_with_loop.mlir

echo
echo "✅ Compilation complete!"
echo

echo "Step 5: Validation..."
ls -lh mel_with_loop.xclbin insts.txt 2>/dev/null || ls -lh *.xclbin
file mel_with_loop.xclbin 2>/dev/null || file *.xclbin
echo

echo "======================================================================"
echo "✅ MEL KERNEL WITH LOOP BUILD COMPLETE!"
echo "======================================================================"
echo
echo "Key Features:"
echo "  ✅ Infinite loop in MLIR core body"
echo "  ✅ ObjectFIFO acquire/release synchronization"
echo "  ✅ C kernel is pure computation (no loop, no main)"
echo "  ✅ Event-driven execution triggered by DMA"
echo
echo "Next: Test with test_mel_with_loop.py"
echo
