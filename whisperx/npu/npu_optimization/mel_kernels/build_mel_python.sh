#!/bin/bash
# Build MEL kernel using Python IRON API (the correct way!)
# Based on working matrix_transpose example from MLIR-AIE

set -e

echo "======================================================================"
echo "MEL Kernel Build - Python IRON API (Event-Driven Execution)"
echo "======================================================================"
echo

# Setup paths
PEANO=/home/ucadmin/mlir-aie-source/ironenv/lib/python3.13/site-packages/llvm-aie/bin/clang++
AIE_OPT=/home/ucadmin/mlir-aie-source/build/bin/aie-opt
AIECC=/home/ucadmin/mlir-aie-source/build/bin/aiecc.py
PYTHON=/home/ucadmin/mlir-aie-source/ironenv/bin/python

WORK_DIR=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels
BUILD_DIR=$WORK_DIR/build_python

cd $WORK_DIR

echo "Step 1: Clean build directory..."
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
echo "✅ Build directory created"
echo

echo "Step 2: Compile C kernel..."
$PEANO -O2 --target=aie2-none-unknown-elf \
    -c mel_kernel_simple.c -o $BUILD_DIR/mel_kernel_simple.o
echo "✅ C kernel compiled: $(stat -c%s $BUILD_DIR/mel_kernel_simple.o) bytes"
echo

echo "Step 3: Generate MLIR from Python..."
cd $BUILD_DIR
$PYTHON ../aie_mel_python.py > aie_mel.mlir
echo "✅ MLIR generated: $(stat -c%s aie_mel.mlir) bytes"
echo

echo "Step 4: Compile MLIR to XCLBIN (aiecc.py)..."
# This does everything: MLIR lowering, CDO generation, PDI, XCLBIN packaging
$PYTHON $AIECC \
    --no-aiesim \
    --aie-generate-npu-insts \
    --aie-generate-xclbin \
    --no-compile-host \
    --xclbin-name=mel_final.xclbin \
    --npu-insts-name=insts.txt \
    aie_mel.mlir

echo "✅ Compilation complete!"
echo

echo "Step 5: Validation..."
ls -lh mel_final.xclbin insts.txt
file mel_final.xclbin
echo

echo "======================================================================"
echo "✅ MEL KERNEL BUILD COMPLETE (Python IRON API)"
echo "======================================================================"
echo
echo "Generated Files:"
echo "  - mel_kernel_simple.o   : C kernel ELF"
echo "  - aie_mel.mlir          : Generated MLIR with infinite loop"
echo "  - mel_final.xclbin      : Complete NPU executable"
echo "  - insts.txt             : NPU instructions"
echo
echo "Key Difference: Core has INFINITE LOOP with lock synchronization!"
echo "Next: Test with test_mel_python.py"
echo
