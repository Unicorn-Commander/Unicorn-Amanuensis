#!/bin/bash
set -e

echo "=================================================="
echo "Compiling 32x32 Matmul Kernel (Final Method)"
echo "=================================================="
echo

cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels

echo "Step 1: Compiling C kernel..."
/home/ucadmin/.local/lib/python3.13/site-packages/llvm-aie/bin/clang \
    --target=aie2 \
    -I/home/ucadmin/.local/lib/python3.13/site-packages/llvm-aie/../aie_kernels/aie2/include \
    -c matmul_int8_32x32.c \
    -o matmul_32x32.o
echo "✅ C kernel compiled"
echo

echo "Step 2: Creating archive..."
/home/ucadmin/.local/lib/python3.13/site-packages/llvm-aie/bin/llvm-ar rcs \
    build_matmul_32x32/matmul_combined.o \
    matmul_32x32.o
echo "✅ Archive created"
echo

echo "Step 3: Copying to project directory..."
mkdir -p build_matmul_32x32/matmul_32x32.mlir.prj
cp build_matmul_32x32/matmul_combined.o build_matmul_32x32/matmul_32x32.mlir.prj/main_input.o
echo "✅ Object staged"
echo

echo "Step 4: Generating XCLBIN..."
cd build_matmul_32x32
export PATH=/opt/xilinx/xrt/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:/usr/bin:/bin:$PATH

/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/python3 \
    /home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=matmul_32x32.xclbin \
    --npu-insts-name=insts_32x32.bin \
    ../matmul_32x32.mlir

echo
echo "=================================================="
echo "✅ 32x32 Matmul Kernel Compiled!"
echo "=================================================="
echo
ls -lh matmul_32x32.xclbin insts_32x32.bin
echo
