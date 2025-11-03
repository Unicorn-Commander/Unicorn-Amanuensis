#!/bin/bash
set -e

echo "Recompiling with FIXED C kernel..."

cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels

# Create combined archive with FIXED object file
/home/ucadmin/.local/lib/python3.13/site-packages/llvm-aie/bin/llvm-ar rcs \
    build_attention_64x64/attention_combined_fixed.o \
    attention_int8_64x64_tiled_fixed.o

# Copy the combined archive to project directory  
cp build_attention_64x64/attention_combined_fixed.o build_attention_64x64/attention_64x64_fixed.mlir.prj/main_input.o

# Generate XCLBIN again
cd build_attention_64x64
export PATH=/opt/xilinx/xrt/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:/usr/bin:/bin:$PATH

/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/python3 \
    /home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=attention_64x64_fixed_v2.xclbin \
    --npu-insts-name=insts_fixed_v2.bin \
    attention_64x64_fixed.mlir

echo "Done! Generated attention_64x64_fixed_v2.xclbin"
ls -lh attention_64x64_fixed_v2.xclbin insts_fixed_v2.bin
