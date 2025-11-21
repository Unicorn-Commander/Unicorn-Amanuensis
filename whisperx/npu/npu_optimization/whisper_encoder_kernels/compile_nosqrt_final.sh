#!/bin/bash
set -e

# Setup environment
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PYTHONPATH=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build_final:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

# Create build directory
mkdir -p build_layernorm_nosqrt
cd build_layernorm_nosqrt

# Copy files
cp ../kernels_xdna1/layernorm_512_nosqrt.o .
cp ../test_nosqrt_ln.mlir .

# Run aiecc.py
echo "Starting compilation..."
/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
  --alloc-scheme=basic-sequential \
  --aie-generate-xclbin \
  --no-xchesscc \
  --no-xbridge \
  test_nosqrt_ln.mlir

echo "Done! XCLBIN should be generated."
ls -lh *.xclbin 2>/dev/null || echo "No XCLBIN found yet..."
