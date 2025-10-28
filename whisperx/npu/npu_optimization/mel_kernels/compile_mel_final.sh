#!/bin/bash
set -e

cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/build_fft

# Activate environment
source /home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/activate

# Set environment exactly like ResNet example + XRT
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:$PATH

echo "=== MEL Kernel with FFT Compilation ==="
echo ""

# Step 1: Verify tools
echo "✅ C kernel: $(ls -lh mel_kernel_combined.o | awk '{print $9,$5}')"
echo "✅ xclbinutil: $(which xclbinutil)"
echo "✅ aiecc.py: $(which aiecc.py)"

# Step 2: Run aiecc.py
echo ""
echo "=== Running aiecc.py ==="
aiecc.py \
  --alloc-scheme=basic-sequential \
  --aie-generate-xclbin \
  --aie-generate-npu-insts \
  --no-compile-host \
  --no-xchesscc \
  --no-xbridge \
  --xclbin-name=mel_fft_final.xclbin \
  --npu-insts-name=insts_fft.bin \
  mel_with_fft.mlir

echo ""
echo "🎉 SUCCESS! Files generated:"
ls -lh mel_fft_final.xclbin insts_fft.bin
