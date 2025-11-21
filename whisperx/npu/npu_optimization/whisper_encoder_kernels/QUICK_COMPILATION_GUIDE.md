# QUICK MLIR-AIE COMPILATION GUIDE

One-page quick reference for compiling NPU kernels.

## Setup (One Time)

```bash
export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH
```

## Compile Existing Kernel

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels
bash compile_gelu.sh          # GELU activation
bash compile_matmul_32x32.sh  # Matrix multiply
bash compile_layernorm.sh     # Layer normalization
bash compile_attention_64x64.sh # Attention mechanism
bash kernels_xdna1/compile_softmax_bf16.sh # Softmax BF16
```

## Create New Kernel (5 Files)

### 1. C Kernel File (`mykernel.c`)

```c
void mykernel_func(int8_t *input, int8_t *output, int32_t size) {
    for (int i = 0; i < size; i++) {
        output[i] = (int8_t)(input[i] / 2);  // Example: divide by 2
    }
}
```

### 2. MLIR Wrapper (`mykernel.mlir`)

Copy from `gelu_simple.mlir` or `matmul_simple.mlir` and update:
- Function name: `@mykernel_func`
- Buffer sizes: `memref<Nxi8>`
- Output filename in compilation step

### 3. Compilation Script (`compile_mykernel.sh`)

```bash
#!/bin/bash
set -e

export PEANO_INSTALL_DIR=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/llvm-aie
export PYTHONPATH=/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/lib/python3.13/site-packages/aie:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PEANO_INSTALL_DIR/bin:/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin:$PATH

WORK_DIR=$(pwd)
BUILD_DIR=$WORK_DIR/build_mykernel
mkdir -p $BUILD_DIR
cd $WORK_DIR

# Compile C kernel
$PEANO_INSTALL_DIR/bin/clang -O2 -std=c11 --target=aie2-none-unknown-elf \
    -c mykernel.c -o $BUILD_DIR/mykernel.o

# Create archive
$PEANO_INSTALL_DIR/bin/llvm-ar rcs $BUILD_DIR/mykernel_combined.o $BUILD_DIR/mykernel.o

# Verify symbols
$PEANO_INSTALL_DIR/bin/llvm-nm $BUILD_DIR/mykernel_combined.o | grep mykernel || true

# Copy MLIR
cp mykernel.mlir $BUILD_DIR/mykernel.mlir

# Generate XCLBIN
cd $BUILD_DIR
/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --xclbin-name=mykernel.xclbin \
    --npu-insts-name=insts.bin \
    mykernel.mlir

cd $WORK_DIR
ls -lh $BUILD_DIR/mykernel.xclbin
echo "✅ Compilation complete!"
```

### 4. Test Script (`test_mykernel.py`)

```python
#!/usr/bin/env python3
import xrt
import numpy as np
import os

# Load XCLBIN
xclbin_path = "build_mykernel/mykernel.xclbin"
device = xrt.xrt_device(0)
device.load_xclbin(xclbin_path)

# Get kernel
kernel = device.get_kernel("mykernel")

# Create buffers
size = 512
input_data = np.array([i % 128 for i in range(size)], dtype=np.int8)
output_data = np.zeros(size, dtype=np.int8)

input_buf = xrt.xrt_bo(device, size, xrt.xrt_bo.cacheable_flags.normal, kernel.group_id(0))
output_buf = xrt.xrt_bo(device, size, xrt.xrt_bo.cacheable_flags.normal, kernel.group_id(1))

# Write input
input_buf.write(input_data)
input_buf.sync(xrt.xrt_bo.sync_direction.host2device)

# Execute
kernel(input_buf, output_buf)
device.wait()

# Read output
output_buf.sync(xrt.xrt_bo.sync_direction.device2host)
result = output_buf.read()

print(f"Input (first 10): {input_data[:10]}")
print(f"Output (first 10): {result[:10]}")
print("✅ Test complete!")
```

### 5. Run Compilation

```bash
chmod +x compile_mykernel.sh
bash compile_mykernel.sh
python3 test_mykernel.py
```

## File Structure for New Kernel

```
whisper_encoder_kernels/
├── mykernel.c                    # Kernel implementation
├── mykernel.mlir                 # MLIR wrapper
├── compile_mykernel.sh           # Compilation script
├── test_mykernel.py              # Test script
└── build_mykernel/               # Generated automatically
    ├── mykernel.o
    ├── mykernel_combined.o
    ├── mykernel.mlir
    ├── mykernel.xclbin           # Final executable
    └── insts.bin                 # Instruction sequence
```

## Troubleshooting

### Clang Not Found
```bash
# Check PEANO_INSTALL_DIR
ls $PEANO_INSTALL_DIR/bin/clang
# Should output: /path/to/clang
```

### aiecc.py Not Found
```bash
# Check PATH includes venv313/bin
echo $PATH | grep venv313
# Should show path in output
```

### Compilation Errors
```bash
# Check MLIR syntax
$PEANO_INSTALL_DIR/bin/mlir-opt mykernel.mlir
# Should output MLIR without errors
```

### XCLBIN Generation Failed
```bash
# Check build directory
ls -lh build_mykernel/
# Look for .o and .mlir files

# Re-run aiecc.py with debugging
cd build_mykernel
/home/ucadmin/mlir-aie-fresh/mlir-aie/venv313/bin/aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc \
    --no-xbridge \
    --verbose \
    --xclbin-name=mykernel.xclbin \
    --npu-insts-name=insts.bin \
    mykernel.mlir
```

## Working Examples Reference

| Kernel | Script | Files | Size | Perf |
|--------|--------|-------|------|------|
| GELU | `compile_gelu.sh` | 512-elem | <0.5ms | 512/0.5 = 1024 el/ms |
| MatMul 32x32 | `compile_matmul_32x32.sh` | 1024 elem | <1ms | Fast |
| Attention 64x64 | `compile_attention_64x64.sh` | 4096 bytes | 8-10ms | Tiled |
| LayerNorm | `compile_layernorm.sh` | 768 byte | ~1ms | Good |
| Softmax BF16 | `compile_softmax_bf16.sh` | 1024 elem | <1ms | Stable |

## Key Commands

```bash
# Verify clang works
$PEANO_INSTALL_DIR/bin/clang --version

# Verify aiecc.py works
aiecc.py --version

# List symbols in object file
$PEANO_INSTALL_DIR/bin/llvm-nm $BUILD_DIR/mykernel_combined.o

# Verify MLIR syntax
$PEANO_INSTALL_DIR/bin/mlir-opt mykernel.mlir

# Check XRT is working
xrt-smi examine

# Load XCLBIN (Python)
device = xrt.xrt_device(0)
device.load_xclbin("build_mykernel/mykernel.xclbin")
```

## Critical Points

1. **Device spec**: Always use `aie.device(npu1)` in MLIR (not npu1_4col)
2. **Tiles**: Use (0,0) for ShimNOC, (0,2) for compute
3. **Link object files**: Use `{link_with="mykernel_combined.o"}` in MLIR core
4. **Environment**: Export PEANO_INSTALL_DIR, PYTHONPATH, and PATH before compiling
5. **Compilation time**: Should be <1 second per XCLBIN

---

**Quick Reference Generated**: November 20, 2025
**For full details**: See AIETOOLS_AND_MLIR_COMPILATION_REPORT.md
