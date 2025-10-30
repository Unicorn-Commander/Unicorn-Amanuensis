# Multi-Kernel Runtime Quick Reference

**For**: Developers working with XDNA2 Whisper encoder
**Last Updated**: October 30, 2025

---

## Available Kernels

| Kernel Name | Dimensions | File | Use Case |
|-------------|------------|------|----------|
| 512×512×512 | M=512, K=512, N=512 | `matmul_4tile_int8.xclbin` | Attention projections |
| 512×512×2048 | M=512, K=512, N=2048 | `matmul_4tile_int8_512x512x2048.xclbin` | FFN fc1 expansion |
| 512×2048×512 | (chunked) | Uses 512×512×512 4× | FFN fc2 projection |

---

## How It Works

### Automatic Kernel Selection

The runtime automatically selects the best kernel based on matrix dimensions:

```python
from runtime.whisper_xdna2_runtime import create_runtime

# Create runtime (loads all kernels)
runtime = create_runtime(model_size="base", use_4tile=True)

# Automatically selects 512×512×512 kernel
runtime._run_matmul_npu(A, B, 512, 512, 512)

# Automatically selects 512×512×2048 kernel
runtime._run_matmul_npu(A, B, 512, 512, 2048)

# Automatically chunks using 512×512×512 kernel (4×)
runtime._run_matmul_npu(A, B, 512, 2048, 512)
```

**No user code changes needed!**

### Selection Logic

```
Input: M, K, N dimensions

1. Check for exact kernel match (e.g., "512x512x2048")
   → If found: Use dedicated kernel
   → Else: Go to step 2

2. Check if K > 512 and "512x512x512" available
   → If yes: Chunk K dimension into 512-sized pieces
   → Else: Go to step 3

3. No match
   → Raise error (fail fast)
```

---

## K-Dimension Chunking

### What is it?

When K dimension exceeds 512, the runtime automatically:
1. Splits K into chunks of 512
2. Executes multiple 512×512×512 matmuls
3. Accumulates results in int32

### Example: 512×2048×512

```python
# User code (simple)
C = runtime._run_matmul_npu(A, B, 512, 2048, 512)

# What happens internally (automatic)
C = np.zeros((512, 512), dtype=np.int32)
for i in range(4):  # 2048 ÷ 512 = 4 chunks
    A_chunk = A[:, i*512:(i+1)*512]    # (512, 512)
    B_chunk = B[i*512:(i+1)*512, :]    # (512, 512)
    C += matmul_512x512x512(A_chunk, B_chunk)
return C
```

### Properties

- **Mathematically identical** to single kernel
- **Zero precision loss** (int32 accumulation)
- **4× overhead** for 4 chunks (still faster than CPU)
- **Automatic** (no user code changes)

---

## Compilation

### Quick Build

```bash
# Navigate to kernel directory
cd /home/ccadmin/CC-1L/kernels/common

# Compile 512×512×2048 kernel
./build_512x512x2048.sh

# Output:
# - build/matmul_4tile_int8_512x512x2048.xclbin (23 KB)
# - build/insts_4tile_int8_512x512x2048.bin (660 bytes)
```

### Manual Build

```bash
# Activate environment
source ~/mlir-aie/ironenv/bin/activate
export PYTHONPATH="/opt/xilinx/xrt/python:$PYTHONPATH"
export PEANO_INSTALL_DIR=~/mlir-aie/ironenv/lib/python3.13/site-packages/llvm-aie

# Generate MLIR
python3 matmul_iron_xdna2_4tile_int8.py --M 512 --K 512 --N 2048 --verbose > \
    build/matmul_4tile_int8_512x512x2048.mlir

# Compile to XCLBin
cd build
aiecc.py \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --no-compile-host \
    --xclbin-name=matmul_4tile_int8_512x512x2048.xclbin \
    --aie-generate-npu-insts \
    --npu-insts-name=insts_4tile_int8_512x512x2048.bin \
    --no-xchesscc \
    --no-xbridge \
    --peano ${PEANO_INSTALL_DIR} \
    matmul_4tile_int8_512x512x2048.mlir
```

---

## Performance

### Latency Estimates

| Operation | Dimensions | Kernel | Latency | Notes |
|-----------|------------|--------|---------|-------|
| Attention Q/K/V/O | 512×512×512 | Dedicated | ~55ms | Per projection |
| FFN fc1 | 512×512×2048 | Dedicated | ~55ms | Optimized |
| FFN fc2 | 512×2048×512 | Chunked 4× | ~220ms | 4× overhead |

### Full Encoder (6 layers)

- **Single layer**: ~495ms (4 × 55ms + 55ms + 4 × 55ms)
- **6 layers**: ~3 seconds
- **Audio**: 15 seconds (512 frames)
- **Realtime factor**: **5× realtime** (conservative)

---

## Testing

### Without Hardware (Logic Validation)

```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2
python3 test_kernel_selection.py
```

Expected output:
```
✅ 512x512x512: Attention Q/K/V/O projections
   Kernel: 512x512x512
   Chunked: No

✅ 512x512x2048: FFN fc1 expansion
   Kernel: 512x512x2048
   Chunked: No

✅ 512x2048x512: FFN fc2 projection (chunked)
   Kernel: 512x512x512
   Chunked: Yes (4 chunks)
```

### With Hardware (NPU Testing)

```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2
python3 test_multi_kernel.py
```

Tests:
1. 512×512×512 accuracy (should be 100%)
2. 512×512×2048 accuracy (should be 100%)
3. 512×2048×512 chunked accuracy (should be 100%)

---

## Troubleshooting

### Error: "No module named 'aie'"

**Cause**: XRT Python bindings not in PYTHONPATH

**Solution**:
```bash
export PYTHONPATH="/opt/xilinx/xrt/python:$PYTHONPATH"
```

### Error: "No kernel for dimensions XXX"

**Cause**: Requested dimensions not supported by any kernel

**Solutions**:
1. Check if dimensions are multiples of 512
2. Ensure 512×512×512 kernel is compiled
3. Add new kernel variant if needed

### Warning: "Kernel not found: XXX - skipping"

**Cause**: Kernel file missing from build directory

**Solution**: Compile the kernel:
```bash
cd /home/ccadmin/CC-1L/kernels/common
./build_512x512x2048.sh
```

---

## API Reference

### Create Runtime

```python
from runtime.whisper_xdna2_runtime import create_runtime

runtime = create_runtime(model_size="base", use_4tile=True)
```

**Returns**: `WhisperXDNA2Runtime` instance with all kernels loaded

### Run Matmul

```python
C = runtime._run_matmul_npu(A, B, M, K, N)
```

**Parameters**:
- `A`: Input matrix (M × K, int8)
- `B`: Input matrix (K × N, int8)
- `M, K, N`: Matrix dimensions (int)

**Returns**: Output matrix (M × N, int32)

**Behavior**:
- Automatically selects best kernel
- Falls back to chunking if needed
- Raises error if dimensions unsupported

---

## Whisper Encoder Layer Operations

### Attention Block (4 matmuls)

```python
# Q, K, V, O projections all use 512×512×512
Q = runtime._run_matmul_npu(x, q_weight, 512, 512, 512)
K = runtime._run_matmul_npu(x, k_weight, 512, 512, 512)
V = runtime._run_matmul_npu(x, v_weight, 512, 512, 512)
O = runtime._run_matmul_npu(attn, o_weight, 512, 512, 512)
```

**Kernel**: 512×512×512 (dedicated)

### FFN Block (2 matmuls)

```python
# fc1 uses dedicated 512×512×2048 kernel
fc1_out = runtime._run_matmul_npu(x, fc1_weight, 512, 512, 2048)

# fc2 uses chunked 512×512×512 kernel (4×)
fc2_out = runtime._run_matmul_npu(fc1_out, fc2_weight, 512, 2048, 512)
```

**Kernels**:
- fc1: 512×512×2048 (dedicated)
- fc2: 512×512×512 (chunked 4×)

---

## Files & Locations

### Kernel Binaries
```
/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/kernels/common/build/
├── matmul_4tile_int8.xclbin               # 512×512×512
├── insts_4tile_int8.bin
├── matmul_4tile_int8_512x512x2048.xclbin  # 512×512×2048
└── insts_4tile_int8_512x512x2048.bin
```

### Runtime Code
```
/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/runtime/
└── whisper_xdna2_runtime.py               # Multi-kernel runtime
```

### Build Scripts
```
/home/ccadmin/CC-1L/kernels/common/
├── build_512x512x2048.sh                  # Build fc1 kernel
└── matmul_iron_xdna2_4tile_int8.py        # MLIR generator
```

### Tests
```
/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/
├── test_kernel_selection.py               # Logic test (no hardware)
└── test_multi_kernel.py                   # Hardware test
```

---

## Best Practices

### 1. Let the Runtime Decide

**Don't**:
```python
if K == 512:
    use_kernel_512()
elif K == 2048:
    use_kernel_chunked()
```

**Do**:
```python
C = runtime._run_matmul_npu(A, B, M, K, N)
# Runtime handles kernel selection automatically
```

### 2. Keep Dimensions as Multiples of 512

For best performance, ensure:
- `M % 512 == 0` (or 256 for some kernels)
- `K % 512 == 0` (required for chunking)
- `N % 512 == 0` (or 32 for some kernels)

### 3. Monitor Kernel Usage

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Runtime will log which kernel is used:
# "NPU matmul (512x512x2048): 55.23ms, 24.3 GFLOPS"
# "NPU matmul (512x2048x512, 4 chunks): 220.15ms, 24.1 GFLOPS"
```

### 4. Validate Accuracy

Always validate NPU output vs CPU reference:
```python
C_cpu = A.astype(np.int32) @ B.astype(np.int32)
C_npu = runtime._run_matmul_npu(A, B, M, K, N)
assert np.array_equal(C_cpu, C_npu), "NPU output mismatch!"
```

---

## Quick Checklist

Before running encoder:
- [ ] XRT bindings available (`import aie.utils.xrt`)
- [ ] Kernel files exist in `build/` directory
- [ ] PYTHONPATH includes `/opt/xilinx/xrt/python`
- [ ] NPU device accessible
- [ ] Dimensions are multiples of required values

---

**Last Updated**: October 30, 2025
**Version**: 1.0 (Phase 2 Complete)
