# Quick Start Guide: XDNA1/XDNA2 Development

**Date**: November 17, 2025
**Version**: 1.0
**Audience**: Developers new to the project

---

## 5-Minute Quick Start

### Prerequisites

```bash
# Verify NPU hardware is available
ls -l /dev/accel/accel0  # Should exist

# Verify XRT is installed
/opt/xilinx/xrt/bin/xrt-smi examine

# Verify Python environment
python3 -c "import xrt; print('XRT Python bindings OK')"
```

### Run Your First Kernel (XDNA1)

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels

# Test attention kernel
python3 -c "
from npu_attention_wrapper import NPUAttention
import numpy as np

attention = NPUAttention()
Q, K, V = [np.random.randint(-64, 64, (64, 64), dtype=np.int8) for _ in range(3)]
output = attention(Q, K, V, quantize=False)

print(f'✅ Attention working: {output.shape}, {np.count_nonzero(output)}/{output.size} non-zero')
"

# Test matmul kernel
python3 -c "
from npu_matmul_wrapper_batched import NPUMatmul
import numpy as np

matmul = NPUMatmul()
A = np.random.randint(-32, 32, (64, 64), dtype=np.int8)
B = np.random.randint(-32, 32, (64, 64), dtype=np.int8)
C = matmul(A, B, quantize=False)

print(f'✅ MatMul working: {C.shape}')
"
```

**Expected Output**:
```
✅ Attention working: (64, 64), 3661/4096 non-zero
✅ MatMul working: (64, 64)
```

---

## Understanding the Codebase

### Directory Structure

```
whisper_encoder_kernels/
├── docs/                               # 📚 You are here
│   ├── README.md                       # Documentation index
│   ├── QUICK_START_XDNA1_XDNA2.md     # This file
│   ├── XDNA1_XDNA2_ARCHITECTURE.md    # Architecture deep dive
│   ├── XDNA2_INTEGRATION_ROADMAP.md   # Integration timeline
│   ├── KERNEL_COMPARISON_XDNA1_XDNA2.md  # Performance comparison
│   ├── PORTABILITY_CHECKLIST.md       # Code review guidelines
│   └── PHASE1_XDNA2_INTEGRATION_ADDENDUM.md
│
├── kernels/                            # 🔧 Kernel implementations
│   ├── common/                         # C++ kernels (95% shared)
│   │   ├── attention_int8.c            # Attention computation
│   │   ├── matmul_int8.c               # Matrix multiply
│   │   ├── gelu_int8.c                 # GELU activation
│   │   └── layernorm_int8.c            # Layer normalization
│   │
│   ├── xdna1/                          # XDNA1 platform (4 columns)
│   │   ├── attention_xdna1.mlir        # 4-column MLIR
│   │   ├── matmul_xdna1.mlir
│   │   └── compile_xdna1.sh            # Build script
│   │
│   └── xdna2/                          # XDNA2 platform (8 columns)
│       ├── attention_xdna2.mlir        # 8-column MLIR (future)
│       ├── matmul_xdna2.mlir
│       └── compile_xdna2.sh            # Build script (future)
│
├── runtime/                            # 🐍 Python runtime wrappers
│   ├── common/                         # Shared runtime (95%)
│   │   ├── npu_base.py                 # Base NPU class
│   │   └── buffer_manager.py           # Buffer allocation
│   │
│   ├── xdna1/                          # XDNA1 runtime
│   │   ├── npu_attention_wrapper.py
│   │   └── npu_matmul_wrapper.py
│   │
│   └── xdna2/                          # XDNA2 runtime (future)
│       └── ...
│
├── build/                              # 📦 Compiled kernels
│   ├── xdna1/
│   │   ├── attention_64x64.xclbin      # Compiled NPU binary
│   │   └── matmul_32x32.xclbin
│   │
│   └── xdna2/                          # (future)
│
└── tests/                              # ✅ Test suites
    ├── common/                         # Shared tests
    │   ├── test_attention_accuracy.py
    │   └── benchmark_suite.py
    │
    ├── xdna1/
    └── xdna2/
```

### Key Components

**C++ Kernels** (where computation happens):
- Location: `kernels/common/*.c`
- Language: AIE C++ with vector intrinsics
- Portable: 100% shared between XDNA1/XDNA2
- Examples: attention_int8.c, matmul_int8.c

**MLIR Files** (hardware mapping):
- Location: `kernels/xdna1/*.mlir`, `kernels/xdna2/*.mlir`
- Purpose: Map C++ kernels to NPU tiles
- Different for XDNA1 (4 columns) vs XDNA2 (8 columns)
- 5% of total code

**Python Wrappers** (user interface):
- Location: `runtime/*/`
- Purpose: Easy-to-use Python API
- Handles buffer management, kernel loading
- 95% shared between platforms

**XCLBINs** (compiled binaries):
- Location: `build/xdna1/*.xclbin`
- Created by: `aie-opt` + `aie-translate`
- Format: Xilinx FPGA binary (for NPU)
- Load with XRT

---

## Common Tasks

### Task 1: Compile a Kernel (XDNA1)

```bash
cd kernels/xdna1

# Compile attention kernel
./compile_xdna1.sh attention

# Or manually:
aie-opt \
  --aie-canonicalize-device \
  --aie-objectFifo-stateful-transform \
  --aie-create-pathfinder-flows \
  attention_xdna1.mlir | \
aie-translate --aie-generate-xclbin \
  -o ../../build/xdna1/attention_64x64.xclbin

# Verify XCLBIN created
ls -lh ../../build/xdna1/attention_64x64.xclbin
```

**Expected Output**: `attention_64x64.xclbin` (~12-25 KB)

### Task 2: Test a Kernel

```bash
cd tests/xdna1

# Run single kernel test
python3 test_attention_accuracy.py

# Run full test suite
pytest test_xdna1_kernels.py -v

# Benchmark performance
python3 ../../benchmark_suite.py --kernel attention
```

### Task 3: Add a New Kernel

**Step 1**: Write C++ kernel in `kernels/common/`

```c
// kernels/common/my_kernel.c
void my_kernel_int8(int8_t* input, int8_t* output, int size) {
    for (int i = 0; i < size; i += 32) {
        // Vectorized operation
        v32int8 vec = *(v32int8*)&input[i];
        // ... computation ...
        *(v32int8*)&output[i] = result;
    }
}
```

**Step 2**: Create MLIR wrapper for XDNA1

```mlir
// kernels/xdna1/my_kernel_xdna1.mlir
module @my_kernel_xdna1 {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)

    // ObjectFIFO for input/output
    %fifo_in = aie.objectFifo.createObjectFifo(...)
    %fifo_out = aie.objectFifo.createObjectFifo(...)

    // Call C++ kernel
    aie.core(%tile_0_2) {
      func.call @my_kernel_int8(...) : (...) -> ()
      aie.end
    }
  }
}
```

**Step 3**: Compile and test

```bash
cd kernels/xdna1
./compile_xdna1.sh my_kernel

# Create Python wrapper
# (see existing wrappers for examples)

# Test
python3 tests/xdna1/test_my_kernel.py
```

### Task 4: Port Kernel to XDNA2

**Step 1**: Copy MLIR from XDNA1

```bash
cp kernels/xdna1/my_kernel_xdna1.mlir kernels/xdna2/my_kernel_xdna2.mlir
```

**Step 2**: Update device target

```diff
# kernels/xdna2/my_kernel_xdna2.mlir
module @my_kernel_xdna2 {
-  aie.device(npu1) {
+  aie.device(npu2) {
```

**Step 3**: Update tile mapping (if using multiple columns)

```diff
# XDNA1: 4 columns
%tile_0_2 = aie.tile(0, 2)
%tile_1_2 = aie.tile(1, 2)
%tile_2_2 = aie.tile(2, 2)
%tile_3_2 = aie.tile(3, 2)

# XDNA2: 8 columns
+%tile_4_2 = aie.tile(4, 2)
+%tile_5_2 = aie.tile(5, 2)
+%tile_6_2 = aie.tile(6, 2)
+%tile_7_2 = aie.tile(7, 2)
```

**Step 4**: Compile for XDNA2

```bash
cd kernels/xdna2
./compile_xdna2.sh my_kernel
```

**C++ kernel code**: No changes needed! (100% portable)

---

## Development Workflow

### Option 1: XDNA1 Development (Current Hardware)

```
1. Write C++ kernel (kernels/common/)
   ↓
2. Create XDNA1 MLIR (kernels/xdna1/)
   ↓
3. Compile to XCLBIN (build/xdna1/)
   ↓
4. Create Python wrapper (runtime/xdna1/)
   ↓
5. Test on NPU hardware (tests/xdna1/)
   ↓
6. Benchmark and optimize
```

**Timeline**: 1-2 days per kernel

### Option 2: XDNA2 Preparation (Future Hardware)

```
1. Copy MLIR from XDNA1 (kernels/xdna1/ → xdna2/)
   ↓
2. Update device target (npu1 → npu2)
   ↓
3. Update tile mapping (4 cols → 8 cols)
   ↓
4. Create XDNA2 wrapper (runtime/xdna2/)
   ↓
5. Mock testing (no hardware needed)
   ↓
6. Document expected performance
```

**Timeline**: 2-4 hours per kernel (no hardware testing)

---

## Testing Guide

### Unit Tests

```bash
# Test single kernel accuracy
python3 tests/common/test_attention_accuracy.py

# Expected: >95% correlation with CPU reference
```

### Integration Tests

```bash
# Test full encoder layer
python3 tests/xdna1/test_encoder_layer.py

# Tests attention + FFN + layer norm together
```

### Performance Benchmarks

```bash
# Benchmark suite
python3 benchmark_suite.py --kernel all --iterations 100

# Example output:
# Attention 64×64: 3.62 ± 0.15 ms
# MatMul 32×32: 0.46 ± 0.03 ms
# GELU 2048: 0.0013 ± 0.0001 ms
```

### Hardware Validation

```bash
# Verify NPU is accessible
python3 -c "
import xrt
device = xrt.xrt_device(0)
print(f'NPU detected: {device.get_info(xrt.info_device.name)}')
"

# Load and test XCLBIN
python3 -c "
import xrt
device = xrt.xrt_device(0)
uuid = device.load_xclbin('build/xdna1/attention_64x64.xclbin')
print(f'✅ XCLBIN loaded: {uuid}')
"
```

---

## Troubleshooting

### Problem: "No NPU device found"

**Solution**:
```bash
# Check device node exists
ls -l /dev/accel/accel0

# If missing, load kernel module
sudo modprobe amdxdna

# Verify with xrt-smi
/opt/xilinx/xrt/bin/xrt-smi examine
```

### Problem: "XCLBIN fails to load"

**Solution**:
```bash
# Check XCLBIN is for correct device
aie-dump build/xdna1/attention_64x64.xclbin

# Should show: device=npu1 (for XDNA1)

# Recompile if wrong device target
cd kernels/xdna1
./compile_xdna1.sh attention
```

### Problem: "Kernel returns all zeros"

**Possible Causes**:
1. DMA configuration incorrect
2. C++ kernel not linked
3. Buffer alignment issue

**Debug Steps**:
```bash
# 1. Check MLIR lowering
aie-opt --aie-canonicalize-device attention_xdna1.mlir -o lowered.mlir
cat lowered.mlir  # Inspect DMA config

# 2. Verify C++ kernel is compiled
nm build/xdna1/attention_64x64.xclbin | grep attention_int8
# Should show symbol

# 3. Test with passthrough kernel first
# (copy input → output without computation)
```

### Problem: "Performance worse than expected"

**Debug Steps**:
```bash
# Profile execution
python3 -c "
import time
from npu_attention_wrapper import NPUAttention
import numpy as np

attention = NPUAttention()
Q, K, V = [np.random.randint(-64, 64, (64, 64), dtype=np.int8) for _ in range(3)]

# Warm-up
attention(Q, K, V, quantize=False)

# Benchmark with profiling
times = []
for _ in range(100):
    start = time.perf_counter()
    output = attention(Q, K, V, quantize=False)
    times.append(time.perf_counter() - start)

import numpy as np
print(f'Mean: {np.mean(times)*1000:.2f} ms')
print(f'Std: {np.std(times)*1000:.2f} ms')
print(f'Min: {np.min(times)*1000:.2f} ms')
print(f'Max: {np.max(times)*1000:.2f} ms')
"

# Expected for attention 64×64: ~3.6 ms
# If much slower, check for:
# - Excessive DMA syncs
# - Python overhead
# - Buffer allocation in loop
```

---

## Best Practices

### 1. Write Portable C++ Kernels

**Good** (portable):
```c
// Uses standard AIE intrinsics
void kernel(int8_t* in, int8_t* out, int size) {
    for (int i = 0; i < size; i += 32) {
        v32int8 vec = *(v32int8*)&in[i];
        // ... computation ...
    }
}
```

**Bad** (hardcoded for XDNA1):
```c
// DON'T hardcode column count
#define NUM_COLUMNS 4  // ❌ Will break on XDNA2

// Instead, use runtime parameter or macro
```

### 2. Parameterize MLIR by Columns

**Good** (easy to port):
```mlir
// Use consistent tile naming
%tile_0_2 = aie.tile(0, 2)
%tile_1_2 = aie.tile(1, 2)
// Add more tiles for XDNA2 easily
```

**Bad** (hard to port):
```mlir
// Don't use hardcoded offsets
%tile_A = aie.tile(0, 2)
%tile_B = aie.tile(2, 4)  // ❌ Non-uniform layout
```

### 3. Test Early, Test Often

```bash
# Test after every change
python3 tests/common/test_attention_accuracy.py

# Benchmark after optimization
python3 benchmark_suite.py --kernel attention

# Compare before/after
echo "Before: 3.62 ms" > before.txt
./optimize_kernel.sh
python3 benchmark_suite.py --kernel attention > after.txt
diff before.txt after.txt
```

### 4. Document Performance Expectations

```python
# In your wrapper code
class NPUAttention:
    """
    XDNA1 Performance:
    - 64×64 single tile: 3.62 ms
    - 4-column parallel: ~1.2 ms (est.)

    XDNA2 Performance (projected):
    - 64×64 single tile: 3.62 ms (same)
    - 8-column parallel: ~0.65 ms (1.85x)
    """
```

---

## Useful Commands Reference

### Compilation

```bash
# Compile MLIR to XCLBIN
aie-opt --aie-canonicalize-device kernel.mlir | \
aie-translate --aie-generate-xclbin -o kernel.xclbin

# Inspect XCLBIN
aie-dump kernel.xclbin

# Lower MLIR (debug)
aie-opt --aie-canonicalize-device \
        --aie-objectFifo-stateful-transform \
        kernel.mlir -o lowered.mlir
```

### XRT Commands

```bash
# List NPU devices
/opt/xilinx/xrt/bin/xrt-smi examine

# Reset NPU
/opt/xilinx/xrt/bin/xrt-smi reset

# Monitor NPU utilization
watch -n 1 '/opt/xilinx/xrt/bin/xrt-smi examine'
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/common/test_attention_accuracy.py::test_attention_correlation -v

# Benchmark
python3 benchmark_suite.py --kernel all --output results.json
```

---

## Next Steps

### For XDNA1 Development

1. Read: [PHASE1_DAY1_EXECUTIVE_SUMMARY.md](../PHASE1_DAY1_EXECUTIVE_SUMMARY.md)
2. Try: Run existing kernels (see 5-minute quick start above)
3. Optimize: Multi-column parallelism (see [XDNA2_INTEGRATION_ROADMAP.md](./XDNA2_INTEGRATION_ROADMAP.md) Phase 3)
4. Learn: Study existing kernels in `kernels/common/`

### For XDNA2 Preparation

1. Read: [XDNA1_XDNA2_ARCHITECTURE.md](./XDNA1_XDNA2_ARCHITECTURE.md)
2. Review: [KERNEL_COMPARISON_XDNA1_XDNA2.md](./KERNEL_COMPARISON_XDNA1_XDNA2.md)
3. Plan: [XDNA2_INTEGRATION_ROADMAP.md](./XDNA2_INTEGRATION_ROADMAP.md)
4. Prepare: Create XDNA2 MLIR variants (no hardware needed)

---

## Support and Resources

### Documentation

- Architecture: `docs/XDNA1_XDNA2_ARCHITECTURE.md`
- Roadmap: `docs/XDNA2_INTEGRATION_ROADMAP.md`
- Comparison: `docs/KERNEL_COMPARISON_XDNA1_XDNA2.md`
- Checklist: `docs/PORTABILITY_CHECKLIST.md`

### Examples

- Attention kernel: `kernels/common/attention_int8.c`
- MatMul kernel: `kernels/common/matmul_int8.c`
- XDNA1 MLIR: `kernels/xdna1/*.mlir`
- Runtime wrapper: `runtime/xdna1/npu_attention_wrapper.py`

### Tools

- MLIR-AIE compiler: `aie-opt`, `aie-translate`
- XRT runtime: `/opt/xilinx/xrt/`
- Python bindings: `import xrt`

---

**Document Version**: 1.0
**Last Updated**: November 17, 2025
**Next Review**: After Phase 1 completion
**Maintained By**: NPU Documentation Team
