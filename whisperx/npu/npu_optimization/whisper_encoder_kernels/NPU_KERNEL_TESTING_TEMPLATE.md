# NPU Kernel Testing Template

**Purpose**: Standard template for testing any NPU kernel on AMD Phoenix NPU
**Based on**: Working matmul_16x16 and fixed attention_64x64 kernels
**Date**: October 30, 2025

---

## Template Code

```python
#!/usr/bin/env python3
"""
NPU Kernel Test Template
Replace KERNEL_NAME, INPUT_SIZE, OUTPUT_SIZE with your values
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
import time
from pathlib import Path

# ============================================================================
# CONFIGURATION - CUSTOMIZE FOR YOUR KERNEL
# ============================================================================

KERNEL_NAME = "your_kernel"  # e.g., "attention_64x64", "matmul_16x16"
XCLBIN_PATH = f"{KERNEL_NAME}.xclbin"
INSTS_PATH = f"build_{KERNEL_NAME}/insts.bin"  # or main_sequence.bin

INPUT_SIZE = 12288   # Bytes (customize for your kernel)
OUTPUT_SIZE = 4096   # Bytes (customize for your kernel)

# ============================================================================
# STEP 1: LOAD XCLBIN AND INSTRUCTIONS
# ============================================================================

def load_kernel():
    """Load XCLBIN and instruction sequence"""

    print("="*70)
    print(f"Loading NPU Kernel: {KERNEL_NAME}")
    print("="*70)
    print()

    # Check files exist
    if not Path(XCLBIN_PATH).exists():
        print(f"❌ ERROR: XCLBIN not found at {XCLBIN_PATH}")
        sys.exit(1)
    if not Path(INSTS_PATH).exists():
        print(f"❌ ERROR: Instructions not found at {INSTS_PATH}")
        sys.exit(1)

    # Initialize device
    device = xrt.device(0)
    print(f"✅ NPU device: /dev/accel/accel0")

    # Load XCLBIN
    xclbin = xrt.xclbin(XCLBIN_PATH)
    uuid = xclbin.get_uuid()
    device.register_xclbin(xclbin)
    print(f"✅ XCLBIN loaded: {XCLBIN_PATH}")

    # Create hardware context
    hw_ctx = xrt.hw_context(device, uuid)
    kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
    print(f"✅ Kernel found: MLIR_AIE")

    # Load instructions (CRITICAL!)
    with open(INSTS_PATH, "rb") as f:
        insts = f.read()
    n_insts = len(insts)
    print(f"✅ Instructions loaded: {n_insts} bytes")
    print()

    return device, kernel, insts, n_insts

# ============================================================================
# STEP 2: ALLOCATE BUFFERS (GROUP IDs: 1, 3, 4)
# ============================================================================

def allocate_buffers(device, kernel, n_insts, insts):
    """Allocate instruction, input, and output buffers"""

    print("Allocating NPU buffers...")

    # INSTRUCTION BUFFER (group_id 1, cacheable)
    instr_bo = xrt.bo(device, n_insts,
                      xrt.bo.flags.cacheable,
                      kernel.group_id(1))
    instr_bo.write(insts, 0)
    instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                  n_insts, 0)
    print(f"  ✅ Instruction buffer: {n_insts} bytes (group_id 1)")

    # INPUT BUFFER (group_id 3, host_only)
    input_bo = xrt.bo(device, INPUT_SIZE,
                      xrt.bo.flags.host_only,
                      kernel.group_id(3))
    print(f"  ✅ Input buffer: {INPUT_SIZE} bytes (group_id 3)")

    # OUTPUT BUFFER (group_id 4, host_only)
    output_bo = xrt.bo(device, OUTPUT_SIZE,
                       xrt.bo.flags.host_only,
                       kernel.group_id(4))
    print(f"  ✅ Output buffer: {OUTPUT_SIZE} bytes (group_id 4)")
    print()

    return instr_bo, input_bo, output_bo

# ============================================================================
# STEP 3: RUN KERNEL (OPCODE 3)
# ============================================================================

def run_kernel(kernel, instr_bo, n_insts, input_bo, output_bo):
    """Execute kernel on NPU"""

    # CRITICAL: Use opcode 3 and pass all 5 arguments
    opcode = 3
    run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)

    # Wait for completion (1 second timeout)
    state = run.wait(1000)

    # Check state
    if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        raise RuntimeError(f"Kernel execution failed: state {state}")

    return True

# ============================================================================
# STEP 4: BENCHMARK PERFORMANCE
# ============================================================================

def benchmark_kernel(kernel, instr_bo, n_insts, input_bo, output_bo,
                     input_data, num_iterations=10):
    """Run performance benchmark"""

    print("Running performance benchmark...")

    # Write input data
    input_bo.write(input_data.tobytes(), 0)
    input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                  INPUT_SIZE, 0)

    # Warmup (3 iterations)
    print("  Warming up...")
    for i in range(3):
        run_kernel(kernel, instr_bo, n_insts, input_bo, output_bo)
    print("  ✅ Warmup complete")

    # Benchmark
    print(f"  Benchmarking ({num_iterations} iterations)...")
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        run_kernel(kernel, instr_bo, n_insts, input_bo, output_bo)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    # Statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print()
    print("="*70)
    print("PERFORMANCE RESULTS")
    print("="*70)
    print(f"  Average time: {avg_time:.3f} ms")
    print(f"  Std deviation: {std_time:.3f} ms")
    print(f"  Min time: {min_time:.3f} ms")
    print(f"  Max time: {max_time:.3f} ms")
    print()

    return avg_time

# ============================================================================
# STEP 5: READ AND VERIFY OUTPUT
# ============================================================================

def read_output(output_bo, output_shape, dtype=np.int8):
    """Read output from NPU"""

    print("Reading output from NPU...")

    # Sync from device
    output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
                   OUTPUT_SIZE, 0)

    # Read data
    output_data = np.frombuffer(output_bo.read(OUTPUT_SIZE, 0), dtype=dtype)
    output_matrix = output_data.reshape(output_shape)

    print(f"✅ Output shape: {output_matrix.shape}")
    print(f"  Range: [{output_matrix.min()}, {output_matrix.max()}]")
    print(f"  Mean: {output_matrix.mean():.2f}")
    print(f"  Non-zero: {np.count_nonzero(output_matrix)}/{output_matrix.size}")
    print()

    return output_matrix

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_kernel():
    """Main test function"""

    # Step 1: Load kernel
    device, kernel, insts, n_insts = load_kernel()

    # Step 2: Allocate buffers
    instr_bo, input_bo, output_bo = allocate_buffers(device, kernel,
                                                      n_insts, insts)

    # Step 3: Create test data (customize for your kernel)
    print("Generating test data...")
    np.random.seed(42)
    input_data = np.random.randint(-64, 64, INPUT_SIZE, dtype=np.int8)
    print(f"  Input: {input_data.shape}, range [{input_data.min()}, {input_data.max()}]")
    print()

    # Step 4: Benchmark
    avg_time = benchmark_kernel(kernel, instr_bo, n_insts,
                                input_bo, output_bo, input_data)

    # Step 5: Read output (customize output_shape for your kernel)
    output_shape = (64, 64)  # Example: 64x64 matrix
    output = read_output(output_bo, output_shape)

    # Step 6: Print summary
    print("="*70)
    print("TEST COMPLETE!")
    print("="*70)
    print(f"✅ Kernel: {KERNEL_NAME}")
    print(f"✅ Average time: {avg_time:.3f} ms")
    print(f"✅ Output generated successfully")
    print()

    return 0

# ============================================================================
# RUN TEST
# ============================================================================

if __name__ == "__main__":
    try:
        sys.exit(test_kernel())
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

---

## Critical Points

### 1. Buffer Group IDs (Phoenix NPU)

**MUST USE THESE VALUES**:

```python
# Instruction buffer
instr_bo = xrt.bo(device, n_insts,
                  xrt.bo.flags.cacheable,    # IMPORTANT: cacheable
                  kernel.group_id(1))        # MUST be 1

# Input buffer
input_bo = xrt.bo(device, INPUT_SIZE,
                  xrt.bo.flags.host_only,    # IMPORTANT: host_only
                  kernel.group_id(3))        # MUST be 3

# Output buffer
output_bo = xrt.bo(device, OUTPUT_SIZE,
                   xrt.bo.flags.host_only,   # IMPORTANT: host_only
                   kernel.group_id(4))       # MUST be 4
```

**DO NOT** use other group_id values on Phoenix NPU!

### 2. Kernel Call Signature

**MUST USE THIS FORMAT**:

```python
opcode = 3  # Standard opcode for NPU kernels
run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
#           ^^^^^^  ^^^^^^^^  ^^^^^^^  ^^^^^^^^  ^^^^^^^^^
#           arg1    arg2      arg3     arg4      arg5
```

**ALL 5 ARGUMENTS ARE REQUIRED!**

### 3. Instruction Binary

**ALWAYS** load and sync instruction buffer:

```python
# Load from file
with open(INSTS_PATH, "rb") as f:
    insts = f.read()

# Write to buffer
instr_bo.write(insts, 0)

# Sync to device (CRITICAL!)
instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
              n_insts, 0)
```

**Without instructions, kernel will fail with `ERT_CMD_STATE_ERROR`!**

---

## Customization Guide

### For Different Kernels

1. **Change kernel name**:
```python
KERNEL_NAME = "gelu_2048"  # or "layernorm_simple", etc.
```

2. **Update paths**:
```python
XCLBIN_PATH = "gelu_2048.xclbin"
INSTS_PATH = "build_gelu/insts.bin"
```

3. **Set buffer sizes**:
```python
INPUT_SIZE = 8192   # Your input size in bytes
OUTPUT_SIZE = 8192  # Your output size in bytes
```

4. **Customize test data**:
```python
# Example: GELU input (2048 elements)
input_data = np.random.randint(-128, 127, 2048, dtype=np.int8)

# Example: Attention input (Q+K+V combined)
Q = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
K = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
V = np.random.randint(-64, 64, (64, 64), dtype=np.int8)
input_data = np.concatenate([Q.flatten(), K.flatten(), V.flatten()])
```

5. **Set output shape**:
```python
# Example: Attention output (64x64)
output_shape = (64, 64)

# Example: GELU output (2048 elements)
output_shape = (2048,)

# Example: LayerNorm output (256 elements)
output_shape = (256,)
```

---

## Troubleshooting

### Error: `ERT_CMD_STATE_ERROR`

**Check**:
1. Is instruction buffer allocated? (`instr_bo`)
2. Is instruction buffer synced? (`instr_bo.sync(...)`)
3. Is kernel called with opcode? (`kernel(opcode, ...)`)
4. Are all 5 arguments passed?

### Error: "No compute units with connectivity"

**Check**:
1. Are you using correct group_id values? (1, 3, 4)
2. Are flags correct? (cacheable for 1, host_only for 3 and 4)

### Error: Timeout (kernel hangs)

**Check**:
1. Is XCLBIN file correct?
2. Is instruction binary correct?
3. Are buffer sizes correct?
4. Increase timeout: `run.wait(5000)` (5 seconds)

### Error: Wrong output

**Check**:
1. Is input data in correct format?
2. Is output shape correct?
3. Test with known inputs (e.g., zeros, identity matrix)

---

## Testing Checklist

Before running any NPU kernel:

- [ ] XCLBIN file exists and is correct
- [ ] Instruction binary exists (insts.bin or main_sequence.bin)
- [ ] NPU device accessible (`/dev/accel/accel0`)
- [ ] XRT 2.20.0 installed (`xrt-smi examine`)
- [ ] Instruction buffer allocated with group_id(1)
- [ ] Input buffer allocated with group_id(3)
- [ ] Output buffer allocated with group_id(4)
- [ ] Instruction buffer synced to device
- [ ] Kernel called with opcode (3)
- [ ] All 5 arguments passed to kernel
- [ ] Timeout sufficient (>1000ms)

---

## Example Kernels

### Matmul 16×16
```python
KERNEL_NAME = "matmul_16x16"
XCLBIN_PATH = "build_matmul_fixed/matmul_16x16.xclbin"
INSTS_PATH = "build_matmul_fixed/main_sequence.bin"
INPUT_SIZE = 512    # A(16x16) + B(16x16) = 512 bytes
OUTPUT_SIZE = 256   # C(16x16) = 256 bytes
output_shape = (16, 16)
```

### Attention 64×64
```python
KERNEL_NAME = "attention_64x64"
XCLBIN_PATH = "attention_64x64.xclbin"
INSTS_PATH = "build_attention_64x64/insts.bin"
INPUT_SIZE = 12288  # Q+K+V (3×64×64) = 12288 bytes
OUTPUT_SIZE = 4096  # Attention(64×64) = 4096 bytes
output_shape = (64, 64)
```

### GELU 2048
```python
KERNEL_NAME = "gelu_2048"
XCLBIN_PATH = "build_gelu/gelu_2048.xclbin"
INSTS_PATH = "build_gelu/insts.bin"
INPUT_SIZE = 2048   # 2048 elements
OUTPUT_SIZE = 2048  # 2048 elements
output_shape = (2048,)
```

### LayerNorm
```python
KERNEL_NAME = "layernorm_simple"
XCLBIN_PATH = "build_layernorm/layernorm_simple.xclbin"
INSTS_PATH = "build_layernorm/insts.bin"
INPUT_SIZE = 768    # Input + gamma + beta
OUTPUT_SIZE = 256   # Normalized output
output_shape = (256,)
```

---

## Summary

**Key Pattern for ALL Phoenix NPU kernels**:

1. Load instructions from file ✅
2. Allocate buffers with correct group_ids (1, 3, 4) ✅
3. Sync instruction buffer to device ✅
4. Call kernel with opcode and 5 arguments ✅
5. Check return state == COMPLETED ✅

**If you follow this template, your kernel WILL work!**

---

**Template by**: NPU Kernel Debug Team
**Date**: October 30, 2025
**Based on**: Successful debugging of attention_64x64 kernel
