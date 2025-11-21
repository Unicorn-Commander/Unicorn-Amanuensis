# Quick NPU API Reference - AMD Phoenix

## The 3-Line Fix for Loading XCLBINs

**WRONG** ❌:
```python
device.load_xclbin("kernel.xclbin")  # Returns "Operation not supported"
```

**RIGHT** ✅:
```python
xclbin_obj = xrt.xclbin("kernel.xclbin")
uuid = xclbin_obj.get_uuid()
device.register_xclbin(xclbin_obj)
```

## Complete Minimal Example

```python
import pyxrt as xrt
import numpy as np

# 1. Load XCLBIN
device = xrt.device(0)
xclbin_obj = xrt.xclbin("my_kernel.xclbin")
uuid = xclbin_obj.get_uuid()
device.register_xclbin(xclbin_obj)

# 2. Create context and kernel
hw_ctx = xrt.hw_context(device, uuid)
kernel = xrt.kernel(hw_ctx, "MLIR_AIE")

# 3. Load instructions
with open("insts.bin", "rb") as f:
    insts = f.read()

# 4. Allocate buffers
buffer_size = 2048  # 1024 elements * 2 bytes (BF16)
bo_instr = xrt.bo(device, len(insts), xrt.bo.flags.cacheable, kernel.group_id(1))
bo_input = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(3))
bo_output = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(4))

# 5. Write instructions
bo_instr.write(insts, 0)
bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

# 6. Execute kernel
input_data = np.random.randn(1024).astype(np.float32).tobytes()[:buffer_size]
bo_input.write(input_data, 0)
bo_input.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

run = kernel(3, bo_instr, len(insts), bo_input, bo_output)  # opcode=3
run.wait()

# 7. Read output
bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
output_data = bo_output.read(buffer_size, 0).tobytes()

print(f"✅ Executed! Got {len(output_data)} bytes")
```

## Buffer Group IDs

| Group ID | Purpose | Flags | Description |
|----------|---------|-------|-------------|
| 1 | Instructions | `cacheable` | Instruction sequence for NPU |
| 3 | Input | `host_only` | Input data buffer |
| 4 | Output | `host_only` | Output data buffer |
| 5 | Input B | `host_only` | Second input (for matmul) |

## Kernel Execution Arguments

**Single input operation** (softmax, gelu, layernorm):
```python
run = kernel(3, bo_instr, len(insts), bo_input, bo_output)
```

**Dual input operation** (matrix multiply):
```python
run = kernel(3, bo_instr, len(insts), bo_input_A, bo_input_B, bo_output)
```

## Data Conversion Helpers

### BF16 ↔ Float32

```python
import struct

def float_to_bf16(floats):
    """Float32 array to BF16 bytes"""
    result = bytearray(len(floats) * 2)
    for i, val in enumerate(floats):
        bits = struct.unpack('I', struct.pack('f', val))[0]
        upper = (bits >> 16) & 0xFFFF
        struct.pack_into('H', result, i*2, upper)
    return bytes(result)

def bf16_to_float(bf16_bytes):
    """BF16 bytes to Float32 array"""
    result = np.zeros(len(bf16_bytes) // 2, dtype=np.float32)
    for i in range(len(result)):
        upper = struct.unpack('H', bf16_bytes[i*2:(i+1)*2])[0]
        result[i] = struct.unpack('f', struct.pack('I', upper << 16))[0]
    return result
```

## Sync Directions

```python
# To NPU (before kernel execution)
bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

# From NPU (after kernel execution)
bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
```

## Error Handling

```python
try:
    device = xrt.device(0)
    xclbin_obj = xrt.xclbin(xclbin_path)
    uuid = xclbin_obj.get_uuid()
    device.register_xclbin(xclbin_obj)
    hw_ctx = xrt.hw_context(device, uuid)
    kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
except Exception as e:
    print(f"NPU initialization failed: {e}")
    # Fall back to CPU implementation
```

## Performance Measurement

```python
import time

# Measure execution time
start = time.perf_counter()
run = kernel(3, bo_instr, len(insts), bo_input, bo_output)
run.wait()
end = time.perf_counter()

exec_time_ms = (end - start) * 1000
print(f"Kernel execution: {exec_time_ms:.3f} ms")
```

## Multiple Iterations

```python
times = []
for i in range(10):
    bo_input.write(input_data, 0)
    bo_input.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    start = time.perf_counter()
    run = kernel(3, bo_instr, len(insts), bo_input, bo_output)
    run.wait()
    end = time.perf_counter()
    times.append(end - start)

    bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

avg_time = np.mean(times) * 1000
print(f"Average: {avg_time:.3f} ms")
```

## Reusing Buffers (Efficient)

```python
# Allocate once
bo_instr = xrt.bo(device, len(insts), xrt.bo.flags.cacheable, kernel.group_id(1))
bo_input = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(3))
bo_output = xrt.bo(device, buffer_size, xrt.bo.flags.host_only, kernel.group_id(4))

# Write instructions once
bo_instr.write(insts, 0)
bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

# Reuse for multiple executions
for batch in batches:
    bo_input.write(batch, 0)
    bo_input.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    run = kernel(3, bo_instr, len(insts), bo_input, bo_output)
    run.wait()

    bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    result = bo_output.read(buffer_size, 0).tobytes()
    process_result(result)
```

## Working Test Scripts

Copy any of these as templates:
```bash
kernels_xdna1/test_softmax.py       # Simplest example
kernels_xdna1/test_gelu.py          # With reference implementation
kernels_xdna1/test_matmul.py        # Dual-input example
kernels_xdna1/benchmark_all_kernels.py  # Multi-kernel class pattern
```

## Quick Test Command

```bash
cd /path/to/whisper_encoder_kernels
python3 test_your_xclbin.py <xclbin> <insts> [elements]
```

## What Changed from Xilinx Docs

| Old (doesn't work) | New (works) |
|--------------------|-------------|
| `device.load_xclbin(path)` | `device.register_xclbin(xclbin_obj)` |
| One-step loading | Two-step: create + register |
| String path | xrt.xclbin object |

## Why You Got "Operation not supported"

The `device.load_xclbin(path)` method:
- Is for older XRT versions
- Not implemented for Phoenix NPU
- Returns errno -EOPNOTSUPP (Operation not supported)

The `device.register_xclbin(obj)` method:
- Required for XRT 2.20.0 on Phoenix NPU
- Takes xclbin object not path
- Works with MLIR-AIE generated XCLBINs

## Device Information

Check your NPU:
```bash
/opt/xilinx/xrt/bin/xrt-smi examine
```

Should show:
```
Device: /dev/accel/accel0
Name: RyzenAI-npu1
Status: Running
Firmware: 1.5.5.391
```

## Troubleshooting

**"Operation not supported"**:
- ✅ Use `register_xclbin()` not `load_xclbin()`

**"Kernel not found"**:
- ✅ Use kernel name "MLIR_AIE" for MLIR-compiled kernels

**"Invalid group ID"**:
- ✅ Use group_id(1) for instructions, group_id(3) for input, group_id(4) for output

**Zeros in output**:
- ✅ Check kernel C code computation
- ✅ Verify input data format (BF16 vs FP32)
- ✅ Check instruction sequence generation

## References

- Full guide: `WORKING_NPU_RUNTIME_API.md`
- Test script: `test_your_xclbin.py`
- Working examples: `kernels_xdna1/test_*.py`

---

**Platform**: AMD Phoenix NPU, XRT 2.20.0
**Date**: November 21, 2025
