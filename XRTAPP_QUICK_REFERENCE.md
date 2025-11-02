# XRTApp Quick Reference

**Purpose**: Real XRT buffer operations for NPU-accelerated computation
**Location**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/server.py` (lines 215-456)
**Week**: 16 (November 2, 2025)

---

## Quick Start

### Initialize XRTApp

```python
from xdna2.server import load_xrt_npu_application

# Load XRT application with real buffers
npu_app = load_xrt_npu_application()
# Returns XRTApp instance with device, context, kernel loaded
```

### Register Buffers

```python
import numpy as np

# Register input buffer (100x200 float32 matrix)
npu_app.register_buffer(
    idx=0,                  # Buffer index (kernel argument)
    dtype=np.float32,       # Data type
    shape=(100, 200)        # Buffer dimensions
)
# Allocates 80,000 bytes on NPU

# Register output buffer
npu_app.register_buffer(idx=1, dtype=np.float32, shape=(100, 512))
```

### Write Data to Buffer

```python
# Create input data
input_data = np.random.randn(100, 200).astype(np.float32)

# Write to NPU buffer
npu_app.write_buffer(idx=0, data=input_data)
# Data is automatically synced to device
```

### Execute Kernel

```python
# Run kernel with all registered buffers
npu_app.run(
    input_buffers=[0],      # Input buffer indices (optional)
    output_buffers=[1]      # Output buffer indices (optional)
)
# Kernel executes on NPU and waits for completion
```

### Read Results

```python
# Read output from NPU
output_data = npu_app.read_buffer(idx=1)
# Returns numpy array with shape (100, 512)
# Data is automatically synced from device
```

### Cleanup (optional)

```python
# Release all buffers
npu_app.cleanup()
```

---

## Method Reference

### `register_buffer(idx, dtype, shape)`

Allocates real XRT buffer object on NPU.

**Parameters**:
- `idx` (int): Buffer index (0-based, maps to kernel argument)
- `dtype` (numpy dtype): Data type (e.g., np.float32, np.bfloat16)
- `shape` (tuple): Buffer dimensions (e.g., (1500, 512))

**Returns**: None

**Raises**: `RuntimeError` if allocation fails

**Example**:
```python
# Allocate 2MB buffer for matrix
npu_app.register_buffer(0, np.float32, (512, 1024))
```

**Memory Calculation**:
```python
size_bytes = np.prod(shape) * np.dtype(dtype).itemsize
# (512 × 1024) × 4 bytes = 2,097,152 bytes (2 MB)
```

---

### `write_buffer(idx, data)`

Writes data to XRT buffer and syncs to NPU device.

**Parameters**:
- `idx` (int): Buffer index
- `data` (numpy.ndarray): Data to write (must match buffer shape/dtype)

**Returns**: None

**Raises**:
- `KeyError` if buffer not registered
- `ValueError` if shape/dtype mismatch
- `RuntimeError` if write or sync fails

**Example**:
```python
# Write input matrix
A = np.random.randn(512, 1024).astype(np.float32)
npu_app.write_buffer(0, A)
```

**Auto-conversion**: If dtype doesn't match, data is automatically converted with a warning.

---

### `read_buffer(idx)`

Reads data from XRT buffer after kernel execution.

**Parameters**:
- `idx` (int): Buffer index

**Returns**: `numpy.ndarray` with buffer contents (original shape/dtype)

**Raises**:
- `KeyError` if buffer not registered
- `RuntimeError` if sync or read fails

**Example**:
```python
# Read output matrix
C = npu_app.read_buffer(2)
# C.shape = (512, 1024), C.dtype = np.float32
```

**Sync**: Automatically syncs from device before reading.

---

### `run(input_buffers=None, output_buffers=None)`

Executes NPU kernel with all registered buffers.

**Parameters**:
- `input_buffers` (list, optional): Input buffer indices (for logging)
- `output_buffers` (list, optional): Output buffer indices (for logging)

**Returns**: `True` if execution successful

**Raises**:
- `RuntimeError` if no buffers registered
- `RuntimeError` if kernel execution fails

**Example**:
```python
# Execute matmul: C = A × B
npu_app.run(
    input_buffers=[0, 1],    # A, B
    output_buffers=[2]       # C
)
```

**Execution Flow**:
1. Builds argument list from all registered buffers (sorted by index)
2. Calls `kernel(*args)` to get run handle
3. Waits for kernel completion with `run_handle.wait()`

---

### `cleanup()`

Releases all XRT buffer objects.

**Parameters**: None

**Returns**: None

**Example**:
```python
# Free NPU memory
npu_app.cleanup()
```

**Note**: Usually not needed as buffers are released when XRTApp is destroyed.

---

## Buffer Metadata

Each registered buffer stores metadata for validation:

```python
npu_app.buffer_metadata[idx] = {
    'dtype': np.float32,        # Original dtype
    'shape': (512, 1024),       # Original shape
    'size': 2097152             # Size in bytes
}
```

Access metadata:
```python
metadata = npu_app.buffer_metadata[0]
print(f"Buffer 0: {metadata['dtype']} {metadata['shape']} ({metadata['size']} bytes)")
```

---

## Error Handling

### Common Errors

**1. Buffer not registered**:
```python
npu_app.write_buffer(5, data)
# KeyError: Buffer 5 not registered
```
**Solution**: Call `register_buffer(5, dtype, shape)` first

**2. Shape mismatch**:
```python
npu_app.register_buffer(0, np.float32, (100, 200))
data = np.zeros((200, 100))  # Wrong shape!
npu_app.write_buffer(0, data)
# ValueError: Data shape (200, 100) doesn't match buffer 0 shape (100, 200)
```
**Solution**: Ensure data shape matches registered shape

**3. No buffers registered**:
```python
npu_app.run()
# RuntimeError: No buffers registered
```
**Solution**: Register at least one buffer before calling `run()`

**4. XRT allocation failure**:
```python
npu_app.register_buffer(0, np.float32, (100000, 100000))  # 40 GB!
# RuntimeError: Buffer registration failed: [XRT error]
```
**Solution**: Use reasonable buffer sizes (< 100 MB per buffer)

---

## Integration with Encoder

### NPU Callback Chain

```python
# 1. Load XRT application
npu_app = load_xrt_npu_application()

# 2. Create encoder
from xdna2.encoder_cpp import create_encoder_cpp
encoder = create_encoder_cpp(num_layers=6, use_npu=True)

# 3. Register NPU callback
encoder.register_npu_callback(npu_app)
# This internally registers buffers with XRTApp

# 4. Forward pass (uses NPU automatically)
output = encoder.forward(input_data)
```

**Buffer Registration** (automatic):
- Buffer 3: 1,179,648 bytes (A matrix)
- Buffer 4: 4,718,592 bytes (B matrix)
- Buffer 5: 1,179,648 bytes (C matrix)

---

## Performance Tips

### 1. Reuse Buffers
Don't register/cleanup buffers for every request:

**Bad**:
```python
for request in requests:
    npu_app.register_buffer(0, np.float32, (512, 512))
    npu_app.write_buffer(0, request.data)
    npu_app.run()
    result = npu_app.read_buffer(1)
    npu_app.cleanup()  # ❌ Slow!
```

**Good**:
```python
# Register once
npu_app.register_buffer(0, np.float32, (512, 512))
npu_app.register_buffer(1, np.float32, (512, 512))

for request in requests:
    npu_app.write_buffer(0, request.data)
    npu_app.run()
    result = npu_app.read_buffer(1)
# Cleanup at end
```

### 2. Minimize Transfers
Transfer overhead is ~100-200 µs per sync. For large buffers, consider:
- Keeping data on device between operations
- Only reading results when needed
- Batching multiple operations before reading

### 3. Buffer Sizing
- **Small buffers** (< 1 MB): Low overhead, fast transfers
- **Medium buffers** (1-10 MB): Good balance (our case)
- **Large buffers** (> 10 MB): Transfer overhead becomes significant

---

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Debug output shows:
```
DEBUG: Wrote 80,000 bytes to buffer 0
DEBUG: Executing kernel MLIR_AIE with 3 buffers...
DEBUG: Kernel MLIR_AIE execution complete
DEBUG: Read 2,097,152 bytes from buffer 2
```

### Check Buffer Status

```python
# List registered buffers
print(f"Registered buffers: {list(npu_app.xrt_buffers.keys())}")

# Check metadata
for idx, metadata in npu_app.buffer_metadata.items():
    print(f"Buffer {idx}: {metadata}")
```

### Verify Data Integrity

```python
# Write and immediately read back
test_data = np.random.randn(100, 100).astype(np.float32)
npu_app.write_buffer(0, test_data)
read_data = npu_app.read_buffer(0)

if np.allclose(test_data, read_data):
    print("✓ Buffer integrity verified")
else:
    print("✗ Buffer corruption detected")
```

---

## XRT Sync Directions

### Host → Device
```python
bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
```
Used by: `write_buffer()`

### Device → Host
```python
bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
```
Used by: `read_buffer()`

**Important**: Must sync before reading results!

---

## Memory Layout

### Buffer Types

**host_only**:
```python
xrt.bo(device, size, xrt.bo.host_only, group_id)
```
- Accessible from host (CPU)
- Used for input/output data
- Our default choice

**device_only**:
```python
xrt.bo(device, size, xrt.bo.device_only, group_id)
```
- Only accessible from device (NPU)
- Faster for internal computations
- Not used in current implementation

### Group IDs

```python
group_id = kernel.group_id(buffer_index)
```
Maps buffer to specific memory bank on NPU. Required for proper buffer allocation.

---

## Common Patterns

### Pattern 1: Matrix Multiplication

```python
# Register buffers
npu_app.register_buffer(0, np.float32, (M, K))  # A
npu_app.register_buffer(1, np.float32, (K, N))  # B
npu_app.register_buffer(2, np.float32, (M, N))  # C

# Write inputs
npu_app.write_buffer(0, A)
npu_app.write_buffer(1, B)

# Execute
npu_app.run(input_buffers=[0, 1], output_buffers=[2])

# Read result
C = npu_app.read_buffer(2)
```

### Pattern 2: In-place Operation

```python
# Register single buffer
npu_app.register_buffer(0, np.float32, (1024, 1024))

# Write input
npu_app.write_buffer(0, data)

# Execute (modifies buffer 0)
npu_app.run(input_buffers=[0], output_buffers=[0])

# Read modified result
result = npu_app.read_buffer(0)
```

### Pattern 3: Multi-stage Pipeline

```python
# Stage 1: A → B
npu_app.register_buffer(0, np.float32, (512, 512))  # Input
npu_app.register_buffer(1, np.float32, (512, 512))  # Intermediate

npu_app.write_buffer(0, input_data)
npu_app.run(input_buffers=[0], output_buffers=[1])

# Stage 2: B → C (reuse buffer 1 as input)
npu_app.register_buffer(2, np.float32, (512, 512))  # Output

npu_app.run(input_buffers=[1], output_buffers=[2])
result = npu_app.read_buffer(2)
```

---

## Limitations

1. **Buffer count**: Limited by kernel arguments (typically 8-16)
2. **Buffer size**: Limited by NPU memory (~1-2 GB total)
3. **Concurrent execution**: Single kernel at a time (sequential)
4. **Data types**: Must match kernel expectations (BF16/FP32)

---

## Troubleshooting

**Problem**: Kernel execution hangs
**Solution**: Check that all input buffers are written before `run()`

**Problem**: Results are zeros
**Solution**: Ensure `read_buffer()` is called after `run()` completes

**Problem**: Memory leak
**Solution**: Call `cleanup()` when done with buffers

**Problem**: Slow performance
**Solution**: Reuse buffers, minimize transfers, check buffer sizes

---

## References

- **XRT Documentation**: `/opt/xilinx/xrt/docs/`
- **Week 16 Report**: `WEEK16_COMPLETE.md`
- **Integration Tests**: `WEEK16_INTEGRATION_TEST.py`
- **Server Implementation**: `xdna2/server.py` (lines 215-456)

---

**Last Updated**: November 2, 2025
**Version**: 1.0.0
**Status**: Production Ready

**Built with 🦄 by Magic Unicorn Unconventional Technology & Stuff Inc**
