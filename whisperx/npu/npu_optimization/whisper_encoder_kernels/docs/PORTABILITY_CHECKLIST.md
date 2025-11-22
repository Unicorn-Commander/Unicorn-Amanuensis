# XDNA1/XDNA2 Portability Checklist

**Date**: November 17, 2025
**Version**: 1.0
**Purpose**: Ensure code is portable between XDNA1 and XDNA2

---

## Overview

Use this checklist when developing NPU kernels to ensure 95% code reuse between XDNA1 (Phoenix, 4 columns) and XDNA2 (Strix, 8 columns).

**Goal**: Write once, run on both platforms with minimal changes.

---

## Quick Checklist

### For C++ Kernel Development

- [ ] Use only standard AIE intrinsics (no platform-specific extensions)
- [ ] Avoid hardcoded column counts or tile indices
- [ ] Use vector sizes that work on both platforms (v32int8, v16int16, etc.)
- [ ] Document computational assumptions (tile size, data types)
- [ ] Add header comment explaining kernel purpose
- [ ] Include accuracy validation logic

### For MLIR Development

- [ ] Place in correct directory (`kernels/xdna1/` or `kernels/xdna2/`)
- [ ] Use correct device target (`npu1` for XDNA1, `npu2` for XDNA2)
- [ ] Use IRON ObjectFIFO API (not manual DMA)
- [ ] Use consistent tile naming (tile_X_Y format)
- [ ] Document column mapping strategy
- [ ] Test MLIR lowering with `aie-opt`

### For Python Runtime Development

- [ ] Place shared code in `runtime/common/`
- [ ] Place platform-specific code in `runtime/xdna1/` or `runtime/xdna2/`
- [ ] Inherit from `NPUBase` class
- [ ] Use runtime column detection (not hardcoded)
- [ ] Document performance expectations
- [ ] Include error handling for missing hardware

### For Testing

- [ ] Place shared tests in `tests/common/`
- [ ] Use parameterized tests (test both platforms with same code)
- [ ] Validate accuracy against CPU reference
- [ ] Benchmark performance (compare to targets)
- [ ] Test edge cases (zero inputs, max values, etc.)
- [ ] Mock test XDNA2 logic (before hardware available)

---

## Detailed Checklists

## C++ Kernel Checklist

### 1. File Location and Naming

```
✅ CORRECT:
kernels/common/attention_int8.c
kernels/common/matmul_int8.c
kernels/common/gelu_int8.c

❌ WRONG:
kernels/attention_int8_xdna1.c     # Platform in name
kernels/xdna1/attention_int8.c     # C++ in platform dir
attention_int8_64x64.c              # Size in filename (use comments instead)
```

**Rule**: All C++ kernels in `kernels/common/` with generic names.

### 2. Header Comment Template

```c
/*
 * Kernel: Attention Mechanism (INT8)
 * Purpose: Compute scaled dot-product attention
 * Input: Q, K, V matrices (64×64 INT8 each)
 * Output: Attention output (64×64 INT8)
 * Algorithm: Q @ K^T → softmax → @ V
 * Portability: 100% (works on XDNA1 and XDNA2)
 * Author: [Your Name]
 * Date: [Date]
 */
```

**Check**: [ ] Header comment present with purpose, I/O, algorithm

### 3. Avoid Platform-Specific Code

**✅ GOOD** (portable):
```c
void matmul_int8(int8_t* A, int8_t* B, int8_t* C, int M, int K, int N) {
    // Use standard AIE intrinsics
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k += 32) {
                v32int8 a_vec = *(v32int8*)&A[i*K + k];
                v32int8 b_vec = *(v32int8*)&B[k*N + j];
                acc = mac16(acc, a_vec, 0, 0x76543210, 16, b_vec, 0, 0, 2);
            }
            C[i*N + j] = (int8_t)srs(acc, 8);
        }
    }
}
```

**❌ BAD** (platform-specific):
```c
void matmul_int8_xdna1(int8_t* A, int8_t* B, int8_t* C) {
    #ifdef XDNA1
        // 4-column specific code
        #define NUM_COLS 4
    #elif XDNA2
        // 8-column specific code
        #define NUM_COLS 8
    #endif

    // Column-specific logic (should be in MLIR, not C++)
    process_on_column(0, NUM_COLS);  // ❌ Wrong place
}
```

**Check**: [ ] No `#ifdef XDNA1` or `#ifdef XDNA2` in C++ code

### 4. Use Standard Vector Types

**✅ GOOD**:
```c
v32int8  vec8;   // 32 elements of INT8
v16int16 vec16;  // 16 elements of INT16
v8int32  vec32;  // 8 elements of INT32
```

**❌ BAD**:
```c
v64int8 vec;  // ❌ Not supported on all AIE variants
```

**Check**: [ ] Only use vector types supported on AIE (v32int8, v16int16, v8int32, v4int64)

### 5. Parameter Passing

**✅ GOOD** (flexible):
```c
void kernel(int8_t* input, int8_t* output, int size) {
    // Size is parameter, not hardcoded
}
```

**❌ BAD** (hardcoded):
```c
void kernel(int8_t* input, int8_t* output) {
    #define SIZE 256  // ❌ Hardcoded
}
```

**Check**: [ ] Dimensions passed as parameters, not hardcoded defines

---

## MLIR Checklist

### 1. File Location and Naming

```
✅ CORRECT:
kernels/xdna1/attention_xdna1.mlir
kernels/xdna1/matmul_xdna1.mlir
kernels/xdna2/attention_xdna2.mlir
kernels/xdna2/matmul_xdna2.mlir

❌ WRONG:
kernels/common/attention.mlir       # MLIR is platform-specific
kernels/attention_4col.mlir          # Column count in name (use xdna1/xdna2)
```

**Rule**: MLIR goes in platform directory (`xdna1/` or `xdna2/`).

**Check**: [ ] MLIR file in correct platform directory

### 2. Device Target

**✅ GOOD**:
```mlir
// kernels/xdna1/kernel_xdna1.mlir
module @kernel_xdna1 {
  aie.device(npu1) {  // ✅ Correct for XDNA1
    // ...
  }
}

// kernels/xdna2/kernel_xdna2.mlir
module @kernel_xdna2 {
  aie.device(npu2) {  // ✅ Correct for XDNA2
    // ...
  }
}
```

**❌ BAD**:
```mlir
// kernels/xdna1/kernel_xdna1.mlir
module @kernel_xdna1 {
  aie.device(npu2) {  // ❌ Wrong device for XDNA1
    // ...
  }
}
```

**Check**: [ ] Device target matches platform (`npu1` for XDNA1, `npu2` for XDNA2)

### 3. Use IRON ObjectFIFO (Preferred)

**✅ GOOD** (IRON):
```mlir
// Automatic DMA management
%fifo_in = aie.objectFifo.createObjectFifo(%tile_shim, {%tile_compute},
                                            4096 : i32) : !aie.objectFifo<memref<64x64xi8>>

aie.core(%tile_compute) {
  %in = aie.objectFifo.acquire %fifo_in : !aie.objectFifoSubview<memref<64x64xi8>>
  // Use input
  aie.objectFifo.release %fifo_in
  aie.end
}
```

**⚠️ ACCEPTABLE** (Manual DMA, if needed):
```mlir
// Only use if IRON doesn't support your pattern
%buf = aie.buffer(%tile) : memref<64x64xi8>
%lock = aie.lock(%tile, 0)

%dma = aie.dma_start(S2MM, 0, ^bd0, ^end)
^bd0:
  aie.use_lock(%lock, AcquireGreaterEqual, 1)
  aie.dma_bd(%buf : memref<64x64xi8>, 0, 4096)
  aie.use_lock(%lock, Release, 0)
  aie.next_bd ^end
^end:
  aie.end
```

**Check**: [ ] IRON ObjectFIFO used (unless justified exception)

### 4. Consistent Tile Naming

**✅ GOOD**:
```mlir
// Use (column, row) format consistently
%tile_0_0 = aie.tile(0, 0)  // Shim at column 0
%tile_0_2 = aie.tile(0, 2)  // Compute at column 0, row 2
%tile_1_2 = aie.tile(1, 2)  // Compute at column 1, row 2
```

**❌ BAD**:
```mlir
%tile_A = aie.tile(0, 2)  // ❌ Unclear naming
%tile_B = aie.tile(2, 4)  // ❌ Non-standard layout
```

**Check**: [ ] Tile names follow `tile_X_Y` format (column, row)

### 5. Document Column Mapping

```mlir
// kernels/xdna1/attention_xdna1.mlir
module @attention_xdna1 {
  aie.device(npu1) {
    // XDNA1: 4 columns available (0-3)
    // Column 0: Q input + processing
    // Column 1: K input + processing
    // Column 2: V input + processing
    // Column 3: Output computation

    %tile_0_2 = aie.tile(0, 2)  // Q processing
    %tile_1_2 = aie.tile(1, 2)  // K processing
    %tile_2_2 = aie.tile(2, 2)  // V processing
    %tile_3_2 = aie.tile(3, 2)  // Output
  }
}
```

**Check**: [ ] Comments explain column usage

---

## Python Runtime Checklist

### 1. File Organization

**✅ GOOD**:
```
runtime/
├── common/
│   ├── npu_base.py              # Base class (shared)
│   ├── buffer_manager.py        # Buffer utilities (shared)
│   └── kernel_loader.py         # XCLBIN loading (shared)
│
├── xdna1/
│   ├── npu_xdna1.py            # XDNA1 runtime (4 columns)
│   ├── npu_attention_wrapper.py
│   └── npu_matmul_wrapper.py
│
└── xdna2/
    ├── npu_xdna2.py            # XDNA2 runtime (8 columns)
    ├── npu_attention_wrapper.py
    └── npu_matmul_wrapper.py
```

**Check**: [ ] Shared code in `common/`, platform code in `xdna1/` or `xdna2/`

### 2. Inherit from NPUBase

**✅ GOOD**:
```python
# runtime/xdna1/npu_xdna1.py
from runtime.common.npu_base import NPUBase

class NPUXDNA1(NPUBase):
    """XDNA1 (Phoenix) NPU Runtime - 4 columns"""

    NUM_COLUMNS = 4
    DEVICE_NAME = "npu1"

    def __init__(self):
        super().__init__()
        self.xclbin_dir = "build/xdna1"
```

**❌ BAD**:
```python
# Doesn't inherit from base
class NPUXDNA1:  # ❌ No inheritance
    def __init__(self):
        self.device = xrt.xrt_device(0)
        # Duplicates base class code
```

**Check**: [ ] Runtime class inherits from `NPUBase`

### 3. Runtime Column Detection

**✅ GOOD**:
```python
def get_tile_mapping(self, operation_width):
    """Map operation to available columns"""
    if operation_width <= self.NUM_COLUMNS:
        return list(range(operation_width))
    else:
        # Time-multiplex if operation wider than columns
        return self._time_multiplex(operation_width, self.NUM_COLUMNS)
```

**❌ BAD**:
```python
def get_tile_mapping(self, operation_width):
    # Hardcoded for 4 columns
    return [0, 1, 2, 3]  # ❌ Will break on XDNA2
```

**Check**: [ ] Column count uses `self.NUM_COLUMNS` (not hardcoded)

### 4. Document Performance Expectations

```python
class NPUAttention:
    """
    Attention mechanism on NPU

    Performance:
    - XDNA1 (64×64): 3.62 ms
    - XDNA1 (4-col 8-head): ~9 ms
    - XDNA2 (8-col 8-head): ~5 ms (projected)

    Accuracy:
    - Correlation with CPU: >0.95
    - Non-zero output: 88-90%
    """
```

**Check**: [ ] Docstring includes performance and accuracy expectations

### 5. Error Handling

**✅ GOOD**:
```python
def __init__(self):
    try:
        self.device = xrt.xrt_device(0)
    except RuntimeError:
        raise RuntimeError("NPU device not found. Check /dev/accel/accel0")

    try:
        self.load_xclbin(self.xclbin_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load XCLBIN: {e}")
```

**Check**: [ ] Graceful error handling with clear messages

---

## Testing Checklist

### 1. Test Organization

**✅ GOOD**:
```
tests/
├── common/
│   ├── test_attention_accuracy.py   # Shared accuracy test
│   ├── test_matmul_accuracy.py
│   └── benchmark_suite.py            # Shared benchmarks
│
├── xdna1/
│   └── test_xdna1_kernels.py        # Platform-specific tests
│
└── xdna2/
    └── test_xdna2_kernels.py        # Platform-specific tests (mock for now)
```

**Check**: [ ] Shared tests in `common/`, platform tests in `xdna1/xdna2/`

### 2. Parameterized Tests

**✅ GOOD**:
```python
# tests/common/test_attention_accuracy.py
def test_attention_accuracy(npu):
    """Works with any NPU (XDNA1 or XDNA2)"""
    Q, K, V = generate_test_inputs(64, 64)
    output_npu = npu.attention(Q, K, V)
    output_cpu = cpu_reference(Q, K, V)

    correlation = np.corrcoef(output_npu.flat, output_cpu.flat)[0, 1]
    assert correlation > 0.95
```

**Usage**:
```python
# tests/xdna1/test_xdna1_kernels.py
from runtime.xdna1.npu_xdna1 import NPUXDNA1
from tests.common.test_attention_accuracy import test_attention_accuracy

npu = NPUXDNA1()
test_attention_accuracy(npu)  # Same test, XDNA1 hardware
```

**Check**: [ ] Tests accept NPU instance as parameter (not hardcoded platform)

### 3. Mock Testing for XDNA2

**✅ GOOD**:
```python
# tests/xdna2/test_xdna2_mock.py
import pytest

@pytest.mark.xdna2
def test_xdna2_column_mapping():
    """Test column mapping logic (no hardware needed)"""
    from runtime.xdna2.npu_xdna2 import NPUXDNA2

    npu = NPUXDNA2()

    # Test 8-column mapping
    assert npu.NUM_COLUMNS == 8
    assert npu.get_tile_mapping(8) == [0, 1, 2, 3, 4, 5, 6, 7]
```

**Check**: [ ] Mock tests for XDNA2 logic (before hardware available)

### 4. CPU Reference Validation

**✅ GOOD**:
```python
def test_matmul_accuracy():
    # NPU computation
    C_npu = npu.matmul(A, B)

    # CPU reference
    C_ref = np.matmul(A.astype(np.int32), B.astype(np.int32))
    C_ref = np.clip(C_ref >> 8, -128, 127).astype(np.int8)

    # Validate
    correlation = np.corrcoef(C_npu.flat, C_ref.flat)[0, 1]
    assert correlation > 0.95, f"Low correlation: {correlation}"
```

**Check**: [ ] All accuracy tests validate against CPU reference

---

## Code Review Checklist

### Before Submitting Code

- [ ] All files in correct directories
- [ ] C++ kernels in `kernels/common/`
- [ ] MLIR in `kernels/xdna1/` or `kernels/xdna2/`
- [ ] Shared Python in `runtime/common/`
- [ ] Platform Python in `runtime/xdna1/` or `runtime/xdna2/`
- [ ] Tests in `tests/common/` or `tests/xdna1/xdna2/`
- [ ] No hardcoded platform assumptions
- [ ] No hardcoded column counts in C++ or Python
- [ ] IRON ObjectFIFO used in MLIR (unless justified)
- [ ] All tests pass
- [ ] Documentation updated (if new kernel)
- [ ] Performance benchmarked (meets targets)

### Portability Validation

- [ ] C++ kernel compiles without warnings
- [ ] MLIR lowers correctly (`aie-opt --aie-canonicalize-device`)
- [ ] XCLBIN generates successfully
- [ ] Kernel loads on NPU
- [ ] Kernel executes without errors
- [ ] Output validated against CPU reference
- [ ] Performance meets targets
- [ ] Code works on current platform (XDNA1)
- [ ] Code structure supports future platform (XDNA2)

---

## Quick Reference: What Goes Where

### 100% Portable (Shared)

| Type | Location | Example |
|------|----------|---------|
| **C++ Kernels** | `kernels/common/` | `attention_int8.c` |
| **Python Base** | `runtime/common/` | `npu_base.py` |
| **Shared Tests** | `tests/common/` | `test_attention_accuracy.py` |
| **Documentation** | `docs/` | `QUICK_START_XDNA1_XDNA2.md` |

### Platform-Specific (5%)

| Type | Location | Example |
|------|----------|---------|
| **XDNA1 MLIR** | `kernels/xdna1/` | `attention_xdna1.mlir` |
| **XDNA2 MLIR** | `kernels/xdna2/` | `attention_xdna2.mlir` |
| **XDNA1 Runtime** | `runtime/xdna1/` | `npu_xdna1.py` |
| **XDNA2 Runtime** | `runtime/xdna2/` | `npu_xdna2.py` |
| **XDNA1 Tests** | `tests/xdna1/` | `test_xdna1_kernels.py` |
| **XDNA2 Tests** | `tests/xdna2/` | `test_xdna2_kernels.py` |

---

## Common Mistakes

### Mistake #1: Platform-Specific C++ Code

**❌ Wrong**:
```c
// kernels/common/attention_int8.c
#ifdef XDNA1
    process_4_columns();
#else
    process_8_columns();
#endif
```

**✅ Fix**: Move column logic to MLIR
```c
// kernels/common/attention_int8.c
void attention_int8(int8_t* Q, int8_t* K, int8_t* V, int8_t* out) {
    // Generic computation, no platform-specific code
}
```

### Mistake #2: Hardcoded Dimensions

**❌ Wrong**:
```python
class NPUMatmul:
    def __init__(self):
        self.tile_size = 32  # Hardcoded
```

**✅ Fix**: Use runtime detection
```python
class NPUMatmul(NPUBase):
    def __init__(self):
        super().__init__()
        self.tile_size = self._detect_optimal_tile_size()
```

### Mistake #3: MLIR in Wrong Directory

**❌ Wrong**:
```
kernels/common/attention.mlir  # MLIR is platform-specific
```

**✅ Fix**:
```
kernels/xdna1/attention_xdna1.mlir
kernels/xdna2/attention_xdna2.mlir
```

### Mistake #4: Duplicate Test Logic

**❌ Wrong**:
```python
# tests/xdna1/test_xdna1_attention.py
def test_attention_xdna1():
    # 100 lines of test logic

# tests/xdna2/test_xdna2_attention.py
def test_attention_xdna2():
    # Same 100 lines (duplicated)
```

**✅ Fix**:
```python
# tests/common/test_attention_accuracy.py
def test_attention_accuracy(npu):
    # 100 lines of shared logic

# tests/xdna1/test_xdna1_kernels.py
npu = NPUXDNA1()
test_attention_accuracy(npu)

# tests/xdna2/test_xdna2_kernels.py
npu = NPUXDNA2()
test_attention_accuracy(npu)
```

---

## Summary

### Portability Principles

1. **Computation in C++, Layout in MLIR** - Kernel logic is portable, tile mapping is platform-specific
2. **Shared Base, Platform Extensions** - 95% shared code, 5% platform overrides
3. **Runtime Detection** - Auto-detect hardware, don't hardcode assumptions
4. **Test Once, Run Everywhere** - Shared test logic, parameterized by platform

### Expected Code Distribution

| Category | % Shared | % Platform-Specific |
|----------|----------|---------------------|
| **C++ Kernels** | 100% | 0% |
| **MLIR** | 0% | 100% |
| **Python Runtime** | 95% | 5% |
| **Tests** | 100% logic | 0% logic (different targets) |
| **Overall** | **95%** | **5%** |

### Review Questions

Before submitting code, ask:
1. Could this C++ kernel work on XDNA2? (Should be yes)
2. Is column count hardcoded anywhere? (Should be no)
3. Is this MLIR in the right directory? (xdna1/ or xdna2/)
4. Do tests work with any NPU instance? (Should be yes)
5. Is documentation updated? (Should be yes)

If all answers correct → **✅ Code is portable!**

---

**Document Version**: 1.0
**Last Updated**: November 17, 2025
**Maintained By**: NPU Documentation Team
**Next Review**: After first XDNA2 port (when hardware available)
