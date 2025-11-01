# Phase 5 Track 2: Implementation Checklist

**Date**: October 30, 2025
**Project**: CC-1L Whisper Encoder NPU Acceleration
**Mission**: Native BFP16 NPU implementation (eliminate 2,240ms conversion overhead)
**Total Tasks**: 23 tasks across 4 weeks
**Status**: READY TO START

---

## Quick Overview

| Week | Focus | Tasks | Est. Time | Status |
|------|-------|-------|-----------|--------|
| **Week 1** | Kernel Compilation | 7 tasks | 3-4 days | ⏳ Ready |
| **Week 2** | Python Integration | 4 tasks | 2-3 days | ⏳ Pending |
| **Week 3** | C++ Integration | 4 tasks | 2-3 days | ⏳ Pending |
| **Week 4** | Validation & Testing | 8 tasks | 4-5 days | ⏳ Pending |
| **Total** | **Full Implementation** | **23 tasks** | **11-15 days** | **⏳ Not Started** |

---

## Week 1: Kernel Compilation (Days 1-4)

### Task 1.1: Environment Setup

**Goal**: Verify chess compiler installation and environment

**Time Estimate**: 30 minutes

**Steps**:
```bash
# 1. Source chess compiler environment
source ~/setup_bfp16_chess.sh

# 2. Verify chess compiler
which chesscc
# Expected: /home/ccadmin/vitis_aie_essentials/tps/lnx64/target_aie2p/bin/LNa64bin/chesscc

# 3. Check version
chesscc --version
# Expected: V-2024.06#84922c0d9f#241219

# 4. Verify AIETOOLS_DIR
echo $AIETOOLS_DIR
# Expected: /home/ccadmin/vitis_aie_essentials

# 5. Test with AMD example
cd ~/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
env dtype_in=bf16 dtype_out=bf16 m=32 k=32 n=32 M=64 K=64 N=64 \
    use_chess=1 make devicename=npu2
```

**Success Criteria**:
- ✅ `chesscc` command found
- ✅ Version V-2024.06 displayed
- ✅ AMD example compiles without errors

**Potential Issues**:
- ❌ `chesscc not found` → Run `source ~/setup_bfp16_chess.sh`
- ❌ Permission denied → Check file permissions
- ❌ LLVM errors → Ensure `use_chess=1` flag set

**Status**: ⬜ Not Started

---

### Task 1.2: Create BFP16 Kernel Directory

**Goal**: Set up build directory for BFP16 kernel

**Time Estimate**: 15 minutes

**Steps**:
```bash
# 1. Create kernel directory structure
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/kernels
mkdir -p bfp16/build
mkdir -p bfp16/src

# 2. Copy AMD's BFP16 matmul example as template
cp ~/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/Makefile \
   bfp16/Makefile

# 3. Copy kernel source
cp ~/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/mm.cc \
   bfp16/src/mm_bfp16.cc

# 4. Verify structure
tree bfp16/
# Expected:
# bfp16/
# ├── build/
# ├── src/
# │   └── mm_bfp16.cc
# └── Makefile
```

**Success Criteria**:
- ✅ Directory structure created
- ✅ Makefile copied
- ✅ Kernel source copied

**Status**: ⬜ Not Started

---

### Task 1.3: Configure Kernel Parameters

**Goal**: Set kernel dimensions for Whisper encoder

**Time Estimate**: 30 minutes

**Steps**:
```bash
# Edit Makefile to set parameters
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/kernels/bfp16

# Add these parameters at top of Makefile
cat >> Makefile <<'EOF'
# Whisper encoder parameters
dtype_in ?= bf16
dtype_out ?= bf16
m ?= 64
k ?= 64
n ?= 64
M ?= 512
K ?= 512
N ?= 512
use_chess ?= 1
devicename ?= npu2
EOF
```

**Kernel Configurations Needed**:
1. **512×512×512** (Q/K/V/Out projections)
   - `M=512 K=512 N=512 m=64 k=64 n=64`
2. **512×512×2048** (FC1 projection)
   - `M=512 K=512 N=2048 m=64 k=64 n=64`
3. **512×2048×512** (FC2 projection)
   - `M=512 K=2048 N=512 m=64 k=64 n=64`

**Success Criteria**:
- ✅ Makefile configured with default parameters
- ✅ `use_chess=1` flag set
- ✅ BF16 data types specified

**Status**: ⬜ Not Started

---

### Task 1.4: Compile BFP16 Kernel (512×512×512)

**Goal**: Compile first BFP16 kernel for 512×512×512 matmul

**Time Estimate**: 2-3 hours (compilation can be slow)

**Steps**:
```bash
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/kernels/bfp16

# Clean previous builds
make clean

# Compile with chess compiler
env dtype_in=bf16 dtype_out=bf16 \
    m=64 k=64 n=64 \
    M=512 K=512 N=512 \
    use_chess=1 \
    make devicename=npu2 2>&1 | tee build/compile_512x512x512.log

# Check for output files
ls -lh build/
# Expected files:
# - matmul_512x512x512_bf16.xclbin (kernel binary, ~3-5 MB)
# - insts_512x512x512_bf16.bin (instructions, ~100-500 KB)
# - *.mlir (intermediate files)
```

**Success Criteria**:
- ✅ Compilation completes without errors
- ✅ XCLBin file generated (3-5 MB)
- ✅ Instructions file generated (~100-500 KB)
- ✅ No warnings about chess compiler

**Potential Issues**:
- ❌ Chess compiler not found → Check Task 1.1
- ❌ Out of memory → Reduce tile count or dimensions
- ❌ LLVM errors → Verify `use_chess=1` is set
- ❌ Linking errors → Check kernel source syntax

**Fallback Plan**:
- Try smaller dimensions first: `M=256 K=256 N=256`
- Reduce tile count: Use 16-tile instead of 32-tile
- Test with AMD's unmodified example first

**Status**: ⬜ Not Started

---

### Task 1.5: Compile Additional Kernel Sizes (Optional)

**Goal**: Compile kernels for FC1 and FC2 projections

**Time Estimate**: 4-6 hours (2-3 hours per kernel)

**Steps**:
```bash
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/kernels/bfp16

# Kernel 2: 512×512×2048 (FC1)
env dtype_in=bf16 dtype_out=bf16 \
    m=64 k=64 n=64 \
    M=512 K=512 N=2048 \
    use_chess=1 \
    make devicename=npu2 2>&1 | tee build/compile_512x512x2048.log

# Kernel 3: 512×2048×512 (FC2)
env dtype_in=bf16 dtype_out=bf16 \
    m=64 k=64 n=64 \
    M=512 K=2048 N=512 \
    use_chess=1 \
    make devicename=npu2 2>&1 | tee build/compile_512x2048x512.log
```

**Note**: This task is OPTIONAL for initial implementation. Can use 512×512×512 kernel for all matmuls (with padding) as first pass.

**Success Criteria** (if attempted):
- ✅ Both kernels compile successfully
- ✅ XCLBin files generated
- ✅ Total 3 kernels available

**Status**: ⬜ Not Started (Optional)

---

### Task 1.6: Validate Kernel Metadata

**Goal**: Verify kernel contains expected BFP16 configuration

**Time Estimate**: 30 minutes

**Steps**:
```bash
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/kernels/bfp16/build

# Inspect XCLBin metadata
xclbinutil --info --input matmul_512x512x512_bf16.xclbin > kernel_metadata.txt

# Check for expected fields
grep -i "bf16\|bfp16" kernel_metadata.txt
grep -i "512" kernel_metadata.txt
grep -i "tile" kernel_metadata.txt

# Verify file sizes
du -h matmul_512x512x512_bf16.xclbin
# Expected: 3-5 MB (if much larger, may indicate issue)

du -h insts_512x512x512_bf16.bin
# Expected: 100-500 KB
```

**Success Criteria**:
- ✅ Metadata contains BF16/BFP16 references
- ✅ Dimensions (512×512×512) present
- ✅ Tile count visible (expect 32 tiles)
- ✅ File sizes reasonable (<10 MB total)

**Status**: ⬜ Not Started

---

### Task 1.7: Test Kernel Loading (XRT)

**Goal**: Verify XRT can load the BFP16 kernel

**Time Estimate**: 1 hour

**Steps**:
```bash
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/kernels/bfp16

# Create test script
cat > test_kernel_load.py <<'EOF'
#!/usr/bin/env python3
import sys
sys.path.insert(0, "/opt/xilinx/xrt/python")
from aie.utils.xrt import AIE_Application
from pathlib import Path

print("Testing BFP16 kernel loading...")

xclbin_path = Path("build/matmul_512x512x512_bf16.xclbin")
insts_path = Path("build/insts_512x512x512_bf16.bin")

assert xclbin_path.exists(), f"XCLBin not found: {xclbin_path}"
assert insts_path.exists(), f"Instructions not found: {insts_path}"

try:
    npu_app = AIE_Application(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    print(f"✅ Kernel loaded successfully!")
    print(f"   XCLBin: {xclbin_path}")
    print(f"   Instructions: {insts_path}")
except Exception as e:
    print(f"❌ Kernel loading failed: {e}")
    sys.exit(1)
EOF

# Run test
python3 test_kernel_load.py
```

**Success Criteria**:
- ✅ No XRT errors during loading
- ✅ Kernel name "MLIR_AIE" found
- ✅ Device initialized successfully

**Potential Issues**:
- ❌ XRT not found → Check `export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib`
- ❌ Device busy → Reboot or kill existing XRT processes
- ❌ Invalid XCLBin → Recompile kernel (Task 1.4)

**Status**: ⬜ Not Started

---

## Week 2: Python Integration (Days 5-7)

### Task 2.1: Update Buffer Registration

**Goal**: Register BFP16 buffers (1.125× size instead of INT8)

**Time Estimate**: 1 hour

**Location**: `tests/test_encoder_layer_bfp16_npu.py`

**Steps**:
```python
# Current (Track 1 - INT8):
MAX_M = 512
MAX_K = 2048
MAX_N = 2048

npu_app.register_buffer(3, np.int8, (MAX_M * MAX_K,))
npu_app.register_buffer(4, np.int8, (MAX_K * MAX_N,))
npu_app.register_buffer(5, np.int32, (MAX_M * MAX_N,))

# Track 2 (BFP16):
MAX_M = 512
MAX_K = 2048
MAX_N = 2048

# Calculate BFP16 buffer sizes (1.125× logical size)
def bfp16_size(logical_dim):
    return ((logical_dim + 7) // 8) * 9

M_BFP16 = bfp16_size(MAX_M)  # 576 bytes per 512 values
K_BFP16 = bfp16_size(MAX_K)  # 2,304 bytes per 2,048 values
N_BFP16 = bfp16_size(MAX_N)  # 2,304 bytes per 2,048 values

npu_app.register_buffer(3, np.uint8, (MAX_M * K_BFP16,))  # Input A
npu_app.register_buffer(4, np.uint8, (MAX_N * K_BFP16,))  # Input B (transposed)
npu_app.register_buffer(5, np.uint8, (MAX_M * N_BFP16,))  # Output C
```

**Changes Summary**:
- Type: `np.int8` → `np.uint8` (BFP16 uses unsigned)
- Size: Logical dimensions → BFP16 dimensions (1.125× multiplier)
- Output: `np.int32` → `np.uint8` (BFP16 format)

**Success Criteria**:
- ✅ Buffer registration succeeds
- ✅ Buffer sizes correct: A=294,912 B=1,179,648 C=294,912 bytes
- ✅ No memory allocation errors

**Status**: ⬜ Not Started

---

### Task 2.2: Rewrite NPU Callback (Remove Conversions)

**Goal**: Eliminate BFP16↔INT8 conversions in callback

**Time Estimate**: 2-3 hours

**Location**: `tests/test_encoder_layer_bfp16_npu.py` lines 249-322

**Steps**:

**Delete These Functions** (Track 1 only):
- `bfp16_to_int8_simple()` (lines 137-177)
- `int32_to_bfp16_simple()` (lines 180-227)

**Rewrite Callback**:
```python
# Track 2 (Native BFP16) - MUCH SIMPLER!
def npu_bfp16_callback(user_data, A_bfp16_ptr, B_bfp16_ptr, C_bfp16_ptr, M, K, N):
    """
    Native BFP16 NPU callback (NO conversion overhead).

    Workflow:
    1. Wrap pointers as NumPy arrays (zero-copy)
    2. Write BFP16 data directly to NPU
    3. Execute BFP16 kernel
    4. Read BFP16 data directly from NPU
    """
    try:
        start_total = time.perf_counter()

        # Calculate BFP16 buffer sizes
        K_bfp16 = ((K + 7) // 8) * 9
        N_bfp16 = ((N + 7) // 8) * 9

        # Wrap C pointers as NumPy arrays (ZERO-COPY)
        A_bfp16 = np.ctypeslib.as_array(A_bfp16_ptr, shape=(M * K_bfp16,))
        B_bfp16 = np.ctypeslib.as_array(B_bfp16_ptr, shape=(N * K_bfp16,))
        C_bfp16 = np.ctypeslib.as_array(C_bfp16_ptr, shape=(M * N_bfp16,))

        # Fallback for oversized matrices
        if M > MAX_M or K > MAX_K or N > MAX_N:
            # Use CPU for oversized (rare case)
            # TODO: Implement BFP16 CPU fallback if needed
            raise NotImplementedError("Oversized matrices not supported in Track 2")

        # Write BFP16 data DIRECTLY to NPU (no conversion!)
        start_npu = time.perf_counter()
        npu_app.buffers[3].write(A_bfp16)
        npu_app.buffers[4].write(B_bfp16)

        # Execute BFP16 kernel on NPU
        npu_app.run()

        # Read BFP16 data DIRECTLY from NPU (no conversion!)
        C_bfp16_result = npu_app.buffers[5].read()
        C_bfp16[:] = C_bfp16_result[:M * N_bfp16]

        npu_time = (time.perf_counter() - start_npu) * 1000

        # Update statistics
        total_time = (time.perf_counter() - start_total) * 1000
        callback_stats['call_count'] += 1
        callback_stats['total_time_ms'] += total_time
        callback_stats['npu_time_ms'] += npu_time
        callback_stats['conversion_time_ms'] += 0  # NO CONVERSION!

        return 0

    except Exception as e:
        print(f"❌ NPU callback error: {e}")
        import traceback
        traceback.print_exc()
        return -1
```

**Key Changes**:
- Removed: `bfp16_to_int8_simple()` call (1120ms saved!)
- Removed: `int32_to_bfp16_simple()` call (1120ms saved!)
- Added: Zero-copy array wrapping
- Added: Direct BFP16 buffer writes/reads
- Result: **2,240ms overhead eliminated!**

**Success Criteria**:
- ✅ Callback code is 50% shorter
- ✅ No conversion functions called
- ✅ Only zero-copy array operations

**Status**: ⬜ Not Started

---

### Task 2.3: Update Kernel Path

**Goal**: Point to BFP16 kernel instead of INT8 kernel

**Time Estimate**: 15 minutes

**Location**: `tests/test_encoder_layer_bfp16_npu.py` lines 100-111

**Steps**:
```python
# Current (Track 1):
kernel_dir = Path(__file__).parent.parent / "kernels" / "common" / "build"
xclbin_path = kernel_dir / "matmul_32tile_int8.xclbin"
insts_path = kernel_dir / "insts_32tile_int8.bin"

# Track 2:
kernel_dir = Path(__file__).parent.parent / "kernels" / "bfp16" / "build"
xclbin_path = kernel_dir / "matmul_512x512x512_bf16.xclbin"
insts_path = kernel_dir / "insts_512x512x512_bf16.bin"

# Verify files exist
if not xclbin_path.exists():
    print(f"❌ BFP16 kernel not found: {xclbin_path}")
    print("   Run Week 1 tasks to compile BFP16 kernel first!")
    sys.exit(1)

if not insts_path.exists():
    print(f"❌ Instructions not found: {insts_path}")
    sys.exit(1)
```

**Success Criteria**:
- ✅ Path points to bfp16/build directory
- ✅ Files exist before attempting to load
- ✅ Clear error messages if files missing

**Status**: ⬜ Not Started

---

### Task 2.4: Test with Dummy Data

**Goal**: Verify callback works with simple test cases

**Time Estimate**: 1 hour

**Steps**:
```python
# Create test script
cat > tests/test_bfp16_callback_dummy.py <<'EOF'
#!/usr/bin/env python3
"""Test BFP16 NPU callback with dummy data."""
import numpy as np
import sys
sys.path.insert(0, "/opt/xilinx/xrt/python")
from test_encoder_layer_bfp16_npu import *

print("Testing BFP16 callback with dummy data...")

# Test 1: All zeros
print("\nTest 1: All zeros")
A = np.zeros((64, 64), dtype=np.float32)
B = np.zeros((64, 64), dtype=np.float32)
# ... convert to BFP16, call NPU, verify output is zeros

# Test 2: Identity matrix
print("\nTest 2: Identity matrix")
A = np.eye(64, dtype=np.float32)
B = np.eye(64, dtype=np.float32)
# ... convert to BFP16, call NPU, verify output is identity

# Test 3: Random small values
print("\nTest 3: Random small values")
np.random.seed(42)
A = np.random.randn(64, 64).astype(np.float32) * 0.1
B = np.random.randn(64, 64).astype(np.float32) * 0.1
# ... convert to BFP16, call NPU, compare with NumPy reference

print("✅ All dummy data tests passed!")
EOF
```

**Success Criteria**:
- ✅ Test 1: Output is all zeros
- ✅ Test 2: Output is identity matrix
- ✅ Test 3: Output matches NumPy matmul (>99.9% similarity)

**Status**: ⬜ Not Started

---

## Week 3: C++ Integration (Days 8-10)

### Task 3.1: Verify C++ API Compatibility

**Goal**: Ensure C++ encoder_layer.cpp works with Track 2

**Time Estimate**: 1 hour

**Location**: `cpp/src/encoder_layer.cpp` lines 152-233

**Check**:
```cpp
// The existing C++ code should already be compatible!
// It already uses uint8_t (BFP16 format) and calls Python callback

void EncoderLayer::run_npu_linear(
    const Eigen::MatrixXf& input,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& weight_bfp16,
    const Eigen::VectorXf& bias,
    Eigen::MatrixXf& output
) {
    // Prepare input for NPU (FP32 → BFP16 + shuffle)
    BFP16Quantizer bfp16_quantizer;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> input_bfp16_shuffled;
    bfp16_quantizer.prepare_for_npu(input, input_bfp16_shuffled);

    // Allocate BFP16 output buffer
    const size_t N_bfp16 = ((N + 7) / 8) * 9;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> output_bfp16_shuffled(M, N_bfp16);

    // NPU callback (already BFP16 signature!)
    typedef int (*NPUCallback)(void*, const uint8_t*, const uint8_t*, uint8_t*, size_t, size_t, size_t);
    auto callback = reinterpret_cast<NPUCallback>(npu_callback_fn_);

    int result = callback(
        npu_user_data_,
        input_bfp16_shuffled.data(),
        weight_bfp16.data(),
        output_bfp16_shuffled.data(),
        M, K, N
    );

    // Convert NPU output back to FP32 (unshuffle + BFP16 → FP32)
    bfp16_quantizer.read_from_npu(output_bfp16_shuffled, output, M, N);

    // Add bias
    for (size_t i = 0; i < M; ++i) {
        output.row(i) += bias;
    }
}
```

**Verification**:
- ✅ Signature already uses `uint8_t` (BFP16 format)
- ✅ `BFP16Quantizer` already does prepare_for_npu/read_from_npu
- ✅ No changes needed to C++ code!

**Success Criteria**:
- ✅ C++ code compiles without changes
- ✅ NPU callback signature matches Track 2
- ✅ BFP16 format is correct

**Status**: ⬜ Not Started

---

### Task 3.2: Rebuild C++ Library

**Goal**: Ensure library builds with no errors

**Time Estimate**: 30 minutes

**Steps**:
```bash
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build

# Clean previous builds
rm -rf *

# Reconfigure and build
cmake ..
make -j16

# Check for build artifacts
ls -lh libwhisper_encoder_cpp.so
# Expected: ~500 KB shared library

# Run C++ unit tests
ctest --output-on-failure

# Expected results:
# - BFP16QuantizationTest: PASS (6/6 tests)
# - BFP16ConverterTest: PASS (8/8 tests)
# - EncoderLayerBFP16Test: PASS (3/3 tests)
```

**Success Criteria**:
- ✅ Library builds without errors
- ✅ All BFP16 tests pass (11/11)
- ✅ No new warnings introduced

**Status**: ⬜ Not Started

---

### Task 3.3: Test Full Forward Pass

**Goal**: Run encoder layer with Track 2 NPU callback

**Time Estimate**: 2 hours

**Steps**:
```bash
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/tests

# Run Track 2 test
python3 test_encoder_layer_bfp16_npu.py

# Expected output:
# ✅ BFP16 NPU INTEGRATION TEST - TRACK 2
# ✅ Loaded C++ library
# ✅ NPU kernel loaded: matmul_512x512x512_bf16.xclbin
# ✅ NPU buffers allocated
# ✅ NPU callback registered
# ✅ Encoder layer created
# ✅ Weights loaded
#
# Warmup run... ✅ (6 NPU calls)
#
# Benchmark runs...
#   Run 1: 12.5 ms (6 NPU calls, NPU: 11.2 ms, Conv: 0.0 ms)
#   Run 2: 12.3 ms (6 NPU calls, NPU: 11.0 ms, Conv: 0.0 ms)
#   Run 3: 12.4 ms (6 NPU calls, NPU: 11.1 ms, Conv: 0.0 ms)
#   Run 4: 12.6 ms (6 NPU calls, NPU: 11.3 ms, Conv: 0.0 ms)
#   Run 5: 12.5 ms (6 NPU calls, NPU: 11.2 ms, Conv: 0.0 ms)
#
# RESULTS:
#   Average: 12.5 ms (154× FASTER than Track 1!)
#   Conversion time: 0.0 ms (ELIMINATED!)
#   Output valid: ✅
```

**Success Criteria**:
- ✅ No crashes or segfaults
- ✅ Per-layer time: 12-15ms (vs 2,317ms Track 1)
- ✅ Conversion time: 0ms (vs 2,240ms Track 1)
- ✅ Output is valid (no NaN/Inf)

**Status**: ⬜ Not Started

---

### Task 3.4: Verify Output Accuracy

**Goal**: Compare Track 2 output against PyTorch reference

**Time Estimate**: 1 hour

**Steps**:
```python
# Use Team 3's accuracy test suite
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/tests

python3 test_npu_accuracy.py

# Expected results:
# Test 1: Small matmul (64×64×64)
#   Cosine similarity: 0.9999 (>99.99%) ✅
#   Relative error: 0.4%
#
# Test 2: Whisper Q projection (512×512×512)
#   Cosine similarity: 0.9995 (>99.9%) ✅
#   Relative error: 0.8%
#
# Test 3: Single encoder layer (512×512)
#   Cosine similarity: 0.9965 (>99.5%) ✅
#   Relative error: 1.2%
#
# Test 4: Full 6-layer encoder
#   Cosine similarity: 0.9920 (>99%) ✅
#   Relative error: 2.1%
```

**Success Criteria**:
- ✅ Small matmul: >99.9% cosine similarity
- ✅ Single layer: >99.5% cosine similarity
- ✅ Full encoder: >99% cosine similarity
- ✅ BETTER than Track 1 (single quantization > double quantization)

**Status**: ⬜ Not Started

---

## Week 4: Validation & Testing (Days 11-15)

### Task 4.1: Run Comprehensive Accuracy Tests

**Goal**: Validate Track 2 maintains Phase 4 quality

**Time Estimate**: 2-3 hours

**Location**: Use Team 3's test suite

**Tests to Run**:
```bash
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/tests

# 1. PyTorch reference comparison
python3 test_npu_accuracy.py

# 2. Edge cases
python3 test_npu_accuracy.py --edge-cases

# 3. Batch processing
python3 test_npu_accuracy.py --batch 1 2 4 8

# 4. Real weights (Whisper Base)
python3 test_accuracy_vs_pytorch.py --model openai/whisper-base
```

**Success Criteria**:
- ✅ All 6 accuracy test suites pass
- ✅ Edge cases handled (zeros, ones, large, small)
- ✅ Batch processing scales linearly
- ✅ Real weights: >99% accuracy

**Status**: ⬜ Not Started

---

### Task 4.2: Run Performance Benchmarks

**Goal**: Measure Track 2 performance vs targets

**Time Estimate**: 2-3 hours

**Location**: Use Team 3's benchmark suite

**Tests to Run**:
```bash
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/tests

# 1. Matmul 512×512×512
python3 benchmark_npu_performance.py --test matmul

# 2. Single encoder layer
python3 benchmark_npu_performance.py --test layer

# 3. Full 6-layer encoder
python3 benchmark_npu_performance.py --test encoder

# 4. Batch scaling
python3 benchmark_npu_performance.py --test batch

# 5. Warmup effect
python3 benchmark_npu_performance.py --test warmup
```

**Expected Results**:
```
Benchmark 1: Matmul 512×512×512
  Latency: 11.2 ms (target: <2ms) ⚠️
  GFLOPS: 24 GFLOPS (target: >100 GFLOPS) ⚠️
  Note: Single matmul, not parallelized

Benchmark 2: Single encoder layer
  Latency: 12.5 ms (target: <8ms) ⚠️
  Throughput: 80 layers/sec (target: >125 layers/sec) ⚠️
  Note: 6 matmuls sequential

Benchmark 3: Full 6-layer encoder
  Latency: 75 ms (target: <50ms) ⚠️
  Throughput: 13.3 encodes/sec (target: >20 encodes/sec) ⚠️
  Realtime factor: 400× (target: >400×) ✅
  Note: Most important metric - MEETS TARGET!

Benchmark 4: Batch scaling
  Batch 1: 75 ms
  Batch 2: 145 ms (1.93× slower, near-linear)
  Batch 4: 290 ms (1.93× slower, near-linear)
  Batch 8: 580 ms (2.00× slower, linear)

Benchmark 5: Warmup effect
  Cold start: 85 ms (13% slower)
  Warm runs: 75 ms (consistent)
```

**Success Criteria**:
- ✅ 6-layer encoder: <100ms (target: <50ms initially, <100ms acceptable)
- ✅ Realtime factor: >68× (target: >20×, stretch: >400×)
- ✅ Batch scaling: Near-linear (1.9-2.1× per doubling)
- ✅ Warm performance: Consistent (<5% variance)

**Note**: Some individual targets may not be met initially (e.g., <2ms matmul). Focus on **end-to-end encoder performance** (75-90ms) as primary metric.

**Status**: ⬜ Not Started

---

### Task 4.3: Run Stability Tests

**Goal**: Verify no crashes or memory leaks over extended runs

**Time Estimate**: 4-6 hours (mostly unattended)

**Steps**:
```bash
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/tests

# Test 1: 1,000 iterations
python3 test_cpp_npu_stability.py --iterations 1000

# Test 2: Extended runtime (1 hour)
python3 test_cpp_steady_state.py --duration 3600

# Test 3: Valgrind memory check
valgrind --leak-check=full --show-leak-kinds=all \
    python3 test_encoder_layer_bfp16_npu.py
```

**Success Criteria**:
- ✅ 1,000 iterations: No crashes
- ✅ 1 hour runtime: No degradation
- ✅ Valgrind: Zero memory leaks
- ✅ Output remains valid (no NaN/Inf)

**Status**: ⬜ Not Started

---

### Task 4.4: Compare Track 1 vs Track 2

**Goal**: Generate side-by-side comparison report

**Time Estimate**: 2 hours

**Metrics to Compare**:

| Metric | Track 1 | Track 2 | Improvement |
|--------|---------|---------|-------------|
| **Per-layer time** | 2,317 ms | 12-15 ms | **154-193× faster** |
| **NPU time** | 11 ms | 11 ms | Same |
| **Conversion overhead** | 2,240 ms | 0 ms | **Eliminated** |
| **6-layer encoder** | 13,902 ms | 72-90 ms | **154-193× faster** |
| **Realtime factor** | 0.18× | 68-100× | **378-556× faster** |
| **Accuracy (cosine sim)** | 99.0% | 99.5% | **+0.5% better** |
| **Memory usage** | 1.57 MB | 0.88 MB | **56% reduction** |

**Generate Report**:
```python
# Create comparison script
cat > tests/compare_track1_track2.py <<'EOF'
#!/usr/bin/env python3
"""Compare Track 1 vs Track 2 performance and accuracy."""

# Run Track 1 (with conversions)
track1_results = run_track1_tests()

# Run Track 2 (native BFP16)
track2_results = run_track2_tests()

# Generate comparison table
print_comparison_table(track1_results, track2_results)

# Generate charts (if matplotlib available)
plot_performance_comparison(track1_results, track2_results)
plot_accuracy_comparison(track1_results, track2_results)
EOF
```

**Success Criteria**:
- ✅ Track 2 is faster (100-200× speedup)
- ✅ Track 2 has better accuracy (single quantization)
- ✅ Track 2 uses less memory (BFP16 < INT8+INT32)

**Status**: ⬜ Not Started

---

### Task 4.5: Profile Performance Bottlenecks

**Goal**: Identify optimization opportunities

**Time Estimate**: 2 hours

**Steps**:
```bash
# Profile with perf
sudo perf record -g python3 tests/test_encoder_layer_bfp16_npu.py
sudo perf report

# Profile Python code
python3 -m cProfile -o profile.stats tests/test_encoder_layer_bfp16_npu.py
python3 -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# Profile C++ code
cd cpp/build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j16
valgrind --tool=callgrind ../tests/test_encoder_layer
```

**Analyze**:
- Where is time spent? (NPU execution vs overhead)
- Any unexpected bottlenecks?
- Can we parallelize Q/K/V projections?

**Success Criteria**:
- ✅ >80% time in NPU execution (not overhead)
- ✅ No obvious bottlenecks in Python/C++
- ✅ Optimization opportunities identified

**Status**: ⬜ Not Started

---

### Task 4.6: Test with Real Whisper Weights

**Goal**: Validate Track 2 with production weights

**Time Estimate**: 2 hours

**Steps**:
```python
# Load Whisper Base from HuggingFace
from tests.pytorch_reference import WhisperEncoderReference

ref = WhisperEncoderReference(model_name="openai/whisper-base")
weights = ref.extract_weights()

# Load weights into C++ encoder
# ... (use existing weight loading code)

# Run inference
# ... (use existing forward pass code)

# Compare outputs
accuracy = ref.compute_accuracy_metrics(pytorch_output, npu_output)

print(f"Cosine similarity: {accuracy['cosine_similarity']:.4f}")
print(f"Relative error: {accuracy['relative_error']:.2f}%")
```

**Success Criteria**:
- ✅ Weights load successfully
- ✅ Inference completes without errors
- ✅ Accuracy >99% vs PyTorch reference
- ✅ Performance same as random weights (~75ms)

**Status**: ⬜ Not Started

---

### Task 4.7: Generate Final Report

**Goal**: Document Track 2 implementation and results

**Time Estimate**: 3-4 hours

**Report Sections**:
1. **Executive Summary**: Track 2 vs Track 1 comparison
2. **Implementation Details**: What was changed
3. **Performance Results**: Benchmarks and profiling
4. **Accuracy Results**: Validation tests
5. **Lessons Learned**: What worked well, what didn't
6. **Future Work**: Optimization opportunities
7. **Conclusion**: Go/no-go recommendation

**Deliverables**:
- `PHASE5_TRACK2_COMPLETE.md` (comprehensive report)
- `TRACK2_PERFORMANCE_SUMMARY.md` (1-page summary)
- Charts/graphs (performance, accuracy)
- Comparison tables

**Success Criteria**:
- ✅ Report is comprehensive (8-12 pages)
- ✅ All metrics documented
- ✅ Clear recommendations
- ✅ Ready for stakeholder review

**Status**: ⬜ Not Started

---

### Task 4.8: Update Documentation

**Goal**: Update project docs with Track 2 information

**Time Estimate**: 1-2 hours

**Files to Update**:
```bash
# 1. Main README
vim ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/README.md
# Add Track 2 section, update performance numbers

# 2. Testing guide
vim ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/TESTING_GUIDE.md
# Add Track 2 test instructions

# 3. Quick start
vim ~/CC-1L/npu-services/unicorn-amanuensis/xdna2/BFP16_QUICK_START.md
# Update with Track 2 kernel compilation steps

# 4. Architecture docs
vim ~/CC-1L/docs/architecture/OVERVIEW.md
# Update NPU performance projections
```

**Success Criteria**:
- ✅ All docs reference Track 2 (not Track 1)
- ✅ Performance numbers updated
- ✅ Compilation instructions include chess
- ✅ No references to "temporary solution"

**Status**: ⬜ Not Started

---

## Summary Dashboard

### Progress Overview

| Phase | Tasks | Completed | Remaining | Status |
|-------|-------|-----------|-----------|--------|
| **Week 1: Kernel Compilation** | 7 | 0 | 7 | ⏳ Ready |
| **Week 2: Python Integration** | 4 | 0 | 4 | ⏳ Pending |
| **Week 3: C++ Integration** | 4 | 0 | 4 | ⏳ Pending |
| **Week 4: Validation** | 8 | 0 | 8 | ⏳ Pending |
| **TOTAL** | **23** | **0** | **23** | **⏳ Not Started** |

### Critical Path

```
Week 1: Kernel Compilation (BLOCKING)
  ├─ Task 1.1: Environment setup ⏰ 30min
  ├─ Task 1.2: Create directories ⏰ 15min
  ├─ Task 1.3: Configure params ⏰ 30min
  └─ Task 1.4: Compile kernel ⏰ 2-3hrs ← CRITICAL

Week 2: Python Integration (DEPENDS ON WEEK 1)
  ├─ Task 2.1: Buffer registration ⏰ 1hr
  ├─ Task 2.2: Rewrite callback ⏰ 2-3hrs ← CRITICAL
  └─ Task 2.3: Update kernel path ⏰ 15min

Week 3: C++ Integration (DEPENDS ON WEEK 2)
  └─ Task 3.3: Full forward pass ⏰ 2hrs ← CRITICAL

Week 4: Validation (DEPENDS ON WEEK 3)
  ├─ Task 4.2: Performance benchmarks ⏰ 2-3hrs ← CRITICAL
  └─ Task 4.7: Final report ⏰ 3-4hrs ← CRITICAL
```

### Risk Indicators

| Risk | Mitigation | Status |
|------|------------|--------|
| Chess compiler issues | Test with AMD examples first | ✅ Ready |
| Kernel compilation fails | Use 16-tile fallback | ✅ Ready |
| Performance below target | Profile and optimize | ✅ Tools ready |
| Accuracy degradation | Use test suite, compare with Phase 4 | ✅ Tests ready |

### Quick Commands

**Start Week 1**:
```bash
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2
source ~/setup_bfp16_chess.sh
# Then follow Task 1.1
```

**Check Progress**:
```bash
cd ~/CC-1L/npu-services/unicorn-amanuensis/xdna2
cat PHASE5_TRACK2_CHECKLIST.md | grep "Status.*✅"
```

**Run Full Test Suite** (after Week 4):
```bash
cd tests
./run_track2_validation.sh  # Generated during Week 4
```

---

## Appendix: Estimated Timeline

**Optimistic Timeline** (experienced developer, no issues):
- Week 1: 3 days
- Week 2: 2 days
- Week 3: 2 days
- Week 4: 4 days
- **Total: 11 days (2.2 weeks)**

**Realistic Timeline** (normal pace, minor issues):
- Week 1: 4 days
- Week 2: 3 days
- Week 3: 3 days
- Week 4: 5 days
- **Total: 15 days (3 weeks)**

**Conservative Timeline** (first-time, major issues):
- Week 1: 5 days
- Week 2: 4 days
- Week 3: 4 days
- Week 4: 6 days
- **Total: 19 days (3.8 weeks)**

**Recommended**: Plan for **3 weeks** (realistic timeline with buffer).

---

**Document Version**: 1.0
**Author**: Phase 5 Track 2 Planning Team
**Date**: October 30, 2025
**Total Tasks**: 23
**Estimated Duration**: 11-15 days (2-3 weeks)
**Status**: READY TO START

---

Built with Claude Code (Anthropic)
Magic Unicorn Unconventional Technology & Stuff Inc
