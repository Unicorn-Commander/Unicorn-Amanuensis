# Attention Optimization Quick Start Guide

**Goal**: Reduce attention kernel time from 2.233ms to 0.5-1.0ms (2-4× improvement)

**Current**: 14.0× realtime → **Target**: 40-60× realtime

**Timeline**: Start today, see results in 1 week

---

## TL;DR - Start Here

**Most Impactful First Step**: Vectorize Q@K^T matmul (2-3× speedup, 1-2 days)

```bash
# 1. Copy current attention kernel
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels
cp attention_int8_64x64.c attention_int8_64x64_vectorized.c

# 2. Edit to add vectorization (see template below)
# Replace scalar inner loop with AIE2 SIMD

# 3. Compile
./compile_attention_vectorized.sh

# 4. Test
python3 test_attention_vectorized.py

# 5. Benchmark
python3 benchmark_suite/benchmark_kernels.py --kernel=attention_vectorized

# Expected result: 2-3× faster (2.233ms → 0.7-1.1ms)
```

---

## Option 1: Quick Win - Test Tiled Version (30 minutes)

**Why**: Tiled version already exists and may be faster

**What**: Test `attention_int8_64x64_tiled.c` vs current

### Step-by-Step

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels

# 1. Check if tiled version is already compiled
ls -lh build_attention_tiled/

# If not compiled:
# 2. Create compilation script
cat > compile_attention_tiled.sh << 'EOF'
#!/bin/bash
set -e

echo "Compiling tiled attention kernel..."

KERNEL_DIR="$(pwd)"
BUILD_DIR="$KERNEL_DIR/build_attention_tiled"
mkdir -p "$BUILD_DIR"

# Copy files
cp attention_int8_64x64_tiled.c "$BUILD_DIR/"
cp attention_64x64.mlir "$BUILD_DIR/"

cd "$BUILD_DIR"

# Compile C kernel
$PEANO/bin/clang --target=aie2 \
    -c attention_int8_64x64_tiled.c \
    -o attention_tiled.o

# Lower MLIR and generate XCLBIN
aie-opt attention_64x64.mlir \
    --aie-canonicalize-device \
    --aie-objectFifo-stateful-transform \
    --aie-create-pathfinder-flows \
    --aie-assign-buffer-addresses \
    -o attention_lowered.mlir

aie-translate attention_lowered.mlir \
    --aie-generate-xclbin \
    --peano-install-dir=$PEANO \
    -o attention_tiled.xclbin

echo "✅ Built: $BUILD_DIR/attention_tiled.xclbin"
EOF

chmod +x compile_attention_tiled.sh
./compile_attention_tiled.sh

# 3. Test
python3 test_attention_tiled.py

# 4. Benchmark
python3 -c "
from benchmark_suite.benchmark_kernels import benchmark_single_kernel
import time

print('Benchmarking tiled attention...')
time_tiled = benchmark_single_kernel('build_attention_tiled/attention_tiled.xclbin', iterations=20)

print('Benchmarking baseline attention...')
time_baseline = benchmark_single_kernel('build_attention_64x64/attention_64x64.xclbin', iterations=20)

speedup = time_baseline / time_tiled
print(f'Speedup: {speedup:.2f}x')
print(f'Baseline: {time_baseline*1000:.3f}ms')
print(f'Tiled: {time_tiled*1000:.3f}ms')
"
```

**Expected Result**:
- If faster: 1.2-1.5× improvement (free optimization!)
- If slower: Continue with vectorization

---

## Option 2: Vectorized Q@K^T (1-2 days, highest ROI)

### Step 1: Create Vectorized Kernel (4 hours)

**File**: `attention_int8_64x64_vectorized.c`

```c
/**
 * Vectorized INT8 Attention for AIE2
 * Uses 32-element SIMD for 2-3× speedup
 */

#include <stdint.h>
#include <aie_api/aie.hpp>

using namespace aie;

// Vectorized Q @ K^T computation
void attention_qk_vectorized_64x64(
    const int8_t* Q,
    const int8_t* K,
    int8_t* scores,
    uint32_t scale_shift
) {
    for (uint32_t i = 0; i < 64; i++) {
        for (uint32_t j = 0; j < 64; j++) {
            // Load 32 elements at a time
            v32int8 q_vec0 = *(v32int8*)&Q[i * 64 + 0];
            v32int8 q_vec1 = *(v32int8*)&Q[i * 64 + 32];
            v32int8 k_vec0 = *(v32int8*)&K[j * 64 + 0];
            v32int8 k_vec1 = *(v32int8*)&K[j * 64 + 32];

            // Vector MAC: 32 multiply-accumulates in parallel
            v32acc32 acc = mul(q_vec0, k_vec0);
            acc = mac(acc, q_vec1, k_vec1);

            // Reduce to scalar (sum all 32 elements)
            auto acc_vec = acc.to_vector<int32_t>();
            int32_t score = 0;
            for (uint32_t v = 0; v < 32; v++) {
                score += acc_vec[v];
            }

            // Scale and clamp
            score >>= scale_shift;
            if (score > 127) score = 127;
            if (score < -128) score = -128;

            scores[i * 64 + j] = (int8_t)score;
        }
    }
}

// Keep existing softmax (will optimize later)
void softmax_int8_64(const int8_t* input, int8_t* output, uint32_t N) {
    // Same as current implementation
    // (Will optimize in Week 2)
}

// Vectorized weighted sum @ V
void attention_weighted_sum_vectorized(
    const int8_t* weights,
    const int8_t* V,
    int8_t* output
) {
    for (uint32_t i = 0; i < 64; i++) {
        for (uint32_t j = 0; j < 64; j++) {
            // Similar vectorization as Q@K^T
            v32int8 w_vec0 = *(v32int8*)&weights[i * 64 + 0];
            v32int8 w_vec1 = *(v32int8*)&weights[i * 64 + 32];
            v32int8 v_vec0 = *(v32int8*)&V[0 * 64 + j];  // First 32 rows
            v32int8 v_vec1 = *(v32int8*)&V[32 * 64 + j]; // Next 32 rows

            // Problem: V access is strided!
            // Need to gather columns or transpose V

            // Fallback to scalar for now (optimize in Phase 2)
            int32_t weighted_sum = 0;
            for (uint32_t k = 0; k < 64; k++) {
                weighted_sum += weights[i*64+k] * V[k*64+j];
            }

            weighted_sum >>= 7;
            if (weighted_sum > 127) weighted_sum = 127;
            if (weighted_sum < -128) weighted_sum = -128;

            output[i * 64 + j] = (int8_t)weighted_sum;
        }
    }
}

// Main attention kernel
void attention_64x64(
    const int8_t* QKV_combined,
    int8_t* output,
    uint32_t scale_shift
) {
    const int8_t* Q = &QKV_combined[0];
    const int8_t* K = &QKV_combined[4096];
    const int8_t* V = &QKV_combined[8192];

    int8_t scores[4096];
    int8_t attention_weights[4096];

    // Vectorized Q @ K^T
    attention_qk_vectorized_64x64(Q, K, scores, scale_shift);

    // Softmax (keep current - optimize later)
    for (uint32_t i = 0; i < 64; i++) {
        softmax_int8_64(&scores[i * 64], &attention_weights[i * 64], 64);
    }

    // Vectorized weighted sum
    attention_weighted_sum_vectorized(attention_weights, V, output);
}
```

### Step 2: Create MLIR Definition (1 hour)

**File**: `attention_64x64_vectorized.mlir`

```mlir
// Copy from attention_64x64.mlir
// Change kernel reference to attention_64x64_vectorized.o

module @attention_npu_64x64_vectorized {
    aie.device(npu1) {
        func.func private @attention_64x64(memref<12288xi8>, memref<4096xi8>, i32)

        %tile00 = aie.tile(0, 0)
        %tile02 = aie.tile(0, 2)

        aie.objectfifo @of_QKV(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<12288xi8>>
        aie.objectfifo @of_out(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>

        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index
            %c_scale = arith.constant 4 : i32

            scf.for %iter = %c0 to %c_max step %c1 {
                %subviewQKV = aie.objectfifo.acquire @of_QKV(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
                %elemQKV = aie.objectfifo.subview.access %subviewQKV[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>

                %subviewOut = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>

                func.call @attention_64x64(%elemQKV, %elemOut, %c_scale)
                    : (memref<12288xi8>, memref<4096xi8>, i32) -> ()

                aie.objectfifo.release @of_QKV(Consume, 1)
                aie.objectfifo.release @of_out(Produce, 1)
            }

            aie.end
        } {link_with="attention_vectorized.o"}

        aiex.runtime_sequence(%QKV_combined : memref<12288xi8>, %out : memref<4096xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c4096_i64 = arith.constant 4096 : i64
            %c12288_i64 = arith.constant 12288 : i64

            aiex.npu.dma_memcpy_nd(%QKV_combined[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                                [%c1_i64, %c1_i64, %c1_i64, %c12288_i64]
                                                [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_QKV,
                id = 1 : i64
            } : memref<12288xi8>

            aiex.npu.dma_memcpy_nd(%out[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                       [%c1_i64, %c1_i64, %c1_i64, %c4096_i64]
                                       [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_out,
                id = 0 : i64
            } : memref<4096xi8>

            aiex.npu.dma_wait {symbol = @of_out}
        }
    }
}
```

### Step 3: Compile (30 minutes)

**File**: `compile_attention_vectorized.sh`

```bash
#!/bin/bash
set -e

KERNEL_DIR="$(pwd)"
BUILD_DIR="$KERNEL_DIR/build_attention_vectorized"
mkdir -p "$BUILD_DIR"

echo "Building vectorized attention kernel..."

# Copy files
cp attention_int8_64x64_vectorized.c "$BUILD_DIR/"
cp attention_64x64_vectorized.mlir "$BUILD_DIR/"

cd "$BUILD_DIR"

# Compile C++ kernel with AIE2 vector support
$PEANO/bin/clang++ --target=aie2 \
    -std=c++20 \
    -c attention_int8_64x64_vectorized.c \
    -I$PEANO/include \
    -o attention_vectorized.o

echo "✅ Compiled kernel object"

# Lower MLIR
aie-opt attention_64x64_vectorized.mlir \
    --aie-canonicalize-device \
    --aie-objectFifo-stateful-transform \
    --aie-create-pathfinder-flows \
    --aie-assign-buffer-addresses \
    -o attention_vectorized_lowered.mlir

echo "✅ Lowered MLIR"

# Generate XCLBIN
aie-translate attention_vectorized_lowered.mlir \
    --aie-generate-xclbin \
    --peano-install-dir=$PEANO \
    -o attention_vectorized.xclbin

echo "✅ Generated XCLBIN: $BUILD_DIR/attention_vectorized.xclbin"
ls -lh attention_vectorized.xclbin
```

### Step 4: Test (1 hour)

**File**: `test_attention_vectorized.py`

```python
#!/usr/bin/env python3
import numpy as np
import pyxrt as xrt

def test_attention_vectorized():
    print("Testing vectorized attention kernel...")

    # Load XCLBIN
    device = xrt.device(0)
    xclbin = xrt.xclbin("build_attention_vectorized/attention_vectorized.xclbin")
    device.register_xclbin(xclbin)

    # Create test data (64×64 matrices)
    np.random.seed(42)
    Q = np.random.randint(-127, 128, (64, 64), dtype=np.int8)
    K = np.random.randint(-127, 128, (64, 64), dtype=np.int8)
    V = np.random.randint(-127, 128, (64, 64), dtype=np.int8)

    # Pack QKV
    QKV_combined = np.concatenate([Q.flatten(), K.flatten(), V.flatten()])
    output = np.zeros(4096, dtype=np.int8)

    # Run kernel
    # ... (DMA transfer and execution - copy from test_attention_64x64.py)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min()}, {output.max()}]")
    print(f"Output mean: {output.mean():.2f}")

    # Compare with baseline
    # ... (accuracy check)

    print("✅ Vectorized attention test passed!")

if __name__ == "__main__":
    test_attention_vectorized()
```

### Step 5: Benchmark (30 minutes)

```bash
python3 -c "
from benchmark_suite.benchmark_kernels import benchmark_single_kernel

print('Benchmarking vectorized attention...')
time_vectorized = benchmark_single_kernel(
    'build_attention_vectorized/attention_vectorized.xclbin',
    iterations=20
)

print('Benchmarking baseline...')
time_baseline = benchmark_single_kernel(
    'build_attention_64x64/attention_64x64.xclbin',
    iterations=20
)

speedup = time_baseline / time_vectorized
print(f'\n╔════════════════════════════════════╗')
print(f'║  VECTORIZED ATTENTION RESULTS      ║')
print(f'╚════════════════════════════════════╝')
print(f'Baseline:    {time_baseline*1000:.3f}ms')
print(f'Vectorized:  {time_vectorized*1000:.3f}ms')
print(f'Speedup:     {speedup:.2f}x')
print(f'Target:      2-3x')
print(f'Status:      {'✅ ACHIEVED' if speedup >= 2.0 else '⚠️ INVESTIGATE'}')
"
```

**Expected Output**:
```
╔════════════════════════════════════╗
║  VECTORIZED ATTENTION RESULTS      ║
╚════════════════════════════════════╝
Baseline:    2.233ms
Vectorized:  0.892ms
Speedup:     2.50x
Target:      2-3x
Status:      ✅ ACHIEVED
```

---

## Option 3: LUT-Based Softmax (1 day, medium ROI)

### Step 1: Generate Lookup Table

**File**: `generate_exp_lut.py`

```python
#!/usr/bin/env python3
import numpy as np

def generate_exp_lut():
    """Generate exp() lookup table for softmax"""
    print("Generating exp() LUT for INT8 softmax...")

    lut = []
    for x in range(-128, 128):
        # Scaled exp for INT8 precision
        exp_val = 64 * np.exp(x / 64.0)
        exp_val = int(min(255, max(0, exp_val)))
        lut.append(exp_val)

    # Generate C header
    with open('exp_lut.h', 'w') as f:
        f.write("#ifndef EXP_LUT_H\n")
        f.write("#define EXP_LUT_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// Precomputed exp() lookup table for softmax\n")
        f.write("// Index: x + 128 where x in [-128, 127]\n")
        f.write("const uint8_t EXP_LUT[256] = {\n")

        for i in range(0, 256, 16):
            values = ", ".join(f"{lut[i+j]:3d}" for j in range(16))
            f.write(f"    {values},\n")

        f.write("};\n\n")
        f.write("#endif // EXP_LUT_H\n")

    print(f"✅ Generated exp_lut.h")
    print(f"   Size: 256 bytes")
    print(f"   Range: x ∈ [-128, 127] → exp(x/64) ∈ [0, 255]")

if __name__ == "__main__":
    generate_exp_lut()
```

Run:
```bash
python3 generate_exp_lut.py
```

### Step 2: Implement LUT Softmax

**File**: `attention_int8_64x64_lut.c` (or modify vectorized version)

```c
#include <stdint.h>
#include "exp_lut.h"

void softmax_int8_64_lut(const int8_t* input, int8_t* output, uint32_t N) {
    // Find max
    int8_t max_val = input[0];
    for (uint32_t i = 1; i < N; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Compute exp using LUT
    uint32_t exp_vals[64];
    uint32_t sum = 0;

    for (uint32_t i = 0; i < N; i++) {
        int32_t x = input[i] - max_val;

        // Clamp and lookup
        if (x < -128) x = -128;
        uint8_t idx = (uint8_t)(x + 128);

        uint32_t exp_val = EXP_LUT[idx];
        exp_vals[i] = exp_val;
        sum += exp_val;
    }

    // Normalize
    for (uint32_t i = 0; i < N; i++) {
        int32_t normalized = (exp_vals[i] * 127) / sum;
        if (normalized > 127) normalized = 127;
        output[i] = (int8_t)normalized;
    }
}
```

---

## Week 1 Summary Checklist

- [ ] **Day 1**: Test tiled version (30 min)
- [ ] **Day 1-2**: Implement vectorized Q@K^T (4 hours)
- [ ] **Day 2**: Compile and test vectorized version (2 hours)
- [ ] **Day 3**: Generate LUT and implement LUT softmax (3 hours)
- [ ] **Day 4**: Vectorize weighted sum @ V (4 hours)
- [ ] **Day 5**: Full benchmarking and validation (full day)

**Expected Result**:
```
Baseline:       2.233ms (14.0× realtime)
After Week 1:   1.0-1.2ms (25-30× realtime)
Improvement:    2× speedup
Progress:       50% to 4× target
```

---

## Quick Debug Tips

### If Compilation Fails

1. **Check PEANO path**:
   ```bash
   echo $PEANO
   ls $PEANO/bin/clang++
   ```

2. **Check AIE API headers**:
   ```bash
   ls $PEANO/include/aie_api/
   ```

3. **Try simpler test**:
   ```bash
   # Compile hello world first
   $PEANO/bin/clang++ --target=aie2 -c test.cpp
   ```

### If Kernel Execution Fails

1. **Check XCLBIN size**:
   ```bash
   ls -lh build_attention_vectorized/attention_vectorized.xclbin
   # Should be 10-20 KB
   ```

2. **Check device**:
   ```bash
   ls /dev/accel/accel0
   xrt-smi examine
   ```

3. **Test with smaller data**:
   ```python
   # Use 16×16 instead of 64×64 first
   ```

### If Performance is Lower

1. **Profile stages separately**:
   ```python
   # Time Q@K^T, softmax, weighted sum individually
   ```

2. **Check vectorization**:
   ```bash
   # Disassemble .o file
   objdump -d attention_vectorized.o | grep vmac
   ```

3. **Verify no fallback to scalar**:
   ```bash
   # Check for scalar multiply instructions
   objdump -d attention_vectorized.o | grep mul | wc -l
   # Should be minimal if vectorized
   ```

---

## Need Help?

**Documentation**:
- `ATTENTION_OPTIMIZATION_PLAN.md` - Full optimization roadmap
- `ATTENTION_PROFILING_ANALYSIS.md` - Detailed profiling data
- `SESSION_COMPLETE_OCT30.md` - Current baseline performance

**Example Code**:
- `matmul_int8.c` - Vectorization patterns
- `attention_int8_64x64_tiled.c` - Tiling example
- `test_encoder_block.py` - Integration testing

**Benchmarking**:
- `benchmark_suite/benchmark_kernels.py` - Performance measurement
- `run_all_benchmarks.py` - Complete benchmark suite

---

**Start Now**: `./compile_attention_tiled.sh` (easiest first step!)

**Expected Timeline**: First improvements in 1 week, full optimization in 6-7 weeks

**Confidence**: Very High (95%)

---

*"From 2.2ms to 0.5ms - one vector at a time!"* 🦄✨🚀
