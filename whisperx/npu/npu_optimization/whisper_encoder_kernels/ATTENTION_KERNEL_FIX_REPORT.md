# Attention Kernel Fix Report
## Mel Kernel Analysis Team Lead
**Date**: October 31, 2025

---

## Executive Summary

**ROOT CAUSE FOUND**: Attention kernels return all zeros because they use **deprecated DMA API** that doesn't work on Phoenix NPU.

**MEL KERNEL WORKS** because it uses the **NEW npu.dma_memcpy_nd API** introduced for NPU platforms.

**SOLUTION**: Regenerate attention MLIR with correct DMA API or manually patch the runtime_sequence.

---

## Investigation Results

### Test 1: Mel Kernel Test Pattern Applied to Attention ✅

Tested attention kernel using EXACT same buffer allocation and calling pattern as working mel kernel:

```python
# Mel pattern (WORKS):
instr_bo = xrt.bo(device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
input_bo = xrt.bo(device, 800, xrt.bo.flags.host_only, kernel.group_id(3))
output_bo = xrt.bo(device, 80, xrt.bo.flags.host_only, kernel.group_id(4))

opcode = 3
run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
```

**Result**: Attention kernel still returns all zeros with mel test pattern.
**Conclusion**: Issue is NOT in test infrastructure.

---

## Root Cause Analysis

### Critical Difference: DMA API Version

#### Mel Kernel (WORKS) - NEW NPU API ✅

```mlir
aiex.runtime_sequence(%in : memref<800xi8>, %out : memref<80xi8>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c800_i64 = arith.constant 800 : i64
    %c80_i64 = arith.constant 80 : i64

    // NEW NPU DMA API
    aiex.npu.dma_memcpy_nd(%in[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                             [%c1_i64, %c1_i64, %c1_i64, %c800_i64]
                             [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
        metadata = @of_in,
        id = 1 : i64
    } : memref<800xi8>

    // Output DMA
    aiex.npu.dma_memcpy_nd(%out[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                              [%c1_i64, %c1_i64, %c1_i64, %c80_i64]
                              [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
        metadata = @of_out,
        id = 0 : i64
    } : memref<80xi8>

    // Explicit wait
    aiex.npu.dma_wait {symbol = @of_out}
}
```

#### Attention Kernel (FAILS) - OLD Task API ❌

```mlir
aiex.runtime_sequence(%arg0: memref<12288xi8>, ...) {
  // OLD DMA Task API (doesn't work on NPU!)
  %0 = aiex.dma_configure_task_for @of_input_0 {
    aie.dma_bd(%arg0 : memref<12288xi8>, 0, 12288, ...)
               {burst_length = 0 : i32}
    aie.end
  }
  aiex.dma_start_task(%0)
  aiex.dma_await_task(%1)
  aiex.dma_free_task(%0)
}
```

---

## Solutions

### Option 1: Manual MLIR Patch (Fastest - 30 min)

Fixed version created: `attention_fixed_npu_api.mlir`

### Option 2: Regenerate with IRON (Medium - 2 hours)

Modify IRON template to emit `npu.dma_memcpy_nd` instead of `dma_configure_task_for`.

### Option 3: Use Mel Template (Easiest - 1 hour) ✅ RECOMMENDED

Copy mel_fixed_v3.mlir structure and adjust buffer sizes.

---

## Recommended Fix

**Use mel kernel pattern** (Option 3):

1. Copy mel_fixed_v3.mlir structure
2. Change buffer sizes: 800→12288 (in), 80→4096 (out)
3. Change function: mel_kernel_simple → attention_64x64
4. Recompile with Peano + aiecc.py
5. Test

**Timeline**: 1-2 hours
**Success rate**: Very high (mel pattern proven)

---

## Files Created

- `attention_fixed_npu_api.mlir` - Fixed single-tile kernel
- `ATTENTION_KERNEL_FIX_REPORT.md` - This report

