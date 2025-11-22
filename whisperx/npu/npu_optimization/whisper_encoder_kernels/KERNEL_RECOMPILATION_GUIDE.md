# Kernel Recompilation Guide for 64×512 Matmul

## Overview

This guide shows how to compile a larger NPU kernel (64×512) to achieve the 8× speedup for matmul operations.

**Current**: `matmul_bf16.xclbin` - 64×64 matrices (8KB each)
**Target**: `matmul_64x512.xclbin` - 64×64 × 64×512 matrices (8KB × 64KB)

---

## Step 1: Create MLIR Source

**Location**: `kernels_xdna1/build_matmul/matmul_64x512.mlir`

### Changes from Original

Copy `matmul_bf16.mlir` and make these modifications:

#### 1. Update Buffer Sizes

```mlir
// OLD (64×64):
memref<8192xi8>  // 64 × 64 × 2 bytes (BF16) = 8192 bytes

// NEW (64×512 for B and C):
memref<8192xi8>   // Input A: 64×64 = 8192 bytes (unchanged)
memref<65536xi8>  // Input B: 64×512 = 65536 bytes
memref<65536xi8>  // Output C: 64×512 = 65536 bytes
```

#### 2. Update Function Signature

```mlir
// Line 21 - OLD:
func.func private @matmul_bf16_64x64(
    memref<8192xi8>,   // A: 64×64
    memref<8192xi8>,   // B: 64×64
    memref<8192xi8>)   // C: 64×64

// NEW:
func.func private @matmul_64x512(
    memref<8192xi8>,   // A: 64×64
    memref<65536xi8>,  // B: 64×512
    memref<65536xi8>)  // C: 64×512
```

#### 3. Update ObjectFIFOs

```mlir
// Lines 28-34 - OLD:
aie.objectfifo @of_input_A(...) : !aie.objectfifo<memref<8192xi8>>
aie.objectfifo @of_input_B(...) : !aie.objectfifo<memref<8192xi8>>
aie.objectfifo @of_output(...) : !aie.objectfifo<memref<8192xi8>>

// NEW:
aie.objectfifo @of_input_A(...) : !aie.objectfifo<memref<8192xi8>>
aie.objectfifo @of_input_B(...) : !aie.objectfifo<memref<65536xi8>>
aie.objectfifo @of_output(...) : !aie.objectfifo<memref<65536xi8>>
```

#### 4. Update Core Logic

```mlir
// Lines 45-58 - Update all buffer accesses:
%elemA = ... -> memref<8192xi8>   // Unchanged
%elemB = ... -> memref<65536xi8>  // Changed
%elemOut = ... -> memref<65536xi8> // Changed

func.call @matmul_64x512(%elemA, %elemB, %elemOut)
    : (memref<8192xi8>, memref<65536xi8>, memref<65536xi8>) -> ()
```

#### 5. Update DMA Transfers

```mlir
// Lines 76-97 - Update transfer sizes:
// Input A: unchanged (8192 bytes)
%c8192_i64 = arith.constant 8192 : i64

// Input B and Output: 65536 bytes
%c65536_i64 = arith.constant 65536 : i64

// Input B DMA:
aiex.npu.dma_memcpy_nd(%inputB[...]
                              [..., %c65536_i64]  // Changed
                              [...]) {
    metadata = @of_input_B,
    id = 2 : i64
} : memref<65536xi8>  // Changed

// Output DMA:
aiex.npu.dma_memcpy_nd(%output[...]
                              [..., %c65536_i64]  // Changed
                              [...]) {
    metadata = @of_output,
    id = 0 : i64
} : memref<65536xi8>  // Changed
```

#### 6. Update Runtime Sequence Signature

```mlir
// Line 70 - OLD:
aiex.runtime_sequence(
    %inputA : memref<8192xi8>,
    %inputB : memref<8192xi8>,
    %output : memref<8192xi8>)

// NEW:
aiex.runtime_sequence(
    %inputA : memref<8192xi8>,
    %inputB : memref<65536xi8>,
    %output : memref<65536xi8>)
```

### Complete Modified File

<details>
<summary>Click to expand full matmul_64x512.mlir</summary>

```mlir
//===- matmul_64x512.mlir ---------------------------------------*- MLIR -*-===//
//
// BF16 Matrix Multiplication 64×512 for Whisper Encoder
// Batched tiling optimization - process full output rows
//
// C = A * B
// A: 64×64 bfloat16 = 8192 bytes
// B: 64×512 bfloat16 = 65536 bytes
// C: 64×512 bfloat16 = 65536 bytes
//
//===----------------------------------------------------------------------===//

module @matmul_64x512_npu {
    aie.device(npu1) {
        // Matrix multiplication kernel for 64×512
        func.func private @matmul_64x512(
            memref<8192xi8>,   // A: 64×64
            memref<65536xi8>,  // B: 64×512
            memref<65536xi8>)  // C: 64×512

        // Declare tiles
        %tile00 = aie.tile(0, 0)  // ShimNOC tile (DMA)
        %tile02 = aie.tile(0, 2)  // Compute tile

        // Input A ObjectFIFO: 64×64 bfloat16 = 8192 bytes
        aie.objectfifo @of_input_A(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>

        // Input B ObjectFIFO: 64×512 bfloat16 = 65536 bytes
        aie.objectfifo @of_input_B(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<65536xi8>>

        // Output C ObjectFIFO: 64×512 bfloat16 = 65536 bytes
        aie.objectfifo @of_output(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<65536xi8>>

        // Core logic
        %core02 = aie.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c_max = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %c_max step %c1 {
                // Acquire input A buffer
                %subviewA = aie.objectfifo.acquire @of_input_A(Consume, 1) : !aie.objectfifosubview<memref<8192xi8>>
                %elemA = aie.objectfifo.subview.access %subviewA[0] : !aie.objectfifosubview<memref<8192xi8>> -> memref<8192xi8>

                // Acquire input B buffer
                %subviewB = aie.objectfifo.acquire @of_input_B(Consume, 1) : !aie.objectfifosubview<memref<65536xi8>>
                %elemB = aie.objectfifo.subview.access %subviewB[0] : !aie.objectfifosubview<memref<65536xi8>> -> memref<65536xi8>

                // Acquire output buffer
                %subviewOut = aie.objectfifo.acquire @of_output(Produce, 1) : !aie.objectfifosubview<memref<65536xi8>>
                %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<65536xi8>> -> memref<65536xi8>

                // Call MatMul kernel: C = A * B
                func.call @matmul_64x512(%elemA, %elemB, %elemOut)
                    : (memref<8192xi8>, memref<65536xi8>, memref<65536xi8>) -> ()

                // Release ObjectFIFOs
                aie.objectfifo.release @of_input_A(Consume, 1)
                aie.objectfifo.release @of_input_B(Consume, 1)
                aie.objectfifo.release @of_output(Produce, 1)
            }

            aie.end
        } {link_with="matmul_64x512_xdna1.o"}

        // Runtime sequence - DMA transfers
        aiex.runtime_sequence(
            %inputA : memref<8192xi8>,
            %inputB : memref<65536xi8>,
            %output : memref<65536xi8>) {
            %c0_i64 = arith.constant 0 : i64
            %c1_i64 = arith.constant 1 : i64
            %c8192_i64 = arith.constant 8192 : i64
            %c65536_i64 = arith.constant 65536 : i64

            // DMA transfer: Input A buffer (host -> NPU)
            aiex.npu.dma_memcpy_nd(%inputA[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c8192_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input_A,
                id = 1 : i64
            } : memref<8192xi8>

            // DMA transfer: Input B buffer (host -> NPU)
            aiex.npu.dma_memcpy_nd(%inputB[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c65536_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_input_B,
                id = 2 : i64
            } : memref<65536xi8>

            // DMA transfer: Output buffer (NPU -> host)
            aiex.npu.dma_memcpy_nd(%output[%c0_i64, %c0_i64, %c0_i64, %c0_i64]
                                          [%c1_i64, %c1_i64, %c1_i64, %c65536_i64]
                                          [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {
                metadata = @of_output,
                id = 0 : i64
            } : memref<65536xi8>

            // Wait for output DMA completion
            aiex.npu.dma_wait {symbol = @of_output}
        }
    }
}
```

</details>

---

## Step 2: Create C++ Kernel Implementation

**Location**: `kernels_xdna1/build_matmul/matmul_64x512_xdna1.cc`

### Modified Kernel Function

```cpp
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>

extern "C" {

// 64×512 matrix multiplication
// A: 64×64 (4096 bf16 elements)
// B: 64×512 (32768 bf16 elements)
// C: 64×512 (32768 bf16 elements)
void matmul_64x512(bfloat16* A, bfloat16* B, bfloat16* C) {
    // Process in 64×64 chunks (8 columns of B)
    for (int col_tile = 0; col_tile < 8; col_tile++) {
        // Offset into B and C for this column tile
        bfloat16* B_col = B + (col_tile * 64 * 64);  // 64×64 tile
        bfloat16* C_col = C + (col_tile * 64 * 64);  // 64×64 tile

        // Call existing 64×64 kernel
        matmul_bf16_64x64_core(A, B_col, C_col);
    }
}

// Original 64×64 kernel (reused)
void matmul_bf16_64x64_core(bfloat16* A, bfloat16* B, bfloat16* C) {
    using namespace aie;

    // Use AIE vector instructions
    // ... (existing implementation from matmul_bf16_xdna1.o)
}

} // extern "C"
```

**Note**: This approach reuses the existing 64×64 kernel 8 times. For maximum performance, implement a native 64×512 kernel using AIE intrinsics.

---

## Step 3: Compile the Kernel

```bash
cd kernels_xdna1/build_matmul

# Set up Peano compiler environment
export CARDANO=/opt/xilinx/aietools  # Or wherever AIE tools are installed
export PATH=$CARDANO/bin:$PATH

# Compile C++ kernel to object file
peano --target=x86aiesimulator \
      --work-dir=. \
      matmul_64x512_xdna1.cc \
      -o matmul_64x512_xdna1.o

# Lower MLIR to physical representation
aie-opt matmul_64x512.mlir \
    --aie-canonicalize-device \
    --aie-objectFifo-stateful-transform \
    --aie-create-pathfinder-flows \
    --aie-assign-buffer-addresses \
    --aie-lower-to-aie \
    --aie-dma-to-npu \
    -o matmul_64x512_lowered.mlir

# Generate XCLBIN
aie-translate matmul_64x512_lowered.mlir \
    --aie-generate-xclbin \
    --xclbin-name=matmul_64x512.xclbin \
    -o matmul_64x512.xclbin

# Extract instruction sequence
aie-translate matmul_64x512_lowered.mlir \
    --aie-generate-npu \
    --npu-insts-name=insts_64x512.bin \
    -o insts_64x512.bin

echo "✅ Compilation complete!"
ls -lh matmul_64x512.xclbin insts_64x512.bin
```

---

## Step 4: Update Python Wrapper

**File**: `attention_npu.py`

### Add New Kernel Loading

```python
def _load_kernels(self):
    """Load matmul kernels"""
    # ... existing code ...

    # Try to load 64×512 kernel (optimized)
    matmul_64x512_path = self.xclbin_dir / "kernels_xdna1/build_matmul/matmul_64x512.xclbin"
    if matmul_64x512_path.exists():
        try:
            print(f"   Loading matmul 64×512: {matmul_64x512_path.name}")
            xclbin_obj = xrt.xclbin(str(matmul_64x512_path))
            uuid = xclbin_obj.get_uuid()
            self.device.register_xclbin(xclbin_obj)
            hw_ctx = xrt.hw_context(self.device, uuid)
            self.matmul_64x512_kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
            self.matmul_64x512_hw_ctx = hw_ctx

            # Load instructions
            insts_path = matmul_64x512_path.parent / "insts_64x512.bin"
            if insts_path.exists():
                with open(insts_path, "rb") as f:
                    self.matmul_64x512_insts = f.read()
                self.matmul_64x512_instr_size = len(self.matmul_64x512_insts)

                # Pre-allocate instruction buffer
                self.matmul_64x512_instr_bo = xrt.bo(
                    self.device,
                    self.matmul_64x512_instr_size,
                    xrt.bo.flags.cacheable,
                    self.matmul_64x512_kernel.group_id(1)
                )
                self.matmul_64x512_instr_bo.write(self.matmul_64x512_insts, 0)
                self.matmul_64x512_instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

                self.use_64x512_kernel = True
                print(f"   ✅ 64×512 kernel loaded ({self.matmul_64x512_instr_size} bytes)")
        except Exception as e:
            print(f"   ⚠️  Failed to load 64×512 kernel: {e}")
            self.use_64x512_kernel = False
    else:
        self.use_64x512_kernel = False
```

### Add 64×512 Kernel Execution Function

```python
def _matmul_npu_64x512(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Execute 64×512 matmul on NPU

    Args:
        A: (64, 64) matrix in float32
        B: (64, 512) matrix in float32

    Returns:
        C: (64, 512) result matrix in float32
    """
    assert A.shape == (64, 64) and B.shape == (64, 512), \
        f"Matrices must be 64×64 and 64×512, got {A.shape} and {B.shape}"

    # Flatten matrices for BF16 conversion
    A_flat = A.flatten()  # 4096 elements
    B_flat = B.flatten()  # 32768 elements

    # Convert to BF16
    A_bf16 = self._float_to_bf16(A_flat)  # 8192 bytes
    B_bf16 = self._float_to_bf16(B_flat)  # 65536 bytes

    # Buffer sizes
    buffer_A = 8192   # 64×64 BF16
    buffer_B = 65536  # 64×512 BF16
    buffer_C = 65536  # 64×512 BF16

    # Create XRT buffers
    kernel = self.matmul_64x512_kernel
    device = self.device

    bo_input_A = xrt.bo(device, buffer_A, xrt.bo.flags.host_only, kernel.group_id(3))
    bo_input_B = xrt.bo(device, buffer_B, xrt.bo.flags.host_only, kernel.group_id(4))
    bo_output = xrt.bo(device, buffer_C, xrt.bo.flags.host_only, kernel.group_id(5))

    # Write inputs
    bo_input_A.write(A_bf16, 0)
    bo_input_A.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_input_B.write(B_bf16, 0)
    bo_input_B.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Execute kernel
    opcode = 3
    run = kernel(opcode, self.matmul_64x512_instr_bo, self.matmul_64x512_instr_size,
                 bo_input_A, bo_input_B, bo_output)
    run.wait()

    # Read output
    bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    output_bytes = bo_output.read(buffer_C, 0).tobytes()

    # Convert back to float32
    output_floats = self._bf16_to_float(output_bytes)

    # Reshape to 64×512
    return output_floats.reshape(64, 512)
```

### Update Tiling Logic

```python
def _matmul_npu_tiled(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tiled matrix multiply - uses 64×512 kernel if available
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match"

    # Pad matrices
    A_pad = self._pad_to_64x64(A)
    B_pad = self._pad_to_64x64(B)

    M_pad, K_pad = A_pad.shape
    K_pad2, N_pad = B_pad.shape

    C_pad = np.zeros((M_pad, N_pad), dtype=np.float32)

    tile_size = 64
    num_tiles_M = M_pad // tile_size
    num_tiles_K = K_pad // tile_size
    num_tiles_N = N_pad // tile_size

    # Use 64×512 kernel if available
    if self.use_64x512_kernel and N_pad == 512:
        print(f"   Using optimized 64×512 kernel")
        # Process full rows at once
        for i in range(num_tiles_M):
            # Accumulate over K dimension
            row_result = np.zeros((tile_size, N_pad), dtype=np.float32)

            for k in range(num_tiles_K):
                # Extract 64×64 tile from A
                A_tile = A_pad[i*tile_size:(i+1)*tile_size,
                               k*tile_size:(k+1)*tile_size]

                # Extract full 64×512 row from B
                B_row = B_pad[k*tile_size:(k+1)*tile_size, :]

                # Single NPU call for entire row
                partial_result = self._matmul_npu_64x512(A_tile, B_row)
                row_result += partial_result

            # Write result row
            C_pad[i*tile_size:(i+1)*tile_size, :] = row_result

    else:
        # Fall back to 64×64 kernel (existing implementation)
        print(f"   Using standard 64×64 kernel")
        for i in range(num_tiles_M):
            for j in range(num_tiles_N):
                C_tile = np.zeros((tile_size, tile_size), dtype=np.float32)

                for k in range(num_tiles_K):
                    A_tile = A_pad[i*tile_size:(i+1)*tile_size,
                                   k*tile_size:(k+1)*tile_size]
                    B_tile = B_pad[k*tile_size:(k+1)*tile_size,
                                   j*tile_size:(j+1)*tile_size]

                    partial_result = self._matmul_npu_64x64(A_tile, B_tile)
                    C_tile += partial_result

                C_pad[i*tile_size:(i+1)*tile_size,
                      j*tile_size:(j+1)*tile_size] = C_tile

    return C_pad[:M, :N]
```

---

## Step 5: Test and Validate

```bash
# Test accuracy
python3 test_attention_matmul.py

# Expected results:
# ✅ All tests pass with max_error < 0.004
# ✅ Using optimized 64×512 kernel

# Benchmark performance
python3 quick_benchmark.py

# Expected results:
# Kernel calls: 376 (vs 3,008 before)
# Time: ~20.4s (vs 163.7s before)
# Speedup: 8.0×
```

---

## Expected Results

### Before (64×64 kernel):
```
Kernel calls: 3,008
Time: 163.7 seconds
Per call: 54.4 ms
```

### After (64×512 kernel):
```
Kernel calls: 376
Time: 20.4 seconds (estimated)
Per call: 54.4 ms (same, but 8× fewer calls)
Speedup: 8.0×
```

### Pipeline Impact:
```
Before: 3+ minutes per test
After: ~25 seconds per test
Speedup: 7-8× overall
```

---

## Troubleshooting

### Issue: Compilation fails with "buffer too large"

**Solution**: NPU has memory limits. Use 2-4 column tiles instead of 8:
- Try 64×256 (32KB) instead of 64×512 (64KB)
- Still gets 4× speedup

### Issue: XCLBIN loads but kernel fails

**Solution**: Check buffer alignment and DMA parameters:
```bash
# Verify instruction sequence
hexdump -C insts_64x512.bin | head

# Check XCLBIN structure
xclbinutil --info --input matmul_64x512.xclbin
```

### Issue: Accuracy degrades

**Solution**: Verify BF16 conversion and accumulation:
- Check that accumulation uses FP32 internally
- Test with smaller matrices first
- Compare against CPU reference

---

## Alternative: 128×512 Kernel

For even better performance (16× speedup), create `matmul_128x512.mlir`:

**Changes**:
- Input A: 128×64 = 16384 bytes
- Input B: 64×512 = 65536 bytes
- Output C: 128×512 = 131072 bytes

**Result**:
- Kernel calls: 3,008 → 188 (16× reduction!)
- Time: 163.7s → 10.2s
- Process 2 row tiles at once

---

## Summary

1. **Copy and modify MLIR file** - 30 minutes
2. **Implement C++ kernel** - 1 hour
3. **Compile XCLBIN** - 30 minutes
4. **Update Python wrapper** - 1 hour
5. **Test and validate** - 1 hour

**Total effort**: 4-5 hours for 8× speedup

**Files created**:
- `matmul_64x512.mlir`
- `matmul_64x512_xdna1.cc`
- `matmul_64x512.xclbin`
- `insts_64x512.bin`

**Code changes**:
- `attention_npu.py` - Add kernel loading and execution

---

**Status**: Implementation guide complete
**Next**: Compile kernel and integrate with Python
**Expected**: 8× speedup for encoder projections
