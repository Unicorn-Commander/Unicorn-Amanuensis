# BFP16 Data Format Documentation

## Overview

BFP16 (Block Floating Point 16) is a specialized data format used in AMD XDNA2 NPU for high-performance matrix multiplication with BF16 (Brain Float 16) emulation. This document explains the BFP16 format, conversion process, and memory layout requirements.

## What is BFP16?

**BFP16** is a block floating-point format where:
- **Block size**: 8x8 elements (64 values per block)
- **Encoding**: Each 8x8 block shares a common exponent
- **Storage**: 8 bits per element + 8-bit shared exponent
- **Total**: 72 bytes per 8x8 block (64 mantissas + 8 exponents)

**Compared to BF16**:
- **BF16**: 16 bits per element = 128 bytes per 8x8 block
- **BFP16**: 72 bytes per 8x8 block = **56% storage reduction**
- **Performance**: 2x faster MAC operations on XDNA2 (8x8x8 vs 4x8x8 tiles)

## Data Type in Code

```cpp
// In AIE C++ code (mm_bfp.cc)
typedef bfp16ebs8  // BFP16 with 8-element block size
```

The `ebs8` suffix means "element block size 8" (8x8 blocks).

## FP32 → BFP16 Conversion Process

### Step 1: Block Extraction

Given an FP32 matrix, extract 8x8 blocks:

```python
def extract_8x8_blocks(matrix_fp32, M, K):
    """Extract 8x8 blocks from FP32 matrix"""
    blocks = []
    for i in range(0, M, 8):
        for j in range(0, K, 8):
            block = matrix_fp32[i:i+8, j:j+8]
            blocks.append(block)
    return blocks
```

### Step 2: Find Block Exponent

For each 8x8 block, find the maximum exponent:

```python
def find_block_exponent(block_fp32):
    """Find shared exponent for 8x8 block"""
    # Extract exponents from FP32 values
    abs_values = np.abs(block_fp32)
    max_value = np.max(abs_values)

    if max_value == 0:
        return 0  # All zeros

    # Calculate exponent (FP32 exponent + bias)
    exponent = int(np.floor(np.log2(max_value))) + 127
    return max(0, min(255, exponent))  # Clamp to [0, 255]
```

### Step 3: Quantize Mantissas

Quantize each element to 8-bit mantissa:

```python
def quantize_to_8bit_mantissa(value_fp32, block_exponent):
    """Quantize FP32 value to 8-bit mantissa with shared exponent"""
    if value_fp32 == 0:
        return 0

    # Extract sign, exponent, mantissa from FP32
    bits = struct.unpack('>I', struct.pack('>f', value_fp32))[0]
    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF

    # Calculate relative exponent
    rel_exp = exponent - block_exponent

    # Shift mantissa based on relative exponent
    if rel_exp >= 0:
        shifted_mantissa = (mantissa >> (23 - 7)) >> rel_exp
    else:
        shifted_mantissa = (mantissa >> (23 - 7)) << (-rel_exp)

    # Clamp to 7 bits + sign bit
    mantissa_8bit = min(127, max(-128, shifted_mantissa))

    # Pack sign + mantissa into 8 bits
    if sign:
        mantissa_8bit = -mantissa_8bit

    return mantissa_8bit & 0xFF
```

### Step 4: Pack BFP16 Block

Pack 64 mantissas + 8 exponents into 72 bytes:

```python
def pack_bfp16_block(block_fp32):
    """Pack 8x8 FP32 block into BFP16 format (72 bytes)"""
    # Find shared exponent
    block_exp = find_block_exponent(block_fp32)

    # Quantize all 64 elements
    mantissas = np.zeros((8, 8), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            mantissas[i, j] = quantize_to_8bit_mantissa(block_fp32[i, j], block_exp)

    # Pack into 72 bytes: 64 mantissas + 8 exponents
    # (Each row has 1 exponent byte)
    bfp16_block = bytearray(72)

    # Layout: [row0_mantissas (8 bytes), row0_exp (1 byte), ...]
    for row in range(8):
        # 8 mantissas per row
        bfp16_block[row*9 : row*9+8] = mantissas[row, :]
        # 1 exponent per row (same for all 8 rows in this block)
        bfp16_block[row*9 + 8] = block_exp

    return bfp16_block
```

## BFP16 Shuffle Operation

BFP16 requires **shuffling** the data layout for efficient AIE core access. This is done via `scalarShuffleMatrixForBfp16ebs8()` in `mm_bfp.cc`.

### Why Shuffle?

AIE cores access memory in **8x8 subtiles**. Shuffling rearranges data so each subtile is contiguous in memory, enabling efficient DMA transfers.

### Shuffle Algorithm

```cpp
void scalarShuffleMatrixForBfp16ebs8(size_t tileWidth, size_t tileHeight,
                                     uint8_t *inBfpMatrix,
                                     uint8_t *outBfpMatrix,
                                     bool unshuffle = false) {
    // Adjust width for BFP16 overhead (9 bytes per 8 elements = 1.125x)
    tileWidth = tileWidth * 1.125;

    size_t subtileWidth = 8 * 1.125;   // 9 bytes per row (8 mantissas + 1 exp)
    size_t subtileHeight = 8;

    size_t tileCountingIndex = 0;
    for (size_t subtileStartY = 0; subtileStartY < tileHeight; subtileStartY += subtileHeight) {
        for (size_t subtileStartX = 0; subtileStartX < tileWidth; subtileStartX += subtileWidth) {
            // Copy each 8x8 subtile contiguously
            for (size_t i = 0; i < subtileHeight; ++i) {
                for (size_t j = 0; j < subtileWidth; ++j) {
                    size_t inputIndex = (subtileStartY + i) * tileWidth + (subtileStartX + j);
                    size_t outputIndex = tileCountingIndex;

                    if (!unshuffle) {
                        outBfpMatrix[outputIndex] = inBfpMatrix[inputIndex];
                    } else {
                        outBfpMatrix[inputIndex] = inBfpMatrix[outputIndex];
                    }
                    tileCountingIndex++;
                }
            }
        }
    }
}
```

### Shuffle Example

**Before shuffle** (row-major):
```
Block (0,0) | Block (0,1) | Block (0,2) | ...
Block (1,0) | Block (1,1) | Block (1,2) | ...
...
```

**After shuffle** (subtile-contiguous):
```
Block (0,0) contiguous 72 bytes
Block (0,1) contiguous 72 bytes
Block (0,2) contiguous 72 bytes
Block (1,0) contiguous 72 bytes
...
```

## Memory Layout Requirements

### Input Matrix A (M x K)

- **Format**: BFP16 (bfp16ebs8)
- **Shape**: M rows, K columns
- **Block size**: 8x8
- **Storage**: `M * K * 1.125` bytes (9 bytes per 8 elements)
- **Alignment**: 32-byte aligned for DMA
- **Shuffle**: Required before DMA to AIE

### Input Matrix B (K x N)

- **Format**: BFP16 (bfp16ebs8)
- **Shape**: K rows, N columns (or N rows, K columns if transposed)
- **Block size**: 8x8
- **Storage**: `K * N * 1.125` bytes
- **Alignment**: 32-byte aligned for DMA
- **Shuffle**: Required before DMA to AIE
- **Transpose**: B matrix is assumed **transposed** in `mm_bfp.cc` kernel

### Output Matrix C (M x N)

- **Format**: BFP16 (bfp16ebs8)
- **Shape**: M rows, N columns
- **Block size**: 8x8
- **Storage**: `M * N * 1.125` bytes
- **Alignment**: 32-byte aligned for DMA
- **Unshuffle**: Required after DMA from AIE to restore row-major layout

## Whisper Encoder Dimensions

### Kernel 1: Attention Projections (512x512x512)

```
Matrix A: 512 x 512 = 262,144 elements = 294,912 bytes (288 KB)
Matrix B: 512 x 512 = 262,144 elements = 294,912 bytes (288 KB) [transposed]
Matrix C: 512 x 512 = 262,144 elements = 294,912 bytes (288 KB)
Total: 864 KB (vs 1.5 MB for BF16)
```

Used for:
- Q, K, V projections (3 kernels)
- Attention output projection (1 kernel)

### Kernel 2: FFN fc1 (512x512x2048)

```
Matrix A: 512 x 512 = 262,144 elements = 294,912 bytes (288 KB)
Matrix B: 512 x 2048 = 1,048,576 elements = 1,179,648 bytes (1.13 MB) [transposed]
Matrix C: 512 x 2048 = 1,048,576 elements = 1,179,648 bytes (1.13 MB)
Total: 2.5 MB (vs 4 MB for BF16)
```

Used for:
- FFN first linear layer (expansion)

### Kernel 3: FFN fc2 (512x2048x512)

```
Matrix A: 512 x 2048 = 1,048,576 elements = 1,179,648 bytes (1.13 MB)
Matrix B: 2048 x 512 = 1,048,576 elements = 1,179,648 bytes (1.13 MB) [transposed]
Matrix C: 512 x 512 = 262,144 elements = 294,912 bytes (288 KB)
Total: 2.5 MB (vs 4 MB for BF16)
```

Used for:
- FFN second linear layer (reduction)

## Performance Characteristics

### MAC Operation Dimensions

```cpp
// BFP16 mode (emulate_bf16_mmul_with_bfp16 = true)
r = 8, s = 8, t = 8  // 8x8x8 tiles

// Native BF16 mode (emulate_bf16_mmul_with_bfp16 = false)
r = 4, s = 8, t = 8  // 4x8x8 tiles
```

### Tile Configuration

```python
# For 512x512x512 kernel
m = 64  # Tile rows
k = 64  # Tile cols (shared dimension)
n = 64  # Tile cols (output)

# Tile counts
M_div_m = 512 // 64 = 8 tiles
K_div_k = 512 // 64 = 8 tiles
N_div_n = 512 // 64 = 8 tiles

# Total tiles: 8 x 8 = 64 output tiles
# MAC operations per tile: (64/8) x (64/8) x (64/8) = 8 x 8 x 8 = 512 MACs
```

### Memory Bandwidth

For 512x512x512 kernel:
- **Input A**: 288 KB read once
- **Input B**: 288 KB read once
- **Output C**: 288 KB read + 288 KB write
- **Total**: 1,152 KB transfer per matmul

At 400-500x realtime Whisper:
- ~16,000 matmuls per second (4 per encoder layer, 6 layers, 512/0.03s mel frames)
- ~18 GB/s memory bandwidth required
- Well within XDNA2's ~100 GB/s DDR bandwidth

## Code Integration

### Python Host Code

```python
import numpy as np

# Convert FP32 weights to BFP16
def convert_weights_to_bfp16(weights_fp32):
    """Convert FP32 weights to BFP16 format"""
    M, K = weights_fp32.shape
    bfp16_data = bytearray()

    for i in range(0, M, 8):
        for j in range(0, K, 8):
            block = weights_fp32[i:i+8, j:j+8]
            bfp16_block = pack_bfp16_block(block)
            bfp16_data.extend(bfp16_block)

    return bytes(bfp16_data)

# Shuffle BFP16 data for AIE
def shuffle_bfp16_for_aie(bfp16_data, M, K):
    """Shuffle BFP16 data for efficient AIE access"""
    # This would call the C++ shuffle function via ctypes or pybind11
    # For now, this is a placeholder
    pass

# Load to NPU
def load_weights_to_npu(weights_fp32, npu_buffer):
    """Load FP32 weights to NPU in BFP16 format"""
    # 1. Convert to BFP16
    bfp16_data = convert_weights_to_bfp16(weights_fp32)

    # 2. Shuffle for AIE
    shuffled_data = shuffle_bfp16_for_aie(bfp16_data, *weights_fp32.shape)

    # 3. DMA to NPU
    npu_buffer.copy_from_host(shuffled_data)
```

### C++ Kernel Compilation

```bash
# Compile mm_bfp.cc with dimension macros
xchesscc -c mm_bfp.cc -o mm_512x512x512.o \
    -DDIM_M=512 -DDIM_K=512 -DDIM_N=512 \
    -I../aie_kernel_utils.h \
    -std=c++20 -target aie-ml
```

## References

- **AMD XDNA2 NPU Documentation**: https://www.amd.com/en/technologies/xdna
- **MLIR-AIE GitHub**: https://github.com/Xilinx/mlir-aie
- **AIE API Reference**: https://xilinx.github.io/aie_api/group__group__mmul.html
- **BFP16 Example**: `mlir-aie/aie_kernels/aie2p/mm_bfp.cc`

## Next Steps

1. **Test MLIR generation**: Run `./build_bfp16_kernels.sh` to generate MLIR
2. **Implement Python conversion**: Create `bfp16_convert.py` with FP32→BFP16 conversion
3. **Test on NPU**: Load shuffled BFP16 data and run matmul kernel
4. **Validate accuracy**: Compare BFP16 results vs FP32 ground truth (expect ~0.1% error)
5. **Optimize performance**: Tune tile sizes and DMA patterns for 400-500x realtime

---

**Created**: October 30, 2025
**Author**: Magic Unicorn Unconventional Technology & Stuff Inc
**License**: Apache License v2.0 with LLVM Exceptions
