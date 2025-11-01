# BFP16 Matrix Multiplication Kernels for Whisper Encoder

This directory contains BFP16 (Block Floating Point 16) matrix multiplication kernels optimized for the Whisper encoder on AMD XDNA2 NPU.

## Overview

**BFP16** is a specialized data format that enables 2x faster matrix multiplication on XDNA2 compared to native BF16, while maintaining similar accuracy. These kernels are designed for three key dimensions used in Whisper encoder:

1. **512x512x512**: Attention Q/K/V/out projections
2. **512x512x2048**: FFN fc1 expansion layer
3. **512x2048x512**: FFN fc2 reduction layer

## Directory Structure

```
bfp16/
├── README.md                           # This file
├── BFP16_FORMAT.md                     # BFP16 format documentation
├── mm_bfp.cc                          # BFP16 matmul kernel (from MLIR-AIE)
├── single_core_iron.py                # Original MLIR generator (reference)
├── generate_whisper_bfp16.py          # Whisper-adapted MLIR generator
├── makefile-common                     # Build utilities (from MLIR-AIE)
├── build_bfp16_kernels.sh             # Automated build script
└── build/                             # Build artifacts
    ├── mlir/                          # Generated MLIR files
    │   ├── matmul_512x512x512_bfp16.mlir
    │   ├── matmul_512x512x2048_bfp16.mlir
    │   └── matmul_512x2048x512_bfp16.mlir
    ├── obj/                           # Compiled kernel objects
    └── xclbin/                        # XCLBin files for NPU
```

## Quick Start

### Prerequisites

1. **MLIR-AIE environment** installed at `~/mlir-aie/`
2. **Python 3.13+** with MLIR-AIE Python API
3. **(Optional) Vitis AIE tools** for kernel compilation

### Generate MLIR (Fast, ~30 seconds)

```bash
# Generate MLIR for all three Whisper dimensions
./build_bfp16_kernels.sh

# Output: build/mlir/*.mlir files
```

### Compile Kernels (Slow, ~5 minutes)

```bash
# Generate MLIR + compile C++ kernels to object files
./build_bfp16_kernels.sh --compile

# Output: build/obj/*.o files
```

### Build XCLBin (Very Slow, ~30 minutes per kernel)

```bash
# Full build including XCLBin for NPU deployment
./build_bfp16_kernels.sh --xclbin

# Output: build/xclbin/*.xclbin files
```

## Manual Usage

### Generate MLIR for a Single Dimension

```bash
# Activate MLIR-AIE environment
source ~/mlir-aie/ironenv/bin/activate

# Generate MLIR for 512x512x512 (attention)
python3 generate_whisper_bfp16.py \
    --dev npu2 \
    -M 512 -K 512 -N 512 \
    --dtype_in bf16 \
    --dtype_out bf16 \
    --emulate-bf16-mmul-with-bfp16 true \
    > build/mlir/matmul_512x512x512_bfp16.mlir
```

### Customize Tile Sizes

```bash
# Use 128x128x128 tiles instead of default 64x64x64
python3 generate_whisper_bfp16.py \
    --dev npu2 \
    -M 512 -K 512 -N 512 \
    -m 128 -k 128 -n 128 \
    --dtype_in bf16 \
    --dtype_out bf16 \
    --emulate-bf16-mmul-with-bfp16 true \
    > build/mlir/matmul_512x512x512_bfp16_128tiles.mlir
```

## Files Explained

### mm_bfp.cc

C++ kernel implementing BFP16 matrix multiplication for AIE-ML cores. Key functions:

- `matmul_vectorized_bfp16()`: Main 2x2 tiled matmul kernel
- `zero_kernel()`: Initialize output to zero
- `scalarShuffleMatrixForBfp16ebs8()`: Shuffle data layout for efficient AIE access

**Compilation**:
```cpp
// Dimensions are passed as macros
#define DIM_M 512
#define DIM_K 512
#define DIM_N 512
```

### generate_whisper_bfp16.py

Python script that generates MLIR-AIE2 code for BFP16 matmul. Key features:

- **Device**: `--dev npu2` (XDNA2)
- **Data type**: `--dtype_in bf16 --dtype_out bf16`
- **BFP16 mode**: `--emulate-bf16-mmul-with-bfp16 true` (8x8x8 tiles)
- **Dimensions**: `-M`, `-K`, `-N` (matrix dimensions)
- **Tile sizes**: `-m`, `-k`, `-n` (tile dimensions, default 64)

**Adapted from**: `mlir-aie/programming_examples/basic/matrix_multiplication/single_core/single_core_iron.py`

### build_bfp16_kernels.sh

Automated build script that:
1. Activates MLIR-AIE environment
2. Generates MLIR for all three Whisper dimensions
3. (Optional) Compiles C++ kernels to object files
4. (Optional) Builds XCLBin files for NPU

## BFP16 Data Format

BFP16 (Block Floating Point 16) stores 8x8 blocks with a shared exponent:

- **Block size**: 8x8 elements (64 values)
- **Storage**: 72 bytes per block (64 x 8-bit mantissas + 8 x 8-bit exponents)
- **Compression**: 56% smaller than BF16 (72 bytes vs 128 bytes per block)
- **Performance**: 2x faster MAC on XDNA2 (8x8x8 vs 4x8x8 tiles)

**See `BFP16_FORMAT.md` for detailed format documentation.**

## Whisper Encoder Integration

### Dimensions Used

| Kernel | M | K | N | Usage | Count |
|--------|---|---|-----|-------|-------|
| 512x512x512 | 512 | 512 | 512 | Attention Q/K/V/out | 4x per layer |
| 512x512x2048 | 512 | 512 | 2048 | FFN fc1 | 1x per layer |
| 512x2048x512 | 512 | 2048 | 512 | FFN fc2 | 1x per layer |

**Total per layer**: 6 matmuls (4 attention + 2 FFN)

**Whisper Base**: 6 encoder layers = **36 matmuls per 30ms audio frame**

### Performance Target

**400-500x realtime Whisper Base**:
- **Audio input**: 30ms frames
- **Processing time**: 60-75 µs per frame
- **Throughput**: ~13,000-16,000 frames/second
- **Matmuls/second**: ~470,000-580,000 total matmuls

**Per-kernel performance required**:
- 512x512x512: ~13,000 matmuls/sec (4 per layer x 6 layers x 16,000 fps)
- 512x512x2048: ~3,200 matmuls/sec
- 512x2048x512: ~3,200 matmuls/sec

## Memory Usage

### Kernel 1: 512x512x512 (Attention)

- **Input A**: 288 KB
- **Input B**: 288 KB (transposed)
- **Output C**: 288 KB
- **Total**: 864 KB per matmul

### Kernel 2: 512x512x2048 (FFN fc1)

- **Input A**: 288 KB
- **Input B**: 1.13 MB (transposed)
- **Output C**: 1.13 MB
- **Total**: 2.5 MB per matmul

### Kernel 3: 512x2048x512 (FFN fc2)

- **Input A**: 1.13 MB
- **Input B**: 1.13 MB (transposed)
- **Output C**: 288 KB
- **Total**: 2.5 MB per matmul

## Next Steps

1. **Generate MLIR**: Run `./build_bfp16_kernels.sh` to generate MLIR files
2. **Review MLIR**: Inspect `build/mlir/*.mlir` to understand AIE2 program structure
3. **Implement Python conversion**: Create `bfp16_convert.py` for FP32→BFP16 conversion
4. **Test on NPU**: Load BFP16 weights and run matmul kernels
5. **Integrate with encoder**: Replace FP32 matmuls in `whisper_encoder_v1/` with BFP16 kernels
6. **Benchmark performance**: Measure actual throughput and compare to 400-500x target

## Troubleshooting

### MLIR generation fails

```bash
# Check MLIR-AIE environment
source ~/mlir-aie/ironenv/bin/activate
python3 -c "import aie; print(aie.__version__)"

# Verify Python dependencies
pip list | grep -E "aie|mlir"
```

### Dimension validation errors

Ensure dimensions are divisible by tile sizes:
- M, K, N must be divisible by m, k, n
- m, k, n must be divisible by r, s, t (8, 8, 8 for BFP16)

Example:
```
M=512, m=64, r=8  ✓ (512 % 64 == 0, 64 % 8 == 0)
M=512, m=60, r=8  ✗ (60 % 8 != 0)
```

### Kernel compilation fails

Requires Vitis AIE tools (xchesscc). If not available:
- Use MLIR-only workflow (generate MLIR, skip compilation)
- Install Vitis 2024.1+ from AMD website

## References

- **MLIR-AIE**: https://github.com/Xilinx/mlir-aie
- **AIE API**: https://xilinx.github.io/aie_api/group__group__mmul.html
- **BFP16 Format**: See `BFP16_FORMAT.md` in this directory
- **Whisper Architecture**: `../../../docs/WHISPER_ARCHITECTURE.md`
- **Week 3 Report**: `../../../docs/WEEK3_COMPLETE.md`

## License

Apache License v2.0 with LLVM Exceptions

Copyright (C) 2025, Magic Unicorn Unconventional Technology & Stuff Inc

---

**Built with 🦄 by Magic Unicorn Tech**
