# Whisper XDNA2 C++ Runtime

High-performance C++ implementation of Whisper encoder for AMD XDNA2 NPU.

## Overview

This is the C++ runtime for XDNA2 NPU-accelerated Whisper speech-to-text. It provides a 3-5x speedup over the Python runtime through:

- Zero-copy buffer management
- Batch kernel dispatch
- Minimal Python/C++ overhead
- Optimized memory access patterns

## Performance Targets

| Metric | Python Baseline | C++ Target | Speedup |
|--------|----------------|------------|---------|
| Realtime Factor | 5.59x | 17-28x | 3-5x |
| Latency (10s audio) | ~1,800 ms | ~360-600 ms | 3-5x |
| Power Draw | 5-15W | 5-15W | Same |

## Building

### Prerequisites

```bash
# Install dependencies
sudo apt install -y \
    build-essential \
    cmake \
    libeigen3-dev

# XRT (AMD XDNA Runtime)
# Should already be installed at /opt/xilinx/xrt
```

### Build

```bash
# Build the project
./build.sh
```

This will:
1. Create a `build/` directory
2. Configure with CMake
3. Build the library and tests
4. Run the test suite

### Clean

```bash
# Clean build artifacts
./clean.sh
```

## Running Tests

```bash
cd build
ctest --output-on-failure
```

Individual tests:
```bash
./test_runtime       # Test runtime initialization
./test_encoder       # Test encoder layer
./test_accuracy      # Test accuracy vs Python (needs reference data)
```

## Running Benchmarks

```bash
cd build
./bench_encoder
```

This will:
1. Initialize the runtime
2. Load kernels
3. Run 10 encoder iterations
4. Report average latency and realtime factor
5. Compare to Python baseline

## Project Structure

```
cpp/
├── src/                       # Source files
│   ├── whisper_xdna2_runtime.{hpp,cpp}  # Main runtime
│   ├── buffer_manager.{hpp,cpp}         # Buffer management
│   ├── kernel_loader.{hpp,cpp}          # XRT kernel loader
│   ├── encoder_layer.{hpp,cpp}          # Transformer layer
│   ├── attention.{hpp,cpp}              # Attention module
│   ├── ffn.{hpp,cpp}                    # Feed-forward network
│   └── quantization.{hpp,cpp}           # INT8 quantization
├── tests/                     # Unit tests
│   ├── test_runtime.cpp
│   ├── test_encoder.cpp
│   └── test_accuracy.cpp
├── benchmarks/                # Performance benchmarks
│   └── bench_encoder.cpp
├── cmake/                     # CMake modules
│   └── FindXRT.cmake
├── CMakeLists.txt            # Build configuration
├── build.sh                  # Build script
├── clean.sh                  # Clean script
└── README.md                 # This file
```

## Architecture

### WhisperXDNA2Runtime

Main runtime class that manages:
- NPU device initialization
- Kernel loading (4-tile or 32-tile)
- Weight loading and quantization
- Encoder execution

### BufferManager

Zero-copy buffer manager for NPU data transfers.

### KernelLoader

XRT kernel loader that:
- Loads `.xclbin` compiled kernels
- Manages kernel dispatch
- Handles dimension-based kernel selection

### EncoderLayer

Whisper transformer layer implementation:
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Residual connections

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| CMake | 3.20+ | Build system |
| C++ Compiler | C++17 | Compilation |
| Eigen3 | 3.4+ | Matrix operations |
| XRT | 2.21.0+ | NPU runtime |

## Performance

### Kernel Variants

The runtime supports multiple kernel configurations:

**4-Tile Kernels** (Stable, Development):
- `512x512x512` matmul
- `512x512x2048` matmul

**32-Tile Kernels** (100% NPU Utilization):
- `512x512x512` matmul
- Automatic chunking for larger dimensions

### Optimization Strategy

1. **Zero-Copy Transfers**: Minimize memory copies between CPU and NPU
2. **Batch Dispatch**: Group kernel calls to reduce overhead
3. **INT8 Quantization**: Use NPU's INT8 capabilities for 4x speedup
4. **Memory Layout**: Optimize for NPU tile memory hierarchy

## Development Status

- [x] Build system (CMake)
- [x] Project structure
- [x] Header interfaces
- [x] Stub implementations
- [x] Unit tests
- [x] Benchmark infrastructure
- [ ] XRT integration
- [ ] Weight loading
- [ ] Full encoder implementation
- [ ] Accuracy validation
- [ ] Performance tuning

## Contributing

1. Follow C++17 style guidelines
2. Add tests for new features
3. Run `ctest` before submitting
4. Document public APIs

## License

MIT License - See LICENSE file for details

## Contact

Part of the Unicorn-Amanuensis project for AMD XDNA2 NPU acceleration.

Built with Magic Unicorn Unconventional Technology & Stuff Inc
