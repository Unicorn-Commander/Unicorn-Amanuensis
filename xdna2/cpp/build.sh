#!/bin/bash
set -e

echo "========================================"
echo "Building C++ XDNA2 Whisper Runtime"
echo "========================================"

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p build
cd build

# Configure
echo ""
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building with make..."
make -j$(nproc)

# Run tests
echo ""
echo "========================================"
echo "Running Tests"
echo "========================================"
ctest --output-on-failure

echo ""
echo "========================================"
echo "Build Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  - Run tests: cd build && ctest"
echo "  - Run benchmark: cd build && ./bench_encoder"
echo ""
