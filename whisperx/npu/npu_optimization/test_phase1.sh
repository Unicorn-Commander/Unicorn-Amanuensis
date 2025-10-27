#!/bin/bash
# Quick test: Phase 1 of MLIR-AIE compilation pipeline
# This validates that aie-opt can transform your MLIR correctly

set -e  # Exit on error

echo "🦄 Testing Phase 1: MLIR Transformations"
echo "========================================="
echo ""

# Tool location
AIE_OPT=/home/ucadmin/mlir-aie-source/build/bin/aie-opt

# Input
INPUT_MLIR=passthrough_step3.mlir

# Check input exists
if [ ! -f "$INPUT_MLIR" ]; then
    echo "❌ Error: Input file not found: $INPUT_MLIR"
    echo "Expected location: $(pwd)/$INPUT_MLIR"
    exit 1
fi

echo "✅ Input file found: $INPUT_MLIR"
echo "   Size: $(stat -c%s "$INPUT_MLIR") bytes"
echo ""

# Create build directory
mkdir -p build
cd build

echo "Step 1: Allocate and lower..."
echo "Command: aie-opt --pass-pipeline=... $INPUT_MLIR -o input_with_addresses.mlir"
echo ""

${AIE_OPT} \
  --pass-pipeline="builtin.module(
    lower-affine,
    aie-canonicalize-device,
    aie.device(
      aie-assign-lock-ids,
      aie-register-objectFifos,
      aie-objectFifo-stateful-transform,
      aie-assign-bd-ids,
      aie-lower-cascade-flows,
      aie-lower-broadcast-packet,
      aie-lower-multicast,
      aie-assign-tile-controller-ids,
      aie-generate-column-control-overlay,
      aie-assign-buffer-addresses{alloc-scheme=bank-aware}
    ),
    convert-scf-to-cf
  )" \
  ../${INPUT_MLIR} \
  -o input_with_addresses.mlir

if [ $? -eq 0 ]; then
    echo "✅ Step 1 PASSED: input_with_addresses.mlir generated"
    echo "   Size: $(stat -c%s input_with_addresses.mlir) bytes"
else
    echo "❌ Step 1 FAILED"
    exit 1
fi

echo ""
echo "Step 2: Create pathfinder flows (routing)..."
echo "Command: aie-opt --aie-create-pathfinder-flows input_with_addresses.mlir -o input_physical.mlir"
echo ""

${AIE_OPT} \
  --aie-create-pathfinder-flows \
  input_with_addresses.mlir \
  -o input_physical.mlir

if [ $? -eq 0 ]; then
    echo "✅ Step 2 PASSED: input_physical.mlir generated"
    echo "   Size: $(stat -c%s input_physical.mlir) bytes"
else
    echo "❌ Step 2 FAILED"
    exit 1
fi

echo ""
echo "========================================="
echo "✅✅✅ Phase 1 COMPLETE! ✅✅✅"
echo "========================================="
echo ""
echo "Generated files:"
echo "  - input_with_addresses.mlir  (allocated)"
echo "  - input_physical.mlir        (routed)"
echo ""
echo "Next steps:"
echo "  1. Inspect input_physical.mlir to verify routing"
echo "  2. Run full pipeline with: ../compile_xclbin.sh"
echo ""
echo "🦄 Phase 1 validation successful! 🦄"
