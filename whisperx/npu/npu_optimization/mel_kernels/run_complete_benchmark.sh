#!/bin/bash
#
# Complete NPU Mel Kernel Performance Benchmark Suite
#
# This script runs the complete benchmarking pipeline:
# 1. Performance benchmarks (100+ iterations)
# 2. Chart generation
# 3. Comprehensive report generation
#
# Author: Magic Unicorn Inc. - Performance Metrics Lead
# Date: October 28, 2025

set -e  # Exit on error

echo "========================================================================"
echo "NPU MEL KERNEL COMPLETE PERFORMANCE BENCHMARK"
echo "========================================================================"
echo ""
echo "Team 3: Performance Metrics Lead"
echo "Magic Unicorn Unconventional Technology & Stuff Inc."
echo ""
echo "This benchmark will:"
echo "  1. Measure processing time per frame (100+ iterations)"
echo "  2. Calculate throughput and realtime factor"
echo "  3. Compare simple vs optimized kernels"
echo "  4. Generate performance charts"
echo "  5. Create comprehensive report"
echo ""
echo "Estimated time: 2-3 minutes"
echo ""

# Configuration
SIMPLE_XCLBIN="${SIMPLE_XCLBIN:-build/mel_simple.xclbin}"
OPTIMIZED_XCLBIN="${OPTIMIZED_XCLBIN:-build/mel_int8_optimized.xclbin}"
ITERATIONS="${ITERATIONS:-100}"
WARMUP="${WARMUP:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results}"

echo "Configuration:"
echo "  Simple kernel:    $SIMPLE_XCLBIN"
echo "  Optimized kernel: $OPTIMIZED_XCLBIN"
echo "  Iterations:       $ITERATIONS"
echo "  Warmup:           $WARMUP"
echo "  Output dir:       $OUTPUT_DIR"
echo ""

# Check if XCLBINs exist
if [ ! -f "$SIMPLE_XCLBIN" ]; then
    echo "❌ Simple XCLBIN not found: $SIMPLE_XCLBIN"
    echo "   Build it first: ./build_mel_complete.sh"
    exit 1
fi

if [ ! -f "$OPTIMIZED_XCLBIN" ]; then
    echo "❌ Optimized XCLBIN not found: $OPTIMIZED_XCLBIN"
    echo "   Build it first: ./build_mel_complete.sh"
    exit 1
fi

# Check if XRT is available
if [ ! -c /dev/accel/accel0 ]; then
    echo "❌ NPU not accessible: /dev/accel/accel0"
    echo "   Check XRT installation: /opt/xilinx/xrt/bin/xrt-smi examine"
    exit 1
fi

echo "✅ Prerequisites checked"
echo ""

# Step 1: Run performance benchmarks
echo "========================================================================"
echo "STEP 1: RUNNING PERFORMANCE BENCHMARKS"
echo "========================================================================"
echo ""

python3 benchmark_performance.py \
    --simple-xclbin "$SIMPLE_XCLBIN" \
    --optimized-xclbin "$OPTIMIZED_XCLBIN" \
    --iterations "$ITERATIONS" \
    --warmup "$WARMUP" \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Benchmark failed!"
    exit 1
fi

echo ""
echo "✅ Benchmarks complete!"
echo ""

# Step 2: Generate charts
echo "========================================================================"
echo "STEP 2: GENERATING PERFORMANCE CHARTS"
echo "========================================================================"
echo ""

# Check if matplotlib is available
if ! python3 -c "import matplotlib" 2>/dev/null; then
    echo "⚠️  matplotlib not available - skipping chart generation"
    echo "   Install with: pip install matplotlib"
    echo ""
else
    python3 create_performance_charts.py \
        --results "$OUTPUT_DIR/performance_benchmarks.json" \
        --output-dir "$OUTPUT_DIR/charts"

    if [ $? -ne 0 ]; then
        echo ""
        echo "⚠️  Chart generation failed (non-fatal)"
        echo ""
    else
        echo ""
        echo "✅ Charts generated!"
        echo ""
    fi
fi

# Step 3: Generate comprehensive report
echo "========================================================================"
echo "STEP 3: GENERATING COMPREHENSIVE REPORT"
echo "========================================================================"
echo ""

python3 generate_performance_report.py \
    --results "$OUTPUT_DIR/performance_benchmarks.json" \
    --output "PERFORMANCE_BENCHMARKS.md"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Report generation failed!"
    exit 1
fi

echo ""
echo "✅ Report generated!"
echo ""

# Summary
echo "========================================================================"
echo "BENCHMARK COMPLETE!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  📊 Benchmark data:  $OUTPUT_DIR/performance_benchmarks.json"
echo "  📈 Charts:          $OUTPUT_DIR/charts/*.png"
echo "  📄 Report:          PERFORMANCE_BENCHMARKS.md"
echo ""

# Show quick summary
echo "Quick Summary:"
echo "──────────────────────────────────────────────────────────────────────"

python3 << 'EOF'
import json
from pathlib import Path

results_file = Path("benchmark_results/performance_benchmarks.json")
if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)

    simple = results['simple']
    optimized = results['optimized']

    time_overhead = ((optimized['mean_time_us'] - simple['mean_time_us']) / simple['mean_time_us']) * 100

    print(f"Simple Kernel:")
    print(f"  Processing Time:  {simple['mean_time_us']:.2f} µs")
    print(f"  Throughput:       {simple['frames_per_second']:,.0f} fps")
    print(f"  Realtime Factor:  {simple['realtime_factor']:.1f}x")
    print()
    print(f"Optimized Kernel:")
    print(f"  Processing Time:  {optimized['mean_time_us']:.2f} µs")
    print(f"  Throughput:       {optimized['frames_per_second']:,.0f} fps")
    print(f"  Realtime Factor:  {optimized['realtime_factor']:.1f}x")
    print()
    print(f"Overhead: {time_overhead:+.1f}%")
    print()

    if time_overhead < 10:
        print("✅ RECOMMENDATION: Use optimized kernel in production")
    elif time_overhead < 20:
        print("⚠️  RECOMMENDATION: Use optimized kernel with monitoring")
    else:
        print("❌ RECOMMENDATION: Use simple kernel in production")
else:
    print("No results found")
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next Steps:"
echo "  1. Review report:  cat PERFORMANCE_BENCHMARKS.md"
echo "  2. View charts:    ls -lh $OUTPUT_DIR/charts/"
echo "  3. Share results:  git add PERFORMANCE_BENCHMARKS.md $OUTPUT_DIR/"
echo ""
echo "Team 3 Mission: COMPLETE ✓"
echo ""
echo "Magic Unicorn Unconventional Technology & Stuff Inc."
echo "Advancing NPU Performance for Real-World AI Applications"
echo ""
