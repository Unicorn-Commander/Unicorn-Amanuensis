#!/bin/bash
# Complete NPU Mel Spectrogram Accuracy Benchmarking Suite
# Runs all steps: test generation, benchmarking, visualization, reporting
#
# Author: Magic Unicorn Inc.
# Date: October 28, 2025

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "NPU Mel Spectrogram Accuracy Benchmarking Suite"
echo "=============================================================================="
echo ""
echo "Location: $SCRIPT_DIR"
echo "Date: $(date)"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
echo "------------------------------------------------------------------------------"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found"
    exit 1
fi
echo "✅ python3: $(python3 --version)"

# Check for NPU XCLBIN
if [ ! -f "build_fixed/mel_fixed.xclbin" ]; then
    echo "❌ NPU XCLBIN not found: build_fixed/mel_fixed.xclbin"
    echo "   Run: ./compile_mel_fixed.sh (or similar build script)"
    exit 1
fi
echo "✅ NPU XCLBIN: build_fixed/mel_fixed.xclbin ($(stat -f%z build_fixed/mel_fixed.xclbin 2>/dev/null || stat -c%s build_fixed/mel_fixed.xclbin) bytes)"

# Check for NPU instruction binary
if [ ! -f "build_fixed/insts_fixed.bin" ]; then
    echo "❌ NPU instructions not found: build_fixed/insts_fixed.bin"
    exit 1
fi
echo "✅ NPU instructions: build_fixed/insts_fixed.bin ($(stat -f%z build_fixed/insts_fixed.bin 2>/dev/null || stat -c%s build_fixed/insts_fixed.bin) bytes)"

# Check for librosa
if python3 -c "import librosa" 2>/dev/null; then
    LIBROSA_VERSION=$(python3 -c "import librosa; print(librosa.__version__)")
    echo "✅ librosa: $LIBROSA_VERSION"
    HAS_LIBROSA=true
else
    echo "⚠️  librosa not installed (CPU reference comparisons will be skipped)"
    echo "   Install with: pip install librosa"
    HAS_LIBROSA=false
fi

# Check for scipy
if python3 -c "import scipy" 2>/dev/null; then
    echo "✅ scipy: installed"
else
    echo "⚠️  scipy not installed (some metrics will be skipped)"
    echo "   Install with: pip install scipy"
fi

# Check for matplotlib
if python3 -c "import matplotlib" 2>/dev/null; then
    echo "✅ matplotlib: installed"
    HAS_MATPLOTLIB=true
else
    echo "⚠️  matplotlib not installed (visualizations will be skipped)"
    echo "   Install with: pip install matplotlib"
    HAS_MATPLOTLIB=false
fi

# Check for XRT
if [ ! -f "/opt/xilinx/xrt/python/xrt_binding.py" ] && [ ! -f "/opt/xilinx/xrt/python/pyxrt.py" ]; then
    echo "❌ XRT Python bindings not found"
    echo "   XRT may not be installed correctly"
    exit 1
fi
echo "✅ XRT: Python bindings found"

echo ""

# Step 1: Generate test signals
echo "=============================================================================="
echo "Step 1: Generating Test Signals"
echo "=============================================================================="
echo ""

if [ -d "test_audio" ] && [ "$(ls -A test_audio/*.raw 2>/dev/null | wc -l)" -gt 0 ]; then
    NUM_EXISTING=$(ls -1 test_audio/*.raw 2>/dev/null | wc -l)
    echo "Found $NUM_EXISTING existing test files in test_audio/"
    read -p "Regenerate test signals? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 generate_test_signals.py
    else
        echo "Using existing test signals"
    fi
else
    python3 generate_test_signals.py
fi

echo ""

# Step 2: Run accuracy benchmark
echo "=============================================================================="
echo "Step 2: Running Accuracy Benchmark on NPU"
echo "=============================================================================="
echo ""

if [ "$HAS_LIBROSA" = false ]; then
    echo "⚠️  WARNING: librosa not available"
    echo "   NPU will be tested but no CPU reference comparison"
    echo ""
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Benchmark cancelled"
        exit 1
    fi
fi

python3 benchmark_accuracy.py \
    --test-dir test_audio \
    --xclbin build_fixed/mel_fixed.xclbin \
    --output-dir benchmark_results

echo ""

# Step 3: Generate visualizations
if [ "$HAS_MATPLOTLIB" = true ] && [ "$HAS_LIBROSA" = true ]; then
    echo "=============================================================================="
    echo "Step 3: Generating Visual Comparisons"
    echo "=============================================================================="
    echo ""

    python3 visual_comparison.py \
        --results benchmark_results/benchmark_results.json \
        --output-dir benchmark_results/plots

    echo ""
else
    echo "=============================================================================="
    echo "Step 3: Skipping Visualizations (matplotlib or librosa not available)"
    echo "=============================================================================="
    echo ""
fi

# Step 4: Generate accuracy report
echo "=============================================================================="
echo "Step 4: Generating Accuracy Report"
echo "=============================================================================="
echo ""

python3 accuracy_report.py \
    --results benchmark_results/benchmark_results.json \
    --output ACCURACY_REPORT.md

echo ""

# Summary
echo "=============================================================================="
echo "Benchmarking Complete!"
echo "=============================================================================="
echo ""
echo "Generated Files:"
echo "------------------------------------------------------------------------------"

if [ -d "test_audio" ]; then
    NUM_TESTS=$(ls -1 test_audio/*.raw 2>/dev/null | wc -l || echo 0)
    echo "  test_audio/               - $NUM_TESTS test audio files"
fi

if [ -f "benchmark_results/benchmark_results.json" ]; then
    RESULT_SIZE=$(stat -f%z benchmark_results/benchmark_results.json 2>/dev/null || stat -c%s benchmark_results/benchmark_results.json)
    echo "  benchmark_results/benchmark_results.json - Detailed metrics ($RESULT_SIZE bytes)"
fi

if [ -d "benchmark_results/plots" ]; then
    NUM_PLOTS=$(ls -1 benchmark_results/plots/*.png 2>/dev/null | wc -l || echo 0)
    echo "  benchmark_results/plots/  - $NUM_PLOTS visualization plots"
fi

if [ -f "ACCURACY_REPORT.md" ]; then
    REPORT_SIZE=$(stat -f%z ACCURACY_REPORT.md 2>/dev/null || stat -c%s ACCURACY_REPORT.md)
    echo "  ACCURACY_REPORT.md        - Comprehensive report ($REPORT_SIZE bytes)"
fi

echo ""
echo "Next Steps:"
echo "------------------------------------------------------------------------------"
echo "  1. Review accuracy report:  cat ACCURACY_REPORT.md"
echo "  2. View visualizations:     ls -l benchmark_results/plots/"
echo "  3. Check JSON results:      cat benchmark_results/benchmark_results.json"
echo ""

if [ "$HAS_LIBROSA" = true ]; then
    # Extract verdict from results
    if [ -f "benchmark_results/benchmark_results.json" ]; then
        # Simple grep-based verdict extraction
        echo "Quick Summary:"
        echo "------------------------------------------------------------------------------"
        python3 -c "
import json
with open('benchmark_results/benchmark_results.json') as f:
    results = json.load(f)

correlations = [r['metrics']['correlation'] for r in results if 'metrics' in r and 'correlation' in r['metrics']]
mses = [r['metrics']['mse'] for r in results if 'metrics' in r and 'mse' in r['metrics']]

if correlations and mses:
    import numpy as np
    avg_corr = np.mean(correlations)
    avg_mse = np.mean(mses)

    if avg_corr > 0.99 and avg_mse < 0.01:
        print('  ✅ EXCELLENT: >99% correlation, <0.01 MSE')
        print('  Status: PRODUCTION READY')
    elif avg_corr > 0.95 and avg_mse < 0.1:
        print('  ✅ GOOD: >95% correlation, <0.1 MSE')
        print('  Status: PRODUCTION READY')
    elif avg_corr > 0.90:
        print('  ⚠️  ACCEPTABLE: >90% correlation')
        print('  Status: Some tuning recommended')
    else:
        print(f'  ❌ NEEDS IMPROVEMENT: {avg_corr*100:.1f}% correlation')
        print('  Status: Optimization required')

    print(f'  Average Correlation: {avg_corr*100:.2f}%')
    print(f'  Average MSE: {avg_mse:.4f}')
" 2>/dev/null || echo "  (Run benchmark to see summary)"
    fi
    echo ""
fi

echo "=============================================================================="
echo "✨ Benchmark suite complete! Review ACCURACY_REPORT.md for detailed analysis."
echo "=============================================================================="
