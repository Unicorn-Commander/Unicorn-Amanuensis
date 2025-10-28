#!/usr/bin/env python3
"""
Generate Comprehensive Performance Report

Creates PERFORMANCE_BENCHMARKS.md with:
- Executive summary
- Detailed performance metrics
- Statistical analysis
- Comparison and recommendations
- Production deployment guidance

Author: Magic Unicorn Inc. - Performance Metrics Lead
Date: October 28, 2025
"""

import json
from pathlib import Path
from datetime import datetime
import sys


def load_results(results_file: Path):
    """Load benchmark results from JSON"""
    with open(results_file, 'r') as f:
        return json.load(f)


def format_microseconds(us: float) -> str:
    """Format microseconds with appropriate precision"""
    if us < 1:
        return f"{us*1000:.1f} ns"
    elif us < 1000:
        return f"{us:.2f} µs"
    else:
        return f"{us/1000:.3f} ms"


def calculate_overhead(simple_val: float, optimized_val: float) -> tuple:
    """Calculate overhead percentage and absolute difference"""
    overhead_pct = ((optimized_val - simple_val) / simple_val) * 100
    overhead_abs = optimized_val - simple_val
    return overhead_pct, overhead_abs


def generate_report(results: dict, output_file: Path):
    """Generate comprehensive markdown report

    Args:
        results: Benchmark results dictionary
        output_file: Output file path
    """
    simple = results['simple']
    optimized = results['optimized']

    # Calculate key metrics
    time_overhead_pct, time_overhead_abs = calculate_overhead(
        simple['mean_time_us'], optimized['mean_time_us']
    )

    size_ratio = optimized['xclbin_size_bytes'] / simple['xclbin_size_bytes']

    simple_cv = (simple['std_time_us'] / simple['mean_time_us']) * 100
    optimized_cv = (optimized['std_time_us'] / optimized['mean_time_us']) * 100

    # Generate report
    report = f"""# NPU Mel Kernel Performance Benchmarks

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Author**: Performance Metrics Lead, Magic Unicorn Inc.
**Location**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels`

---

## Executive Summary

This report presents comprehensive performance benchmarks comparing two NPU mel spectrogram kernels:
- **Simple Kernel**: Basic INT8 implementation
- **Optimized Kernel**: Enhanced INT8 with optimizations

### Key Findings

| Metric | Simple | Optimized | Difference |
|--------|--------|-----------|------------|
| **Processing Time** | {simple['mean_time_us']:.2f} µs | {optimized['mean_time_us']:.2f} µs | {time_overhead_pct:+.1f}% |
| **Throughput** | {simple['frames_per_second']:,.0f} fps | {optimized['frames_per_second']:,.0f} fps | {((optimized['frames_per_second'] - simple['frames_per_second']) / simple['frames_per_second'] * 100):+.1f}% |
| **Realtime Factor** | {simple['realtime_factor']:.1f}x | {optimized['realtime_factor']:.1f}x | {((optimized['realtime_factor'] - simple['realtime_factor']) / simple['realtime_factor'] * 100):+.1f}% |
| **XCLBIN Size** | {simple['xclbin_size_bytes']:,} bytes | {optimized['xclbin_size_bytes']:,} bytes | {size_ratio:.2f}x |
| **Timing Variance** | {simple_cv:.2f}% CV | {optimized_cv:.2f}% CV | {optimized_cv - simple_cv:+.2f}% |

### Recommendation

"""

    # Generate recommendation
    if time_overhead_pct < -10:  # Optimized is >10% faster
        recommendation = f"""✅ **USE OPTIMIZED KERNEL IN PRODUCTION**

The optimized kernel provides **{-time_overhead_pct:.1f}% better performance** while maintaining {optimized['xclbin_size_bytes']/1024:.1f} KB size. The performance improvement justifies any additional complexity.

**Justification**:
- Significant speedup: {-time_overhead_abs:.2f} µs faster per frame
- Throughput: {optimized['frames_per_second']:,.0f} fps vs {simple['frames_per_second']:,.0f} fps
- Production-ready performance characteristics
"""
    elif time_overhead_pct < 10 and optimized_cv < simple_cv:  # Similar speed but more consistent
        recommendation = f"""✅ **USE OPTIMIZED KERNEL IN PRODUCTION**

While processing time is similar ({time_overhead_pct:+.1f}% difference), the optimized kernel offers **{simple_cv - optimized_cv:.2f}% better consistency** (lower variance).

**Justification**:
- More predictable performance ({optimized_cv:.2f}% CV vs {simple_cv:.2f}% CV)
- Similar throughput with better reliability
- Suitable for production workloads requiring consistent latency
"""
    elif time_overhead_pct < 20:  # <20% overhead
        recommendation = f"""⚠️ **USE OPTIMIZED KERNEL (WITH MONITORING)**

The optimized kernel has **{time_overhead_pct:.1f}% overhead** ({time_overhead_abs:.2f} µs per frame), which is acceptable for production if additional features justify it.

**Considerations**:
- Monitor performance in production
- Evaluate if optimizations provide value beyond raw speed
- Consider simple kernel if raw performance is critical
- XCLBIN size increase: {size_ratio:.2f}x ({optimized['xclbin_size_bytes']/1024:.1f} KB vs {simple['xclbin_size_bytes']/1024:.1f} KB)
"""
    else:  # >20% overhead
        recommendation = f"""❌ **USE SIMPLE KERNEL IN PRODUCTION**

The optimized kernel has **{time_overhead_pct:.1f}% overhead** ({time_overhead_abs:.2f} µs per frame), which is **too significant** for production use.

**Justification**:
- Simple kernel is faster: {simple['mean_time_us']:.2f} µs vs {optimized['mean_time_us']:.2f} µs
- Better throughput: {simple['frames_per_second']:,.0f} fps
- Smaller XCLBIN: {simple['xclbin_size_bytes']/1024:.1f} KB vs {optimized['xclbin_size_bytes']/1024:.1f} KB
- Optimizations don't justify performance penalty
"""

    report += recommendation

    report += f"""
---

## Detailed Performance Metrics

### Processing Time

Processing time per frame (25ms of audio at 16kHz):

| Statistic | Simple Kernel | Optimized Kernel | Difference |
|-----------|---------------|------------------|------------|
| **Mean** | {simple['mean_time_us']:.2f} µs | {optimized['mean_time_us']:.2f} µs | {time_overhead_abs:+.2f} µs ({time_overhead_pct:+.1f}%) |
| **Std Dev** | {simple['std_time_us']:.2f} µs | {optimized['std_time_us']:.2f} µs | {optimized['std_time_us'] - simple['std_time_us']:+.2f} µs |
| **Median** | {simple['median_time']*1e6:.2f} µs | {optimized['median_time']*1e6:.2f} µs | {(optimized['median_time'] - simple['median_time'])*1e6:+.2f} µs |
| **Min** | {simple['min_time_us']:.2f} µs | {optimized['min_time_us']:.2f} µs | {optimized['min_time_us'] - simple['min_time_us']:+.2f} µs |
| **Max** | {simple['max_time_us']:.2f} µs | {optimized['max_time_us']:.2f} µs | {optimized['max_time_us'] - simple['max_time_us']:+.2f} µs |
| **CV (%)** | {simple_cv:.2f}% | {optimized_cv:.2f}% | {optimized_cv - simple_cv:+.2f}% |

**Coefficient of Variation (CV)**: Lower is better. Indicates timing consistency.

### Throughput

Frames processed per second:

| Kernel | Throughput | Audio Processed per Second |
|--------|------------|----------------------------|
| **Simple** | {simple['frames_per_second']:,.0f} fps | {(simple['frames_per_second'] * 25 / 1000):.1f} seconds |
| **Optimized** | {optimized['frames_per_second']:,.0f} fps | {(optimized['frames_per_second'] * 25 / 1000):.1f} seconds |

### Realtime Factor

How much faster than realtime (higher is better):

| Kernel | Realtime Factor | Audio Duration | Processing Time |
|--------|-----------------|----------------|-----------------|
| **Simple** | {simple['realtime_factor']:.1f}x | 25 ms | {simple['mean_time']*1000:.3f} ms |
| **Optimized** | {optimized['realtime_factor']:.1f}x | 25 ms | {optimized['mean_time']*1000:.3f} ms |

**Interpretation**: Both kernels process audio **much faster than realtime**, suitable for production use.

### Memory Footprint

XCLBIN file sizes:

| Kernel | Size (bytes) | Size (KB) | Ratio |
|--------|--------------|-----------|-------|
| **Simple** | {simple['xclbin_size_bytes']:,} | {simple['xclbin_size_bytes']/1024:.2f} | 1.00x |
| **Optimized** | {optimized['xclbin_size_bytes']:,} | {optimized['xclbin_size_bytes']/1024:.2f} | {size_ratio:.2f}x |

---

## Statistical Analysis

### Timing Distribution

**Simple Kernel**:
- Mean ± Std: {simple['mean_time_us']:.2f} ± {simple['std_time_us']:.2f} µs
- Range: [{simple['min_time_us']:.2f}, {simple['max_time_us']:.2f}] µs
- Spread: {simple['max_time_us'] - simple['min_time_us']:.2f} µs
- Coefficient of Variation: {simple_cv:.2f}%

**Optimized Kernel**:
- Mean ± Std: {optimized['mean_time_us']:.2f} ± {optimized['std_time_us']:.2f} µs
- Range: [{optimized['min_time_us']:.2f}, {optimized['max_time_us']:.2f}] µs
- Spread: {optimized['max_time_us'] - optimized['min_time_us']:.2f} µs
- Coefficient of Variation: {optimized_cv:.2f}%

### Variance Analysis

"""

    if optimized_cv < simple_cv:
        variance_analysis = f"""The optimized kernel has **{simple_cv - optimized_cv:.2f}% lower variance** than the simple kernel, indicating more consistent performance.

**Production Impact**: More predictable latency characteristics, better for real-time applications.
"""
    else:
        variance_analysis = f"""The simple kernel has **{optimized_cv - simple_cv:.2f}% lower variance** than the optimized kernel, indicating more consistent performance.

**Production Impact**: Simple kernel provides more predictable latency characteristics.
"""

    report += variance_analysis

    report += f"""
---

## Performance Breakdown

### NPU Execution Time vs Overhead

Estimated breakdown of processing time:

**Simple Kernel**:
- NPU Compute: ~{simple['estimated_npu_utilization_pct']:.1f}% of time
- CPU/DMA Overhead: ~{100 - simple['estimated_npu_utilization_pct']:.1f}% of time

**Optimized Kernel**:
- NPU Compute: ~{optimized['estimated_npu_utilization_pct']:.1f}% of time
- CPU/DMA Overhead: ~{100 - optimized['estimated_npu_utilization_pct']:.1f}% of time

**Note**: These are rough estimates. Actual NPU utilization requires hardware profiling.

### Expected Performance at Scale

Processing 1 hour of audio:

| Kernel | Frames | Total Time | Realtime Factor |
|--------|--------|------------|-----------------|
| **Simple** | 144,000 | {(144000 / simple['frames_per_second']):.2f} seconds | {simple['realtime_factor']:.1f}x |
| **Optimized** | 144,000 | {(144000 / optimized['frames_per_second']):.2f} seconds | {optimized['realtime_factor']:.1f}x |

---

## Comparison vs Target Performance

### Target: 220x Realtime (from UC-Meeting-Ops)

Current performance vs target:

| Kernel | Current RTF | Target RTF | Progress |
|--------|-------------|------------|----------|
| **Simple** | {simple['realtime_factor']:.1f}x | 220x | {(simple['realtime_factor'] / 220 * 100):.1f}% |
| **Optimized** | {optimized['realtime_factor']:.1f}x | 220x | {(optimized['realtime_factor'] / 220 * 100):.1f}% |

**Note**: The 220x target is for the **complete Whisper pipeline** (mel + encoder + decoder), not just mel spectrogram preprocessing.

### Current Status

Both kernels are executing on NPU hardware but with **empty compute kernels** (passthrough). The measured times represent:
- XRT kernel invocation overhead
- DMA transfer time (input: 800 bytes, output: 80 bytes)
- NPU setup/teardown
- **No actual mel computation yet**

**Next Step**: Implement actual mel spectrogram computation in C++ kernel to achieve target performance.

---

## Production Deployment Recommendations

### Recommended Kernel
"""

    if time_overhead_pct < 10:
        prod_recommendation = f"""**OPTIMIZED KERNEL** - Use in production

**Reasons**:
1. {"Better" if time_overhead_pct < 0 else "Similar"} performance ({time_overhead_pct:+.1f}% difference)
2. {f"More consistent ({optimized_cv:.2f}% CV vs {simple_cv:.2f}% CV)" if optimized_cv < simple_cv else "Acceptable consistency"}
3. Ready for additional optimizations
4. Size increase ({size_ratio:.2f}x) is acceptable
"""
    elif time_overhead_pct < 20:
        prod_recommendation = f"""**OPTIMIZED KERNEL (WITH MONITORING)** - Use with caution

**Reasons**:
1. Acceptable overhead ({time_overhead_pct:.1f}%)
2. Monitor performance in production
3. Evaluate if optimizations provide value
4. Consider simple kernel if performance is critical
"""
    else:
        prod_recommendation = f"""**SIMPLE KERNEL** - Use in production

**Reasons**:
1. Better performance ({-time_overhead_pct:.1f}% faster)
2. Smaller size ({simple['xclbin_size_bytes']/1024:.1f} KB)
3. More efficient resource usage
4. Optimized kernel overhead too high ({time_overhead_pct:.1f}%)
"""

    report += prod_recommendation

    report += f"""
### Integration Steps

1. **Select Kernel**:
   ```bash
   # Simple kernel
   XCLBIN_PATH="build/mel_simple.xclbin"

   # Optimized kernel
   XCLBIN_PATH="build/mel_int8_optimized.xclbin"
   ```

2. **Initialize NPU**:
   ```python
   import pyxrt as xrt

   device = xrt.device(0)
   xclbin = xrt.xclbin(XCLBIN_PATH)
   device.register_xclbin(xclbin)
   ```

3. **Process Audio**:
   ```python
   # Process 400 INT16 samples at a time
   # Expected throughput: {simple['frames_per_second'] if time_overhead_pct > 10 else optimized['frames_per_second']:,.0f} fps
   ```

### Performance Monitoring

Track these metrics in production:

1. **Processing Time**: Should match benchmark ({simple['mean_time_us'] if time_overhead_pct > 10 else optimized['mean_time_us']:.2f} µs ± {simple['std_time_us'] if time_overhead_pct > 10 else optimized['std_time_us']:.2f} µs)
2. **Throughput**: Should achieve {simple['frames_per_second'] if time_overhead_pct > 10 else optimized['frames_per_second']:,.0f} fps
3. **Variance**: Should stay within {simple_cv if time_overhead_pct > 10 else optimized_cv:.2f}% CV
4. **NPU Utilization**: Monitor with `xrt-smi`

### Scaling Characteristics

Based on benchmark results:

- **Single Stream**: {simple['frames_per_second'] if time_overhead_pct > 10 else optimized['frames_per_second']:,.0f} fps
- **Expected Latency**: {simple['mean_time_us'] if time_overhead_pct > 10 else optimized['mean_time_us']:.2f} µs per frame
- **Memory per Stream**: {simple['xclbin_size_bytes']/1024 if time_overhead_pct > 10 else optimized['xclbin_size_bytes']/1024:.2f} KB XCLBIN + buffers

---

## Benchmark Methodology

### Test Configuration

- **Hardware**: AMD Ryzen 9 8945HS with Phoenix NPU (XDNA1)
- **NPU**: 4×6 tile array, 16 TOPS INT8
- **XRT Version**: 2.20.0
- **Audio Format**: INT16, 16kHz sample rate
- **Frame Size**: 400 samples (25ms)
- **Iterations**: {simple['num_iterations']} (after {10} warmup iterations)

### Timing Methodology

1. **Warmup**: {10} iterations to stabilize NPU and cache
2. **Measurement**: {simple['num_iterations']} timed iterations using `time.perf_counter()`
3. **Precision**: Microsecond-level timing resolution
4. **Overhead Included**: XRT kernel invocation, DMA transfers, buffer management

### What is Measured

- **Kernel Execution**: Time from kernel launch to completion
- **DMA Transfers**: Input buffer (800 bytes) and output buffer (80 bytes)
- **XRT Overhead**: Driver overhead for NPU communication
- **NOT Measured**: Host-side audio preprocessing, result parsing

### Limitations

- **Empty Kernels**: Current kernels are passthroughs (no actual mel computation)
- **Single Thread**: Benchmarks run single-threaded
- **No Batching**: Processing one frame at a time
- **Ideal Conditions**: No system load, dedicated NPU access

---

## Next Steps

### Immediate (Phase 1)

1. ✅ Infrastructure complete (NPU execution working)
2. ✅ Performance baseline established
3. 🔄 **Implement mel computation in C++ kernel**
4. 🔄 Measure performance with real computation

### Short-term (Phases 2-3)

1. Optimize mel kernel for NPU tile array
2. Implement vectorized operations
3. Add memory optimization
4. Target: 80x realtime for mel stage

### Long-term (Phases 4-6)

1. Integrate with Whisper encoder
2. Implement decoder on NPU
3. Full pipeline optimization
4. Target: 220x realtime complete system

---

## Appendix: Raw Benchmark Data

### Simple Kernel

```json
{{
  "mean_time_us": {simple['mean_time_us']:.2f},
  "std_time_us": {simple['std_time_us']:.2f},
  "min_time_us": {simple['min_time_us']:.2f},
  "max_time_us": {simple['max_time_us']:.2f},
  "median_time_us": {simple['median_time']*1e6:.2f},
  "cv_percent": {simple_cv:.2f},
  "frames_per_second": {simple['frames_per_second']:.2f},
  "realtime_factor": {simple['realtime_factor']:.2f},
  "xclbin_size_bytes": {simple['xclbin_size_bytes']},
  "num_iterations": {simple['num_iterations']}
}}
```

### Optimized Kernel

```json
{{
  "mean_time_us": {optimized['mean_time_us']:.2f},
  "std_time_us": {optimized['std_time_us']:.2f},
  "min_time_us": {optimized['min_time_us']:.2f},
  "max_time_us": {optimized['max_time_us']:.2f},
  "median_time_us": {optimized['median_time']*1e6:.2f},
  "cv_percent": {optimized_cv:.2f},
  "frames_per_second": {optimized['frames_per_second']:.2f},
  "realtime_factor": {optimized['realtime_factor']:.2f},
  "xclbin_size_bytes": {optimized['xclbin_size_bytes']},
  "num_iterations": {optimized['num_iterations']}
}}
```

---

## Document Information

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Author**: Performance Metrics Lead (Team 3), Magic Unicorn Inc.
**Contact**: aaron@magicunicorn.tech
**Location**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels`

**Related Documents**:
- `CURRENT_STATUS_OCT27.md` - Current implementation status
- `NPU_MEL_KERNEL_BREAKTHROUGH_OCT27.md` - Breakthrough documentation
- `benchmark_results/` - Raw benchmark data and charts

---

**Magic Unicorn Unconventional Technology & Stuff Inc.**
*Advancing NPU Performance for Real-World AI Applications*
"""

    # Write report
    with open(output_file, 'w') as f:
        f.write(report)


def main():
    """Main report generation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate comprehensive performance report"
    )
    parser.add_argument(
        '--results',
        default='benchmark_results/performance_benchmarks.json',
        help='Path to benchmark results JSON'
    )
    parser.add_argument(
        '--output',
        default='PERFORMANCE_BENCHMARKS.md',
        help='Output markdown file'
    )

    args = parser.parse_args()

    results_file = Path(args.results)
    output_file = Path(args.output)

    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("Run benchmark first: python3 benchmark_performance.py")
        sys.exit(1)

    print("=" * 70)
    print("GENERATING PERFORMANCE REPORT")
    print("=" * 70)
    print()

    # Load results
    results = load_results(results_file)
    print(f"✅ Loaded results from: {results_file}")

    # Generate report
    print(f"📝 Generating report: {output_file}")
    generate_report(results, output_file)
    print(f"✅ Report generated: {output_file}")

    # Get file size
    report_size = output_file.stat().st_size
    print(f"   Size: {report_size:,} bytes ({report_size/1024:.1f} KB)")

    print()
    print("=" * 70)
    print("REPORT COMPLETE!")
    print("=" * 70)
    print()
    print(f"📄 View report: {output_file}")
    print()
    print("Report includes:")
    print("  ✅ Executive summary with recommendations")
    print("  ✅ Detailed performance metrics")
    print("  ✅ Statistical analysis")
    print("  ✅ Comparison and overhead analysis")
    print("  ✅ Production deployment guidance")
    print("  ✅ Benchmark methodology")
    print()


if __name__ == '__main__':
    main()
