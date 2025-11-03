# NPU Whisper Benchmark Suite

Comprehensive benchmarking and validation framework for tracking progress toward **220x realtime transcription** target on AMD Phoenix NPU.

## Overview

This benchmark suite provides:

- **Individual Kernel Benchmarks**: Measure performance of Attention, LayerNorm, GELU, and MatMul kernels
- **End-to-End Pipeline Benchmarks**: Test complete encoder pipeline with various audio lengths
- **Accuracy Validation**: Validate NPU outputs against CPU reference implementations
- **Optimization Comparison**: Compare different optimization strategies
- **Comprehensive Reporting**: Generate markdown and JSON reports with recommendations

## Quick Start

### Run Full Benchmark Suite

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels
python3 run_all_benchmarks.py
```

This will:
1. Benchmark all kernels (100 iterations each)
2. Test pipeline with audio lengths: 10s, 30s, 60s, 120s, 300s
3. Validate accuracy against CPU reference
4. Compare optimization approaches
5. Generate comprehensive report in `benchmark_results/`

### Quick Mode (Faster Testing)

```bash
python3 run_all_benchmarks.py --quick
```

Runs fewer iterations (20 instead of 100) and tests only 30s audio.

### Skip Accuracy Validation

```bash
python3 run_all_benchmarks.py --skip-accuracy
```

Useful when you only want performance metrics.

### Custom Output Directory

```bash
python3 run_all_benchmarks.py --output-dir my_results/
```

## Individual Components

### 1. Kernel Benchmarks

```python
from benchmark_suite import KernelBenchmark

benchmark = KernelBenchmark(num_iterations=100)
results = benchmark.benchmark_all_kernels()

# Individual kernels
attention_results = benchmark.benchmark_attention()
layernorm_results = benchmark.benchmark_layernorm()
gelu_results = benchmark.benchmark_gelu()
matmul_results = benchmark.benchmark_matmul()
```

**Output**: Mean, std, min, max, P50, P95, P99 for each kernel

### 2. Pipeline Benchmarks

```python
from benchmark_suite import PipelineBenchmark

benchmark = PipelineBenchmark()

# Test single audio length
result = benchmark.benchmark_encoder(audio_length_seconds=30)

# Test multiple lengths
results = benchmark.benchmark_multiple_lengths([10, 30, 60, 120, 300])

# Benchmark single encoder block
block_result = benchmark.benchmark_encoder_block(num_iterations=10)
```

**Output**: Mel time, encoder time, total time, realtime factor

### 3. Accuracy Validation

```python
from benchmark_suite import AccuracyBenchmark

benchmark = AccuracyBenchmark()
results = benchmark.validate_all_kernels()

# Individual validations
attention_accuracy = benchmark.validate_attention_accuracy()
layernorm_accuracy = benchmark.validate_layernorm_accuracy()
gelu_accuracy = benchmark.validate_gelu_accuracy()
matmul_accuracy = benchmark.validate_matmul_accuracy()
```

**Output**: Correlation, MSE, Max Difference, MAE, Pass/Fail status

### 4. Optimization Comparison

```python
from benchmark_suite import BenchmarkComparison

benchmark = BenchmarkComparison()

# Compare optimization strategies
results = benchmark.compare_optimizations()

# Compare tile sizes
tile_results = benchmark.compare_tile_sizes()
```

**Output**: Performance impact of each optimization approach

### 5. Report Generation

```python
from benchmark_suite import BenchmarkReport

report = BenchmarkReport()

# Generate markdown report
report.generate_markdown_report(all_results, "report.md")

# Generate JSON report
report.generate_json_report(all_results, "report.json")
```

## Configuration Files

Located in `configs/`:

### baseline.yaml
- No optimizations
- Baseline performance measurement
- Use for comparison

### optimized.yaml
- Current best optimizations
- Buffer reuse + DMA optimization
- Current state: 14.0x realtime

### target.yaml
- All optimizations enabled
- Target: 220x realtime
- Includes multi-core, batching, pipeline overlap

## Output Files

After running benchmarks, you'll find in `benchmark_results/`:

```
benchmark_results/
├── BENCHMARK_REPORT_LATEST.md          # Main report (symlink)
├── BENCHMARK_REPORT_20251030_143022.md # Timestamped report
├── benchmark_report_latest.json        # JSON data (symlink)
├── benchmark_report_20251030_143022.json
├── kernel_results_20251030_143022.json
├── pipeline_results_20251030_143022.json
├── accuracy_results_20251030_143022.json
└── comparison_results_20251030_143022.json
```

## Understanding the Reports

### Performance Summary

Shows:
- Current realtime factor (e.g., 14.0x)
- Target realtime factor (220x)
- Progress percentage
- Gap to target

### Kernel Performance

Detailed statistics for each kernel:
- Mean execution time
- Standard deviation
- Percentiles (P50, P95, P99)
- Min/Max times

### Accuracy Validation

Validates each kernel against CPU reference:
- **Correlation**: Should be > 0.95
- **MSE**: Mean squared error
- **Max Difference**: Largest discrepancy
- **MAE**: Mean absolute error

### Optimization Comparison

Compares different optimization approaches:
- Baseline (no optimizations)
- Buffer optimized
- Batched
- Multi-core
- Fully optimized

Shows speedup vs baseline for each approach.

### Progress to Target

Visual progress bar showing:
- Current vs target realtime factor
- Milestone progress
- Recommendations for next optimizations

## Recommendations

The report automatically generates recommendations based on current performance:

**< 20x realtime**:
- Implement larger matmul tiles (16×16 → 64×64)
- Optimize buffer management

**20-80x realtime**:
- Implement batch processing
- Optimize attention kernel (current bottleneck)

**80-180x realtime**:
- Enable multi-core NPU (4×6 tile array)
- Pipeline optimization (overlap DMA with compute)

**180-220x realtime**:
- Fine-tune parameters
- Profile and optimize remaining hotspots

## Performance Targets

| Phase | Target RTF | Key Optimization | Expected Speedup |
|-------|------------|------------------|------------------|
| Current | 14.0x | Buffer reuse | 1.0x (baseline) |
| Phase 1 | 60x | Larger tiles (64×64 matmul) | 4.3x |
| Phase 2 | 120x | Batch processing | 2.0x |
| Phase 3 | 180x | Multi-core NPU | 1.5x |
| Phase 4 | 220x | Full pipeline optimization | 1.2x |

**Total Expected Speedup**: 15.7x (from 14x to 220x)

## Hardware Information

**Platform**: AMD Ryzen 9 8945HS with Phoenix NPU
**NPU**: 4×6 tile array (24 compute cores)
**Performance**: 16 TOPS INT8
**XRT**: 2.20.0
**Firmware**: 1.5.5.391

## Troubleshooting

### NPU Device Not Found

```bash
# Check NPU device
ls -l /dev/accel/accel0

# Verify XRT installation
/opt/xilinx/xrt/bin/xrt-smi examine
```

### Import Errors

```bash
# Ensure XRT Python bindings are available
export PYTHONPATH=/opt/xilinx/xrt/python:$PYTHONPATH

# Or add to script:
import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
```

### Low Performance

Check:
1. NPU firmware is up to date
2. No other processes using NPU
3. Sufficient system memory available
4. XRT environment variables set correctly

## Development

### Adding New Benchmarks

1. Create new benchmark class in `benchmark_suite/`
2. Inherit from base benchmark pattern
3. Add to `__init__.py`
4. Update `run_all_benchmarks.py` to include new benchmark

### Modifying Reports

Edit `benchmark_report.py`:
- `_generate_*()` methods for each section
- Customize markdown formatting
- Add new metrics or visualizations

## References

- **Main Documentation**: `../MATMUL_INTEGRATION_COMPLETE.md`
- **Encoder Block**: `../test_encoder_block.py`
- **Performance Report**: `../SESSION_PROGRESS_OCT30.md`

## License

Copyright 2025 Magic Unicorn Unconventional Technology & Stuff Inc.

## Support

For issues or questions, see project documentation or contact development team.

---

**Last Updated**: October 30, 2025
**Version**: 1.0.0
