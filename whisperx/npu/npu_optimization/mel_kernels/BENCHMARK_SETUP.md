# NPU Mel Spectrogram Accuracy Benchmarking Setup

## Quick Start

### 1. Install Dependencies

```bash
# Install required Python packages
pip install librosa scipy matplotlib

# Optional: Install in virtual environment
python3 -m venv venv_benchmark
source venv_benchmark/bin/activate
pip install librosa scipy matplotlib numpy
```

### 2. Run Complete Benchmark Suite

```bash
# Make script executable
chmod +x run_full_benchmark.sh

# Run complete suite (generates tests, benchmarks, visualizes, reports)
./run_full_benchmark.sh
```

### 3. View Results

```bash
# Read accuracy report
cat ACCURACY_REPORT.md

# View plots
ls -l benchmark_results/plots/

# Check JSON data
cat benchmark_results/benchmark_results.json
```

---

## Manual Step-by-Step Execution

If you prefer to run steps individually:

### Step 1: Generate Test Signals

```bash
python3 generate_test_signals.py
```

**Output**: `test_audio/` directory with ~20 test audio files (800 bytes each)

**Test Types**:
- Pure tones (100 Hz, 250 Hz, 500 Hz, 1000 Hz, 2000 Hz, 3000 Hz, 4000 Hz, 6000 Hz)
- Chirps (frequency sweeps)
- Noise (white, pink, brown)
- Edge cases (silence, DC offset, impulse, step)
- Multi-tone combinations
- Clipping test
- Very quiet signal

### Step 2: Run Accuracy Benchmark

```bash
python3 benchmark_accuracy.py \
  --test-dir test_audio \
  --xclbin build_fixed/mel_fixed.xclbin \
  --output-dir benchmark_results
```

**Output**: `benchmark_results/benchmark_results.json`

**What it does**:
- Loads each test audio file
- Runs NPU mel spectrogram computation
- Runs CPU reference (librosa) for comparison
- Computes accuracy metrics (correlation, MSE, SNR, etc.)
- Saves detailed results

### Step 3: Generate Visual Comparisons

```bash
python3 visual_comparison.py \
  --results benchmark_results/benchmark_results.json \
  --output-dir benchmark_results/plots
```

**Output**: `benchmark_results/plots/` directory with comparison plots

**Plots Generated**:
- Individual test comparisons (NPU vs CPU side-by-side)
- Difference maps
- Aggregate accuracy analysis
- Per-bin error distribution

### Step 4: Generate Accuracy Report

```bash
python3 accuracy_report.py \
  --results benchmark_results/benchmark_results.json \
  --output ACCURACY_REPORT.md
```

**Output**: `ACCURACY_REPORT.md` - Comprehensive markdown report

**Report Includes**:
- Executive summary with verdict
- Detailed test results tables
- Statistical analysis
- Error analysis and recommendations
- Visual comparison references
- Production deployment guidance

---

## Requirements

### Python Packages

| Package | Version | Purpose | Install Command |
|---------|---------|---------|-----------------|
| **librosa** | ≥0.10.0 | CPU mel spectrogram reference | `pip install librosa` |
| **scipy** | ≥1.9.0 | Statistical metrics (Pearson correlation) | `pip install scipy` |
| **matplotlib** | ≥3.5.0 | Visualization plots | `pip install matplotlib` |
| **numpy** | ≥1.21.0 | Numerical operations | `pip install numpy` |
| **pyxrt** | 2.20.0 | XRT Python bindings (NPU access) | Installed with XRT |

### Hardware

- **NPU**: AMD Phoenix NPU (XDNA1) at `/dev/accel/accel0`
- **XRT**: Version 2.20.0 installed
- **XCLBIN**: `build_fixed/mel_fixed.xclbin` compiled and ready

### Files Required

```
mel_kernels/
├── build_fixed/
│   ├── mel_fixed.xclbin       - NPU executable (16 KB)
│   └── insts_fixed.bin         - Instruction sequence (300 bytes)
├── generate_test_signals.py    - Test signal generator
├── benchmark_accuracy.py       - Main benchmarking script
├── visual_comparison.py        - Visualization generator
├── accuracy_report.py          - Report generator
└── run_full_benchmark.sh       - Complete automated suite
```

---

## Expected Results

### Excellent Performance

- **Correlation**: >99%
- **MSE**: <0.01
- **SNR**: >40 dB
- **Verdict**: PASS (Production Ready)

### Good Performance

- **Correlation**: >95%
- **MSE**: <0.1
- **SNR**: >30 dB
- **Verdict**: PASS (Production Ready with minor optimizations)

### Acceptable Performance

- **Correlation**: >90%
- **MSE**: <1.0
- **SNR**: >20 dB
- **Verdict**: MARGINAL (Some tuning recommended)

---

## Troubleshooting

### "librosa not found"

```bash
pip install librosa

# If using system Python
pip3 install librosa

# If permission denied
pip install --user librosa
```

### "XRT device not found"

```bash
# Check NPU device
ls -l /dev/accel/accel0

# Check XRT
xrt-smi examine

# Verify XRT Python bindings
python3 -c "import sys; sys.path.insert(0, '/opt/xilinx/xrt/python'); import pyxrt"
```

### "XCLBIN not found"

```bash
# Check if XCLBIN exists
ls -l build_fixed/mel_fixed.xclbin

# If missing, rebuild NPU kernel
cd build_fixed
# Run appropriate build script for your setup
```

### "Kernel execution failed"

Check NPU status:
```bash
xrt-smi examine
dmesg | tail -20  # Check for NPU errors
```

### "matplotlib backend error"

```bash
# Use non-interactive backend
export MPLBACKEND=Agg
python3 visual_comparison.py
```

---

## Output Files

### Generated Directories

```
mel_kernels/
├── test_audio/                     - Test audio files
│   ├── tone_100hz.raw              - Pure 100 Hz tone
│   ├── tone_1000hz.raw             - Pure 1 kHz tone
│   ├── chirp_100_4000hz.raw        - Frequency sweep
│   ├── white_noise.raw             - White noise
│   └── ... (20+ test files)
│
├── benchmark_results/              - Benchmark outputs
│   ├── benchmark_results.json      - Detailed metrics (JSON)
│   └── plots/                      - Visualization plots
│       ├── tone_1000hz_comparison.png
│       ├── aggregate_analysis.png
│       └── ... (20+ plots)
│
└── ACCURACY_REPORT.md              - Comprehensive report
```

### File Sizes

- Test audio files: 800 bytes each (~20 files = 16 KB)
- Benchmark JSON: ~50-100 KB
- Plots: ~100-200 KB each
- Accuracy report: ~20-30 KB

**Total disk usage**: ~5-10 MB

---

## Advanced Usage

### Custom Test Audio

To test with your own audio:

```python
import numpy as np

# Create custom audio (400 INT16 samples)
audio = np.array([...], dtype=np.int16)  # 400 samples

# Save as raw file
with open('test_audio/custom.raw', 'wb') as f:
    f.write(audio.tobytes())

# Run benchmark
python3 benchmark_accuracy.py
```

### Batch Testing

Test multiple XCLBIN versions:

```bash
for xclbin in build_*/mel_*.xclbin; do
    echo "Testing: $xclbin"
    python3 benchmark_accuracy.py --xclbin "$xclbin" --output-dir "results_$(basename $xclbin .xclbin)"
done
```

### Headless Server

Run without display (for remote servers):

```bash
export MPLBACKEND=Agg
./run_full_benchmark.sh
```

---

## Performance Benchmarking

To measure NPU performance (separate from accuracy):

```bash
# Time 1000 iterations
time python3 -c "
import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
from benchmark_accuracy import NPUMelBenchmark
import numpy as np

benchmark = NPUMelBenchmark()
audio = np.random.randint(-16000, 16000, 400, dtype=np.int16)

for i in range(1000):
    benchmark.run_npu(audio)
"
```

---

## Validation Metrics Explained

### Correlation (Pearson)

- **What**: Measures linear relationship between NPU and CPU outputs
- **Range**: -1 to +1 (1 = perfect correlation)
- **Target**: >0.99 (excellent), >0.95 (good)
- **Formula**: `corr(X, Y) = cov(X, Y) / (σ_X × σ_Y)`

### Mean Squared Error (MSE)

- **What**: Average squared difference between NPU and CPU
- **Range**: 0 to ∞ (0 = perfect match)
- **Target**: <0.01 (excellent), <0.1 (good)
- **Formula**: `MSE = (1/n) Σ(NPU_i - CPU_i)²`

### Signal-to-Noise Ratio (SNR)

- **What**: Ratio of signal power to error power
- **Range**: 0 to ∞ dB (higher = better)
- **Target**: >40 dB (excellent), >30 dB (good)
- **Formula**: `SNR = 10 log₁₀(signal_power / noise_power)`

### Mean Absolute Error (MAE)

- **What**: Average absolute difference
- **Range**: 0 to ∞ (0 = perfect match)
- **Target**: <1.0 (excellent), <5.0 (good)
- **Formula**: `MAE = (1/n) Σ|NPU_i - CPU_i|`

---

## Next Steps After Validation

### If Accuracy is EXCELLENT (>99% correlation)

1. ✅ **Production Ready!**
2. Integrate with WhisperX pipeline
3. Benchmark end-to-end performance
4. Deploy to production

### If Accuracy is GOOD (>95% correlation)

1. ✅ **Nearly Production Ready**
2. Optional: Implement proper mel filterbank (1-2% improvement)
3. Test on real speech audio
4. Deploy with monitoring

### If Accuracy is MARGINAL (90-95% correlation)

1. ⚠️ **Optimization Recommended**
2. Implement triangular mel filterbank (critical)
3. Add log compression for dynamic range
4. Re-validate accuracy
5. Then deploy

### If Accuracy is LOW (<90% correlation)

1. ❌ **Improvements Required**
2. Review Q15 implementation for precision issues
3. Implement proper mel filterbank
4. Add log compression
5. Consider INT16 output instead of INT8
6. Re-benchmark thoroughly

---

## Contact & Support

**Project**: Unicorn Amanuensis
**Organization**: Magic Unicorn Unconventional Technology & Stuff Inc.
**GitHub**: https://github.com/Unicorn-Commander/Unicorn-Amanuensis

**Documentation**:
- `FFT_NPU_SUCCESS.md` - NPU implementation details
- `FIXED_POINT_IMPLEMENTATION_REPORT.md` - Q15 FFT design
- `ACCURACY_REPORT.md` - Generated benchmark report (after running)

---

**Created**: October 28, 2025
**Author**: Claude + Validation Engineering Team
**Version**: 1.0
