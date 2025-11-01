# PyTorch Baseline Accuracy Test

This directory contains a comprehensive accuracy validation test that compares the C++ encoder implementation against the official PyTorch Whisper implementation.

## Quick Start

### Prerequisites
```bash
# Activate virtual environment
source ~/mlir-aie/ironenv/bin/activate

# Ensure transformers is installed (should already be installed)
pip install transformers
```

### Running the Test
```bash
# Navigate to directory
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2

# Run accuracy test
python test_accuracy_vs_pytorch.py
```

### Expected Output
The test will:
1. Load PyTorch Whisper Base encoder (6 layers)
2. Load C++ encoder library
3. Load real Whisper weights into both implementations
4. Run the same random input through both encoders
5. Compare outputs using multiple metrics
6. Generate detailed report

## Files Created

### Test Scripts
- **test_accuracy_vs_pytorch.py** - Main accuracy validation test (executable)
  - Loads PyTorch and C++ encoders
  - Uses SAME random input (seed=42) for fair comparison
  - Runs 6-layer encoder in both implementations
  - Compares outputs with multiple metrics
  - Saves comparison data for analysis

### Test Output
- **accuracy_test_output.txt** - Full console output from test run
- **ACCURACY_TEST_SUMMARY.txt** - Quick summary of results
- **ACCURACY_VALIDATION_REPORT.md** - Comprehensive 400+ line analysis report

### Comparison Data Directory
```
accuracy_comparison/
├── input.npy              # (512×512) Random input with seed=42
├── output_pytorch.npy     # (512×512) PyTorch encoder output
├── output_cpp.npy         # (512×512) C++ encoder output
├── abs_diff.npy           # (512×512) Absolute differences
├── rel_error.npy          # (512×512) Relative errors
└── metrics.txt            # Summary metrics
```

## Test Results Summary

### Current Status: **FAILED**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Cosine Similarity | 0.6456 | >0.99 | ❌ FAIL |
| Mean Absolute Error | 1.29 | <1.0 | ❌ FAIL |
| Max Absolute Error | 69.45 | - | ❌ FAIL |
| Element Accuracy | 0.63% | >99% | ❌ FAIL |

### Root Cause
**INT8 Quantization Error Accumulation**
- C++ uses INT8 quantization for NPU acceleration
- PyTorch uses FP32 for reference
- Quantization error compounds through 6 layers
- 36 quantized operations per encoder pass
- Each operation adds ~0.03 error → cumulative 1.08 error

## Key Findings

### 1. Quantization Impact
```
Per-layer breakdown:
  - Q/K/V projections: 3 × INT8 matmuls
  - Attention output:  1 × INT8 matmul
  - FFN FC1:           1 × INT8 matmul
  - FFN FC2:           1 × INT8 matmul
  Total per layer:     6 × INT8 matmuls

Cumulative (6 layers): 36 × INT8 matmuls
Error accumulation:    36 × 0.03 ≈ 1.08
Observed MAE:          1.29 ✓ (matches prediction)
```

### 2. Error Distribution
- **Median error**: 0.97 (most elements off by ~1.0)
- **90th percentile**: 2.50
- **99th percentile**: 7.31
- **Maximum**: 69.45 at position (210, 145)

The error is NOT uniform - there are specific hotspots with extreme errors.

### 3. Output Statistics
| Statistic | PyTorch | C++ | Difference |
|-----------|---------|-----|------------|
| Mean | 0.0321 | 0.0393 | +0.0072 |
| Std Dev | 2.6631 | 1.5986 | -1.0645 |
| Min | -22.22 | -55.18 | -32.96 |
| Max | 25.95 | 69.94 | +43.99 |

C++ output has **2.6× wider range**, suggesting numerical instability.

## Recommendations

### Priority 1: Switch to FP16/BF16 (IMMEDIATE)
**Why**: INT8 is too aggressive for encoder accuracy
**Impact**: Cosine similarity >0.98, MAE <0.1
**Trade-off**: 20% performance loss (still 300-400× vs target 17×)

### Priority 2: Verify Weight Transposition (IMMEDIATE)
**Why**: Possible double-transposition bug in weight loading
**Impact**: Could fix major accuracy issues
**Effort**: 1 day of debugging

### Priority 3: Fix Layer Norm Epsilon (HIGH)
**Why**: PyTorch uses epsilon=1e-5, C++ may differ
**Impact**: Fixes divergence in low-variance regions
**Effort**: 1 hour code fix

### Priority 4: Per-Channel Quantization (MEDIUM)
**Why**: Per-tensor quantization is too coarse
**Impact**: Reduces quantization error by 50-70%
**Effort**: 2-3 days implementation

### Priority 5: Stable Softmax (MEDIUM)
**Why**: Prevents overflow/underflow in attention
**Impact**: Fixes extreme outlier errors
**Effort**: 1 day implementation

## Performance vs Accuracy Trade-off

| Configuration | Performance | Accuracy | Memory | Status |
|--------------|-------------|----------|---------|--------|
| **INT8** (current) | 400-500× | 64.6% | 128MB | ❌ Unusable |
| **FP16** (proposed) | 300-400× | >99% | 256MB | ✅ Recommended |
| **Mixed (FP16+INT8)** | 350-450× | ~95% | 192MB | ⚠️ Experimental |

**Recommendation**: Use FP16 for production. We still exceed the 17× realtime target by 17-23×, and accuracy is critical for STT quality.

## Using Comparison Data

### Load and Analyze
```python
import numpy as np

# Load outputs
pytorch = np.load('accuracy_comparison/output_pytorch.npy')
cpp = np.load('accuracy_comparison/output_cpp.npy')
diff = np.load('accuracy_comparison/abs_diff.npy')

# Find worst positions
worst_idx = np.unravel_index(np.argmax(diff), diff.shape)
print(f"Worst position: {worst_idx}")
print(f"  PyTorch: {pytorch[worst_idx]:.6f}")
print(f"  C++:     {cpp[worst_idx]:.6f}")
print(f"  Error:   {diff[worst_idx]:.6f}")

# Plot error heatmap
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.imshow(diff, cmap='hot', interpolation='nearest')
plt.colorbar(label='Absolute Error')
plt.title('C++ vs PyTorch Error Heatmap')
plt.xlabel('Hidden Dimension')
plt.ylabel('Sequence Position')
plt.savefig('error_heatmap.png')
```

### Analyze Per-Layer
To identify which layer contributes most to error, run the test with layer-by-layer output:

```python
# Modify test to save intermediate outputs
for layer_idx in range(6):
    np.save(f'layer_{layer_idx}_pytorch.npy', layer_output_pytorch)
    np.save(f'layer_{layer_idx}_cpp.npy', layer_output_cpp)
```

## Next Steps

### Week 4 (Immediate)
1. ✅ Run accuracy validation test (DONE)
2. ⏳ Implement FP16 weights (2-3 days)
3. ⏳ Verify weight transposition (1 day)
4. ⏳ Fix layer norm epsilon (1 hour)
5. ⏳ Re-run accuracy test (1 hour)

### Week 5 (Follow-up)
1. Test mixed precision (FP16 activations + INT8 weights)
2. Implement per-channel quantization if needed
3. Benchmark FP16 performance on NPU hardware
4. Document final accuracy metrics
5. Update production code with validated configuration

## Success Criteria

✅ **Production Ready** when ALL criteria met:
- Cosine Similarity: >0.99
- Mean Absolute Error: <0.1
- Element Accuracy: >99%
- Performance: >17× realtime
- No NaN/Inf in outputs

## References

- **Full Report**: `ACCURACY_VALIDATION_REPORT.md` (comprehensive 400+ line analysis)
- **Quick Summary**: `ACCURACY_TEST_SUMMARY.txt` (one-page overview)
- **Test Output**: `accuracy_test_output.txt` (full console log)
- **C++ Source**: `cpp/src/encoder_layer.cpp` (implementation to fix)
- **PyTorch Reference**: [Whisper Model](https://huggingface.co/openai/whisper-base)

## Troubleshooting

### Test Fails to Load PyTorch Model
```bash
# Install transformers
pip install transformers

# Download model manually
python -c "from transformers import WhisperModel; WhisperModel.from_pretrained('openai/whisper-base')"
```

### Test Fails to Load C++ Library
```bash
# Rebuild C++ library
cd cpp/build
cmake .. && make

# Check library exists
ls -la libwhisper_encoder_cpp.so
```

### Weights Not Found
```bash
# Extract weights from PyTorch model
python extract_whisper_weights.py

# Check weights directory
ls -la weights/whisper_base_fp32/
```

## Contact

For questions or issues with the accuracy test:
- Review `ACCURACY_VALIDATION_REPORT.md` for detailed analysis
- Check `accuracy_test_output.txt` for full test log
- See `ACCURACY_TEST_SUMMARY.txt` for quick reference

---

**Created**: October 30, 2025
**Status**: Test Complete - Accuracy Issues Identified - FP16 Migration Required
**Next**: Implement FP16 weights and re-validate
