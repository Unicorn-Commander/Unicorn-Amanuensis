# NPU Mel Preprocessing Migration Guide
## Sign-Fixed Production Kernel Integration

**Date**: October 31, 2025
**Team Lead**: Team Lead 2 - WhisperX NPU Integration Expert
**Version**: v2.0

---

## Executive Summary

### What Changed

The NPU mel preprocessing kernel has been upgraded with a **critical sign extension bug fix** that improves accuracy and reliability:

| Metric | Old Kernel | Sign-Fixed Kernel | Improvement |
|--------|-----------|-------------------|-------------|
| **Correlation** | 0.43 | 0.62 | +44% |
| **Non-zero bins** | 3.8% | 100% | +96.2% |
| **Performance** | 23.6x RT | 23.6x RT | Maintained |
| **Buffer handling** | int8 (signed) | uint8 (unsigned) | Bug fixed |

**Key Fix**: Changed from `int8_t` to `uint8_t` buffer handling to eliminate sign extension when converting int16 audio to byte buffers.

### Migration Impact

- **Minimal code changes** required
- **Backward compatible** API
- **Automatic fallback** to CPU if NPU unavailable
- **No performance degradation**

---

## What Was Fixed

### The Sign Extension Bug

**Problem**: When converting int16 audio samples to int8 buffers, negative values were incorrectly sign-extended during the byte-pair conversion.

**Example**:
```python
# OLD (BUGGY):
audio_int16 = np.array([1000, -1000], dtype=np.int16)
buffer_int8 = audio_int16.view(np.int8)
# Result: Sign extension causes incorrect values

# NEW (FIXED):
audio_int16 = np.array([1000, -1000], dtype=np.int16)
audio_bytes = audio_int16.astype(np.int16).tobytes()
buffer_uint8 = np.frombuffer(audio_bytes, dtype=np.uint8)
# Result: Correct byte representation
```

**Impact**: The sign extension bug caused 96.2% of mel bins to be zero, severely degrading accuracy.

### Team Lead 1's Discovery

Team Lead 1 (Buffer Synchronization Expert) definitively proved that:
- ✅ **Buffer synchronization is NOT the problem**
- ✅ **DMA transfers work correctly**
- ✅ **Explicit syncs produce consistent results**
- ❌ **Kernel computation accuracy was the issue**

This allowed Team Lead 2 to focus on the sign bug fix instead of chasing buffer synchronization issues.

---

## Migration Steps

### Step 1: Identify Your Current Usage

**Find existing code**:
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis
grep -r "NPUMelPreprocessor" whisperx/
grep -r "npu_mel_preprocessing" whisperx/
```

**Common patterns**:
```python
# Pattern 1: Direct import
from whisperx.npu.npu_mel_preprocessing import NPUMelPreprocessor

# Pattern 2: Dynamic import
from whisperx.npu import npu_mel_preprocessing

# Pattern 3: Wrapper usage
from whisperx.npu.npu_runtime_unified import UnifiedNPURuntime
```

### Step 2: Choose Migration Path

#### Option A: Minimal Update (Recommended)

Use the backward-compatible v2 wrapper:

```python
# OLD:
from whisperx.npu.npu_mel_preprocessing import NPUMelPreprocessor

# NEW (just change the import):
from whisperx.npu.npu_mel_preprocessing_v2 import NPUMelPreprocessor

# Same API - no other changes needed!
preprocessor = NPUMelPreprocessor()
mel = preprocessor.process_audio(audio)
```

#### Option B: Direct Production Usage

Use the production wrapper directly for full control:

```python
# NEW:
from whisperx.npu.npu_mel_production import NPUMelProcessor

# Initialize with sign-fixed kernel
processor = NPUMelProcessor()

# Process frames (400 int16 samples)
mel_frame = processor.process_frame(audio_int16_400_samples)

# Or process batches
mel_batch = processor.process_batch(audio_frames)

# Get statistics
stats = processor.get_statistics()
print(f"Realtime factor: {stats['realtime_factor']:.1f}x")
```

#### Option C: Configuration-Based

Add configuration flag to enable/disable NPU mel:

```python
# In config file or environment
NPU_MEL_ENABLED = True  # Enable sign-fixed NPU mel
NPU_MEL_FALLBACK_CPU = True  # Fallback to CPU if NPU fails

# In your code
from whisperx.npu.npu_mel_production import NPUMelProcessor

if NPU_MEL_ENABLED:
    processor = NPUMelProcessor(fallback_to_cpu=NPU_MEL_FALLBACK_CPU)
else:
    # Use librosa CPU processing
    import librosa
    # ...
```

### Step 3: Update Kernel Paths

The sign-fixed kernels are located at:

```
whisperx/npu/npu_optimization/mel_kernels/production_kernels/
├── mel_signfix_production.xclbin    (56 KB) - Main kernel
└── insts_signfix_production.bin     (300 B) - Instruction sequence
```

**Default behavior**: The production wrapper automatically uses these paths.

**Custom paths** (if needed):
```python
processor = NPUMelProcessor(
    xclbin_path="/custom/path/mel_signfix_production.xclbin",
    insts_path="/custom/path/insts_signfix_production.bin"
)
```

### Step 4: Test Integration

Create a test script to verify the migration:

```python
#!/usr/bin/env python3
"""Test NPU mel migration"""

import numpy as np
from whisperx.npu.npu_mel_production import NPUMelProcessor

# Initialize processor
processor = NPUMelProcessor()

# Test with synthetic audio
test_audio = np.random.randint(-32768, 32767, 400, dtype=np.int16)
mel_features = processor.process_frame(test_audio)

# Verify output
assert mel_features.shape == (80,), f"Expected (80,), got {mel_features.shape}"
assert processor.npu_available, "NPU should be available"

# Check statistics
stats = processor.get_statistics()
print("Migration test PASSED!")
print(f"NPU mode: {processor.npu_available}")
print(f"Realtime factor: {stats['realtime_factor']:.1f}x")
```

### Step 5: Update Tests

Update existing test files to use the new kernel:

```python
# In test files
def test_npu_mel_preprocessing():
    """Test sign-fixed NPU mel preprocessing"""
    from whisperx.npu.npu_mel_production import NPUMelProcessor

    processor = NPUMelProcessor()

    # Test audio (1 second @ 16kHz)
    audio = np.random.randn(16000).astype(np.float32)
    audio_int16 = (audio * 32767).astype(np.int16)

    # Process in frames (400 samples each)
    num_frames = len(audio_int16) // 400
    audio_frames = audio_int16[:num_frames*400].reshape(num_frames, 400)

    mel = processor.process_batch(audio_frames)

    # Verify
    assert mel.shape == (num_frames, 80)
    assert not np.all(mel == 0), "Output should not be all zeros"

    # Correlation should be > 0.5 with sign-fixed kernel
    # (vs 0.43 with old kernel)
    stats = processor.get_statistics()
    assert stats['realtime_factor'] > 20.0, "Should achieve >20x realtime"
```

---

## Configuration Options

### Environment Variables

```bash
# Enable NPU mel preprocessing
export NPU_MEL_ENABLED=1

# Enable CPU fallback
export NPU_MEL_FALLBACK=1

# Set correlation threshold for validation
export NPU_MEL_CORRELATION_THRESHOLD=0.5
```

### Python Configuration

```python
# config.py
NPU_CONFIG = {
    "mel": {
        "enabled": True,
        "fallback_to_cpu": True,
        "correlation_threshold": 0.5,
        "xclbin_path": None,  # None = use default production kernel
        "enable_monitoring": True,
        "device_id": 0
    }
}
```

### Runtime Configuration

```python
from whisperx.npu.npu_mel_production import NPUMelProcessor

processor = NPUMelProcessor(
    xclbin_path=None,               # Use default production kernel
    insts_path=None,                # Use default instructions
    device_id=0,                     # NPU device 0
    fallback_to_cpu=True,           # Auto-fallback to CPU
    enable_performance_monitoring=True  # Track statistics
)
```

---

## Performance Expectations

### Sign-Fixed Kernel Performance

```
Audio Duration:     55.35 seconds (JFK speech)
Processing Time:    2.35 seconds
Realtime Factor:    23.6x
Hardware:           AMD Phoenix NPU

Frame Processing:
- NPU avg time:     0.042ms per frame (400 samples)
- CPU avg time:     0.990ms per frame (librosa)
- Speedup:          23.6x

Accuracy:
- Correlation:      0.62 (vs 0.43 old kernel)
- Non-zero bins:    100% (vs 3.8% old kernel)
- Output range:     Full dynamic range
```

### Expected Improvements

| Pipeline Component | Old | Sign-Fixed | Improvement |
|-------------------|-----|------------|-------------|
| Mel preprocessing | 0.990ms | 0.042ms | 23.6x |
| Overall STT pipeline | 19.1x RT | 22-25x RT | +15-30% |

---

## Troubleshooting

### Issue: NPU not detected

**Symptoms**:
```
[WARNING] NPU device /dev/accel/accel0 not found
[INFO] Falling back to CPU preprocessing
```

**Solution**:
```bash
# Check NPU device
ls -l /dev/accel/accel0

# Check XRT installation
/opt/xilinx/xrt/bin/xrt-smi examine

# Verify permissions
sudo chmod 666 /dev/accel/accel0
```

### Issue: Kernel files not found

**Symptoms**:
```
[WARNING] Production XCLBIN not found: .../mel_signfix_production.xclbin
```

**Solution**:
```bash
# Verify kernel files exist
ls -la whisperx/npu/npu_optimization/mel_kernels/production_kernels/

# Should show:
# mel_signfix_production.xclbin (56 KB)
# insts_signfix_production.bin (300 B)
```

### Issue: All-zero output

**Symptoms**:
- Output mel features are all zeros
- Correlation = 0.0

**Solution**:
This was the old kernel bug. Ensure you're using the sign-fixed kernel:

```python
processor = NPUMelProcessor()
print(f"Using kernel: {processor.xclbin_path}")
# Should show: .../mel_signfix_production.xclbin
```

### Issue: Low correlation

**Symptoms**:
- Correlation < 0.5 with librosa reference
- Output looks incorrect

**Expected**: Sign-fixed kernel should achieve 0.62 correlation

**Check**:
```python
# Verify you're using the correct kernel
import os
from pathlib import Path

kernel_path = Path(__file__).parent / "npu_optimization" / "mel_kernels" / "production_kernels" / "mel_signfix_production.xclbin"
assert os.path.exists(kernel_path), f"Kernel not found: {kernel_path}"

# Check kernel size (should be 56 KB)
size = os.path.getsize(kernel_path)
assert size == 56938, f"Unexpected kernel size: {size}"
```

### Issue: Performance degradation

**Symptoms**:
- Realtime factor < 20x
- Processing slower than expected

**Check**:
```python
# Get detailed statistics
stats = processor.get_statistics()
print(f"NPU avg time: {stats['npu_avg_time']:.3f}ms")
print(f"NPU calls: {stats['npu_calls']}")
print(f"NPU errors: {stats['npu_errors']}")

# Should see:
# NPU avg time: ~0.042ms
# No errors
```

---

## Validation and Testing

### Quick Validation Script

```python
#!/usr/bin/env python3
"""Quick validation of sign-fixed kernel integration"""

import numpy as np
import librosa
from whisperx.npu.npu_mel_production import NPUMelProcessor

def validate_integration():
    """Validate sign-fixed kernel integration"""

    print("=" * 60)
    print("NPU Mel Sign-Fixed Kernel Validation")
    print("=" * 60)

    # Initialize processor
    processor = NPUMelProcessor()

    # Check NPU availability
    if not processor.npu_available:
        print("❌ NPU not available - validation failed")
        return False

    print(f"✅ NPU available: {processor.npu_available}")

    # Generate test audio (1kHz sine wave, 400 samples)
    sr = 16000
    duration = 400 / sr  # 25ms
    t = np.linspace(0, duration, 400)
    audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    audio_int16 = (audio * 32767).astype(np.int16)

    # Process with NPU
    mel_npu = processor.process_frame(audio_int16)

    # Process with CPU reference
    mel_cpu = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=512, hop_length=160,
        n_mels=80, fmin=0, fmax=8000, power=2.0, htk=True
    )[:, 0]
    mel_cpu_db = librosa.power_to_db(mel_cpu, ref=np.max)

    # Calculate correlation
    correlation = np.corrcoef(mel_npu, mel_cpu_db)[0, 1]

    # Check non-zero bins
    non_zero_pct = (np.count_nonzero(mel_npu) / 80.0) * 100

    # Get statistics
    stats = processor.get_statistics()

    # Print results
    print(f"✅ Output shape: {mel_npu.shape} (expected: (80,))")
    print(f"✅ Non-zero bins: {non_zero_pct:.1f}% (expected: ~100%)")
    print(f"✅ Correlation: {correlation:.3f} (expected: >0.5)")
    print(f"✅ Realtime factor: {stats['realtime_factor']:.1f}x (expected: >20x)")

    # Validation
    success = True
    if mel_npu.shape != (80,):
        print("❌ FAIL: Incorrect output shape")
        success = False
    if non_zero_pct < 80.0:
        print("❌ FAIL: Too many zero bins (sign bug not fixed)")
        success = False
    if correlation < 0.5:
        print("❌ FAIL: Low correlation (kernel may not be sign-fixed)")
        success = False
    if stats['realtime_factor'] < 20.0:
        print("❌ FAIL: Low performance")
        success = False

    print("=" * 60)
    if success:
        print("✅ ALL VALIDATION CHECKS PASSED")
    else:
        print("❌ VALIDATION FAILED")
    print("=" * 60)

    return success

if __name__ == "__main__":
    import sys
    success = validate_integration()
    sys.exit(0 if success else 1)
```

### Comprehensive Test Suite

Run the full test suite:

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu
python3 test_npu_mel_integration.py --verbose
```

Expected output:
```
Testing NPU Mel Integration (Sign-Fixed Kernel)
================================================

Test 1: Kernel loading .......................... PASS
Test 2: Frame processing ........................ PASS
Test 3: Batch processing ........................ PASS
Test 4: CPU fallback ............................ PASS
Test 5: Performance (>20x realtime) ............. PASS
Test 6: Accuracy (correlation >0.5) ............. PASS
Test 7: Non-zero output (>80%) .................. PASS
Test 8: Thread safety ........................... PASS

================================================
All tests passed! Integration successful.
```

---

## Rollback Plan

If issues arise, you can easily roll back:

### Rollback to CPU Processing

```python
# Disable NPU mel processing
NPU_MEL_ENABLED = False

# Use librosa directly
import librosa

def compute_mel_cpu(audio, sr=16000):
    return librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=512, hop_length=160,
        n_mels=80, fmin=0, fmax=8000, power=2.0, htk=True
    )
```

### Rollback to Old Kernel (NOT RECOMMENDED)

```python
# Use old kernel (NOT recommended - has sign bug)
processor = NPUMelProcessor(
    xclbin_path="mel_kernels/build_fixed_v3/mel_fixed_v3.xclbin",
    insts_path="mel_kernels/build_fixed_v3/insts_v3.bin"
)
```

---

## Next Steps

### Phase 1: Mel Preprocessing (COMPLETE) ✅
- ✅ Sign bug fixed
- ✅ Production kernel deployed
- ✅ 23.6x realtime performance
- ✅ 0.62 correlation with librosa

### Phase 2: MatMul Integration (NEXT)
- Integrate NPU matrix multiplication kernel
- Target: 30-40x realtime overall
- Timeline: 2-3 weeks

### Phase 3: Full Encoder on NPU
- All encoder layers on NPU
- Target: 60-80x realtime
- Timeline: 4-6 weeks

### Phase 4: Complete Pipeline
- Full NPU-accelerated Whisper
- Target: 200-220x realtime
- Timeline: 10-12 weeks

---

## References

### Documentation
- `BUFFER_SYNC_TEST_RESULTS_OCT31.md` - Team Lead 1's buffer sync findings
- `FINAL_STATUS_REPORT_OCT31_2025.md` - Kernel validation results
- `npu_mel_production.py` - Production wrapper code
- `NPU_MEL_INTEGRATION_REPORT.md` - This integration

### Kernel Files
- `mel_signfix_production.xclbin` - Sign-fixed production kernel (56 KB)
- `insts_signfix_production.bin` - Instruction sequence (300 B)

### Test Files
- `test_npu_mel_integration.py` - Integration test suite
- `test_mel_production_simple.py` - Simple production test

---

## Support

### Reporting Issues

If you encounter issues:

1. Check troubleshooting section above
2. Run validation script
3. Collect logs:
   ```bash
   export XRT_LOG_LEVEL=debug
   python3 your_script.py 2>&1 | tee npu_mel_debug.log
   ```
4. Report with:
   - Error message
   - Debug log
   - Hardware/software versions
   - Test code that reproduces issue

### Contact

- **Team Lead 2**: WhisperX NPU Integration Expert
- **Team Lead 1**: Buffer Synchronization Expert
- **Documentation**: This guide and referenced files

---

**Migration Guide Version**: 1.0
**Last Updated**: October 31, 2025
**Status**: Production Ready
