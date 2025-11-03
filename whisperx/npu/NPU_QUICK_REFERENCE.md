# NPU Kernel Integration Quick Reference
**Last Updated**: October 29, 2025

---

## WHAT'S ALREADY BUILT (27 XCLBINs Ready to Use)

### Mel Spectrogram (PRODUCTION READY) ⭐
- **Best XCLBIN**: `npu_optimization/mel_kernels/build_fixed_v3/mel_fixed_v3_PRODUCTION_v1.0.xclbin`
- **Accuracy**: FFT perfect (1.0000), Mel 0.38% error vs librosa
- **Source**: `npu_optimization/mel_kernels/mel_kernel_fft_fixed_PRODUCTION_v1.0.c`
- **Test**: `run python test_mel_npu_execution.py`
- **Alternative**: 18 other mel XCLBINs available for benchmarking

### Whisper Encoder Kernels (COMPILED BUT NOT INTEGRATED)
- **Attention**: `whisper_encoder_kernels/build_attention/attention_simple.xclbin`
- **MatMul**: `whisper_encoder_kernels/build/matmul_simple.xclbin`
- **Source Files**: `attention_int8.c`, `matmul_int8.c` (INT8 optimized)
- **Status**: Can run independently, not yet hooked to WhisperX encoder

### Test/Infrastructure (3 Kernels)
- **Latest**: `npu_optimization/build/final.xclbin` and variants
- **Use**: Validate XRT setup and NPU device access

---

## FASTEST WAY TO INTEGRATE (2-3 hours)

### Step 1: Copy Production Mel Kernel
```bash
cp /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/build_fixed_v3/mel_fixed_v3_PRODUCTION_v1.0.xclbin \
   /your/deployment/location/mel_npu.xclbin
```

### Step 2: Update Mel Preprocessing to Use NPU
```python
# In npu_mel_preprocessing.py or similar:
from npu_runtime_fixed import NPURuntime

mel_processor = NPURuntime(
    xclbin_path="mel_npu.xclbin",
    device="/dev/accel/accel0"
)

# In your transcription pipeline:
mel_features = mel_processor.compute_mel_spectrogram(audio)
# Now uses NPU instead of CPU librosa!
```

### Step 3: Test Integration
```bash
# Run this to validate everything works:
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization
python test_whisper_with_fixed_mel.py
```

### Step 4: Deploy
```bash
# Use the provided server with auto-detection:
python /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/server_whisperx_npu.py
# Auto-detects NPU and mel kernel, falls back to CPU if unavailable
```

---

## FILE LOCATIONS (Copy-Paste Ready)

### Main XCLBIN Files
```
Mel (RECOMMENDED):
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/build_fixed_v3/mel_fixed_v3_PRODUCTION_v1.0.xclbin

Encoder Kernels:
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build_attention/attention_simple.xclbin
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build/matmul_simple.xclbin
```

### Source Code for Recompilation
```
Mel Kernel Source:
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/mel_kernel_fft_fixed_PRODUCTION_v1.0.c

Encoder Kernels Source:
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/attention_int8.c
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/matmul_int8.c
```

### MLIR Definitions (If Recompiling)
```
Mel MLIR:
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/mel_fixed_v3.mlir
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/build_fixed_v3/mel_fixed_v3.mlir

Encoder MLIR:
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/attention_simple.mlir
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/matmul_simple.mlir
```

### Runtime Python Files (Already Integrated)
```
Main NPU Runtime:
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_runtime_aie2.py
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_runtime_fixed.py

Integration Frameworks:
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisperx_npu_integration.py
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/unified_stt_diarization.py
```

### Test Scripts
```
Main Mel Test:
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/test_mel_npu_execution.py

Full Pipeline Test:
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/test_whisper_with_fixed_mel.py

Quick Test:
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/test_npu_simple.py
```

---

## EXPECTED PERFORMANCE

### Current (With NPU Mel Kernel Only)
- **Speed**: 5-10x realtime (vs 1x CPU-only)
- **Power**: 10-15W (vs 45W CPU)
- **Bottleneck**: ONNX encoder/decoder still on CPU

### Potential (If You Also Integrate Encoder Kernels)
- **Speed**: 30-50x realtime
- **Power**: 5-10W
- **Status**: Encoder kernels compiled and ready, just need to hook up

### Target (Full Custom Kernels - Phase 5)
- **Speed**: 220x realtime (proven in UC-Meeting-Ops)
- **Power**: 5W
- **Timeline**: 8-10 weeks additional work

---

## WHAT EACH COMPONENT DOES

### Mel Spectrogram Kernel
```
Audio PCM (CPU) 
  ↓ (DMA to NPU)
[NPU: FFT (1024-point), Mel Filterbank (80 channels)]
  ↓ (DMA to CPU)
Mel Features (float32, shape: [T, 80])
```
**Effect**: Eliminates librosa CPU bottleneck, accelerates feature extraction 20-30x

### Attention Kernel (Future Use)
```
[Query, Key, Value] from encoder layer
  ↓ (DMA to NPU)
[NPU: Q@K^T, Softmax, Attention@V]
  ↓ (DMA to CPU)
Attention Output
```
**Effect**: Would accelerate encoder 5-10x more, but needs WhisperX encoder modification

### MatMul Kernel (Future Use)
```
[Matrix A, Matrix B] from encoder layers
  ↓ (DMA to NPU)
[NPU: INT8 Quantized MatMul]
  ↓ (DMA to CPU)
[Matrix Result]
```
**Effect**: Would accelerate all feed-forward layers 5-10x more

---

## TROUBLESHOOTING

### "No device found" error
```bash
# Check NPU is available:
ls -la /dev/accel/accel0
# Should exist

# If not, install XRT:
bash /opt/xilinx/xrt/bin/install-xrt-prebuilt.sh
```

### XCLBIN fails to load
```bash
# Verify XCLBIN is accessible:
file /path/to/mel_fixed_v3_PRODUCTION_v1.0.xclbin
# Should say "data"

# Test with this Python code:
import xrt
device = xrt.xrt_device(0)
xclbin = xrt.xclbin("/path/to/kernel.xclbin")
print("XCLBIN loaded successfully!")
```

### Wrong accuracy from mel kernel
```bash
# Verify you're using PRODUCTION version:
ls -lh /path/to/mel_fixed_v3_PRODUCTION_v1.0.xclbin

# Test with this:
python test_mel_npu_execution.py
# Should show: FFT Correlation = 1.0000
#              Mel Error = 0.38%
```

### Performance degradation
- Make sure you're using NPU, not CPU fallback
- Check `/dev/accel/accel0` is accessible
- Profile with: `python npu_benchmark.py`

---

## DOCUMENTATION ROADMAP

**For Quick Setup** (you are here):
- This file: NPU_QUICK_REFERENCE.md

**For Understanding What We Have**:
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/WHISPERX_NPU_INVENTORY.md`
- 12-section comprehensive inventory of all 27 XCLBINs, 100+ sources, etc.

**For Integration Details**:
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/NPU_INTEGRATION_COMPLETE.md`
- Full setup, API documentation, examples

**For Performance Tuning**:
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/COMPLETE_XCLBIN_DOCUMENTATION.md`
- Detailed performance analysis and optimization tips

**For Compilation from Source**:
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/COMPILATION_SUCCESS.md`
- How to rebuild XCLBINs if needed

---

## NEXT ACTIONS

### Immediate (Do this first)
1. Run: `python /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/test_npu_simple.py`
2. Verify NPU is detected and working
3. Run: `python /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/test_mel_npu_execution.py`
4. Verify mel kernel loads and computes correctly

### Short-term (This week)
1. Integrate mel kernel into main WhisperX pipeline
2. Run benchmarks: `python npu_benchmark.py`
3. Measure actual performance improvement on your audio

### Medium-term (Next 1-2 weeks)
1. Consider integrating encoder kernels (attention + matmul)
2. Run full transcription benchmarks
3. Set up monitoring and metrics

### Long-term (Weeks 4+)
1. Explore Phase 2 optimizations (full custom kernels)
2. Measure power consumption improvements
3. Plan for batch processing support

---

## KEY STATS AT A GLANCE

| Metric | Value | Status |
|--------|-------|--------|
| XCLBINs Ready | 27 | ✅ |
| Source Files | 100+ | ✅ |
| Test Scripts | 32 | ✅ |
| Mel Accuracy | 1.0000 (FFT), 0.38% (Mel) | ✅ |
| Encoder Kernels | Compiled, not integrated | ⚠️ |
| Overall Readiness | 75% | 🎯 |
| Time to Basic Integration | 2-3 hours | ⏱️ |
| Time to Full Integration | 1-2 weeks | 📅 |
| Performance Target | 220x realtime | 🚀 |

---

## ONE-LINER DEPLOYMENT

```bash
# Copy kernel, update config, test, deploy in one shot:
cp /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/build_fixed_v3/mel_fixed_v3_PRODUCTION_v1.0.xclbin /your/deployment/ && \
python /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/test_mel_npu_execution.py && \
python /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/server_whisperx_npu.py --port 9004
```

---

**Questions?** See the full inventory at: `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/WHISPERX_NPU_INVENTORY.md`
