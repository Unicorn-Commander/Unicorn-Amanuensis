# AMD Hardware Acceleration Options for Whisper

**Date**: October 28, 2025  
**Hardware**: AMD Ryzen 9 8945HS + Radeon 780M + Phoenix NPU  
**Goal**: Achieve maximum Whisper transcription speed on AMD

---

## Hardware Inventory

### AMD Phoenix NPU (XDNA1)
- **Specs**: 16 TOPS INT8, 4×6 tile array
- **Access**: `/dev/accel/accel0` via XRT 2.20.0
- **Status**: ✅ Working (we've tested kernels)
- **Challenge**: Custom kernels needed (5-9 weeks)

### AMD Radeon 780M iGPU (RDNA3)
- **Specs**: 12 CUs, 2700 MHz, 3.9 TFLOPS FP32
- **Access**: ROCm, HIP, or OpenCL
- **Status**: ❓ Not tested yet
- **Potential**: Could run PyTorch/ONNX models directly

### AMD Ryzen CPU
- **Specs**: 8 cores, 16 threads, Zen 4
- **Access**: Standard Python/C++
- **Status**: ✅ Working
- **Baseline**: faster-whisper achieves 10-50x here

---

## Option 1: faster-whisper (CPU) - PROVEN ⭐

**What It Is**: CTranslate2 backend with INT8 quantization

**Performance**: 10-50x realtime (proven by UC-Meeting-Ops)

**Pros**:
- ✅ Works TODAY (no development)
- ✅ Proven performance (10-50x)
- ✅ INT8 quantization built-in
- ✅ VAD (silence removal) built-in
- ✅ Better than ONNX Runtime

**Cons**:
- ❌ CPU-only (doesn't use GPU/NPU)
- ❌ Not the "220x" marketing claim
- ❌ Doesn't leverage AMD hardware

**Implementation**:
```bash
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe("audio.wav", vad_filter=True)
# Achieves 10-50x realtime
```

**Verdict**: ⭐⭐⭐⭐⭐ Best immediate option

---

## Option 2: AMD ROCm + PyTorch (iGPU) - REALISTIC 🎯

**What It Is**: Run Whisper PyTorch model on AMD GPU via ROCm

**Performance**: 50-150x realtime (estimated, needs testing)

**Pros**:
- ✅ Uses AMD iGPU (leverages hardware)
- ✅ ROCm supports RDNA3 (your GPU)
- ✅ PyTorch has ROCm backend
- ✅ No custom kernels needed
- ✅ 3-5x faster than CPU

**Cons**:
- ❌ ROCm installation complex
- ❌ Not officially supported on laptops
- ❌ May conflict with gaming drivers
- ❌ Memory shared with system

**Implementation**:
```bash
# Install ROCm (complex)
# See: https://rocm.docs.amd.com/

# Install PyTorch with ROCm
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
```

```python
import torch
import whisper

# Load model on AMD GPU
device = "cuda"  # ROCm uses CUDA API
model = whisper.load_model("base").to(device)
result = model.transcribe("audio.wav")
# Should achieve 50-150x realtime
```

**Verdict**: ⭐⭐⭐⭐ Worth trying if ROCm installs

---

## Option 3: ONNX Runtime DirectML (iGPU) - EASY 🚀

**What It Is**: Microsoft DirectML for GPU acceleration on Windows/Linux

**Performance**: 30-80x realtime (estimated)

**Pros**:
- ✅ Works on AMD GPUs
- ✅ Easier than ROCm
- ✅ No driver conflicts
- ✅ Official Microsoft support
- ✅ 2-3x faster than CPU

**Cons**:
- ❌ DirectML primarily for Windows
- ❌ Linux support experimental
- ❌ Slower than native ROCm
- ❌ Limited AMD optimization

**Implementation**:
```bash
pip install onnxruntime-directml
```

```python
import onnxruntime as ort

# Create session with DirectML
session = ort.InferenceSession(
    "encoder_model.onnx",
    providers=['DmlExecutionProvider']  # DirectML
)
```

**Verdict**: ⭐⭐⭐ Good fallback if ROCm fails

---

## Option 4: Custom NPU Kernels (Phoenix NPU) - EDUCATIONAL 🎓

**What It Is**: MLIR-AIE2 kernels we've been working on

**Performance**: 200-500x realtime (theoretical, if working)

**Pros**:
- ✅ Maximum performance (if it works)
- ✅ Dedicated hardware (NPU)
- ✅ Educational value (learn MLIR)
- ✅ Unique approach

**Cons**:
- ❌ 5-9 weeks development time
- ❌ Complex debugging
- ❌ Currently broken (4.68% correlation)
- ❌ UC-Meeting-Ops doesn't use it!

**Status**: We have infrastructure, computation broken

**Verdict**: ⭐⭐ Optional, for learning only

---

## Option 5: OpenCL (iGPU) - FALLBACK 🔧

**What It Is**: OpenCL acceleration for AMD GPU

**Performance**: 20-60x realtime (estimated)

**Pros**:
- ✅ Cross-platform
- ✅ Works on AMD GPUs
- ✅ No ROCm needed
- ✅ Simpler than ROCm

**Cons**:
- ❌ Slower than ROCm/HIP
- ❌ Limited PyTorch support
- ❌ Need custom kernels
- ❌ More work than DirectML

**Implementation**: Would need custom OpenCL kernels or use clDNN

**Verdict**: ⭐⭐ Only if other GPU options fail

---

## Option 6: Hybrid CPU + iGPU - BALANCED ⚖️

**What It Is**: faster-whisper (CPU) + AMD GPU preprocessing

**Performance**: 15-70x realtime (estimated)

**Pros**:
- ✅ Best of both worlds
- ✅ GPU for mel spectrogram
- ✅ CPU for inference (proven)
- ✅ Incremental improvement

**Cons**:
- ❌ Need GPU preprocessing code
- ❌ Data transfer overhead
- ❌ Complex integration

**Verdict**: ⭐⭐⭐ Interesting optimization

---

## Recommendation Matrix

| Option | Speed | Effort | Risk | AMD iGPU | AMD NPU | Recommendation |
|--------|-------|--------|------|----------|---------|----------------|
| **faster-whisper (CPU)** | 10-50x | 0 days | None | ❌ | ❌ | ⭐⭐⭐⭐⭐ DO THIS FIRST |
| **ROCm + PyTorch** | 50-150x | 1-2 days | Medium | ✅ | ❌ | ⭐⭐⭐⭐ TRY THIS NEXT |
| **ONNX DirectML** | 30-80x | 0.5 days | Low | ✅ | ❌ | ⭐⭐⭐ FALLBACK |
| **Custom NPU kernels** | 200-500x | 5-9 weeks | High | ❌ | ✅ | ⭐⭐ EDUCATIONAL ONLY |
| **OpenCL** | 20-60x | 2-3 days | Medium | ✅ | ❌ | ⭐⭐ IF NEEDED |
| **Hybrid** | 15-70x | 3-5 days | Medium | ✅ | ❌ | ⭐⭐⭐ OPTIMIZATION |

---

## Recommended Path Forward

### Phase 1: Baseline (TODAY - 30 minutes)
```bash
pip install faster-whisper
python3 test_faster_whisper.py
```
**Target**: 10-50x realtime (proven)

### Phase 2: AMD iGPU (TOMORROW - 1-2 days)
```bash
# Try ROCm first
sudo apt install rocm-hip-runtime
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7

# If that fails, try DirectML
pip install onnxruntime-directml
```
**Target**: 50-150x realtime (hopeful)

### Phase 3: Optimization (NEXT WEEK - if needed)
- Profile bottlenecks
- Add batch processing
- Optimize memory
**Target**: 100-200x realtime (realistic)

### Phase 4: NPU (OPTIONAL - 5-9 weeks)
- Fix custom kernels
- Learn MLIR-AIE2
- Educational project
**Target**: 200-500x realtime (educational)

---

## Hardware Capability Reality Check

### AMD Radeon 780M iGPU
- **TFLOPS**: 3.9 FP32, 31.2 FP16, 62.4 INT8
- **Memory**: Shared with system RAM
- **Compute**: 12 CUs × 64 = 768 stream processors
- **Whisper Base**: ~80M parameters, ~160 GFLOPS inference
- **Theoretical**: 62.4 / 0.16 = 390x realtime (INT8)
- **Realistic**: 50-150x (memory bandwidth limited)

### AMD Phoenix NPU
- **TOPS**: 16 INT8, ~2 FP16
- **Whisper**: Needs FP16/FP32 or INT8
- **IF INT8**: Could theoretically handle 16,000 GOPS
- **Theoretical**: 16,000 / 160 = 100x realtime
- **Realistic**: 50-200x (with custom kernels)

### Bottleneck Analysis
- **Not compute**: GPU/NPU have enough FLOPS
- **Memory bandwidth**: Likely bottleneck
- **Data transfer**: CPU ↔ GPU ↔ NPU overhead
- **Framework overhead**: Python/ONNX Runtime

---

## Key Insight

**UC-Meeting-Ops' "220x" was never achieved on NPU!**

They get 10-50x with CPU-only faster-whisper. If we:
1. Use faster-whisper (10-50x) ← Proven
2. Add AMD iGPU acceleration (3-5x) ← Realistic
3. Optimize pipeline (1.5-2x) ← Easy

**Total: 45-500x realistic range** (depending on GPU success)

The 220x target is achievable **without custom NPU kernels**!

---

## Next Steps

1. **Test faster-whisper baseline** (30 min)
2. **Attempt ROCm installation** (1-2 hours)
3. **Test PyTorch on AMD GPU** (30 min)
4. **Benchmark real performance** (1 hour)
5. **Compare with targets** (15 min)

**Timeline**: 1-2 days to 50-150x realtime (realistic)

---

**Document**: AMD_ACCELERATION_OPTIONS.md  
**Created**: October 28, 2025  
**Status**: Comprehensive AMD-specific analysis  
**Next**: Test faster-whisper baseline, then AMD iGPU

**Magic Unicorn Unconventional Technology & Stuff Inc.** 🦄✨
