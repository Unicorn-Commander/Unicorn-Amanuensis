# Whisper NPU Transcription: End-to-End Test Plan

**Document Date**: November 19, 2025
**Target Platform**: AMD Phoenix NPU (XDNA1)
**Status**: Design Phase - Ready for Implementation

---

## 1. Test Pipeline Overview

The end-to-end test validates the complete Whisper NPU transcription pipeline from audio input to encoder output.

```
Audio File
    ↓
[Audio Loading] ────── Load WAV/MP3 at 16 kHz, mono
    ↓
[Mel Spectrogram] ───── Convert to (80, T) mel bins via FFT
    ↓
[NPU Encoder] ────────── Process through 6 transformer layers
    ↓
[Output Validation] ──── Compare with CPU reference implementation
    ↓
Results + Metrics
```

### Test Inputs
- **Audio Sources**: WAV files, 16 kHz mono, various durations (5s to 30s)
- **Ground Truth**: CPU reference (librosa mel-spectrograms, PyTorch encoder)
- **Models**: Whisper Base ONNX encoder (39.27 MB BF16, located at `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/`)

### Test Outputs
- Encoder hidden states: shape (batch_size, 1500, 512)
- Performance metrics: latency, throughput, memory usage
- Accuracy metrics: correlation with CPU reference

---

## 2. Dimension Flow Analysis

### Input Stage
```
Raw Audio: (16000 Hz × duration_seconds) samples
│ (Example: 5 sec = 80,000 samples)
↓
Mel Spectrogram Generation:
  - FFT window: 400 samples (25ms)
  - Hop length: 160 samples (10ms)
  - Num filters: 80 (mel-scale triangular)
  - Output: (80 mel_bins, ~3000 frames)
│ (For 30-second audio = 3000 frames)
↓
Conv1D Preprocessing:
  - Input: (80, 3000)
  - Conv2D layer 1: kernel (3,3), stride (2,2) → (384, 1500, 256)
  - Conv2D layer 2: kernel (3,3), stride (2,2) → (384, 750, 256)
  - Linear projection: (256) → (512 embedding_dim)
  - Positional encoding: add to all frames
  - Output: (512, 1500)
```

### Encoder Layers (6 total)
```
For each layer (0 to 5):

  Input: (seq_len=1500, d_model=512)

  ├─ Self-Attention:
  │  ├─ Q proj: (512, 512) weight × input → (batch, 1500, 512)
  │  ├─ K proj: (512, 512) weight × input → (batch, 1500, 512)
  │  ├─ V proj: (512, 512) weight × input → (batch, 1500, 512)
  │  ├─ Matmul Q·K^T: (1500, 512) × (512, 1500) → (1500, 1500)
  │  ├─ Softmax over sequence dimension
  │  ├─ Matmul attention·V: (1500, 1500) × (1500, 512) → (1500, 512)
  │  └─ O proj: (512, 512) weight × output → (1500, 512)
  │
  ├─ Feed-Forward Network:
  │  ├─ Dense1 (Linear): (512, 2048) weight × (1500, 512) → (1500, 2048)
  │  ├─ GELU activation
  │  └─ Dense2 (Linear): (2048, 512) weight × (1500, 2048) → (1500, 512)
  │
  ├─ Layer Normalization:
  │  ├─ Self-attention output → LayerNorm (512,)
  │  └─ FFN output → LayerNorm (512,)
  │
  └─ Output: (1500, 512) → next layer

Final encoder output: (batch_size, 1500, 512) hidden states
```

### Attention Shape Details
Each of 8 attention heads:
- Attention head size: 512 / 8 = 64
- Q, K, V per head: (1500, 64)
- Attention matrix per head: (1500, 1500)
- Output per head: (1500, 64)
- Concatenated: (1500, 512)

---

## 3. Test Scenarios

### Scenario 1: Short Audio (5 seconds)
```
Input File: short_sample.wav
Duration: 5 seconds
Samples: 80,000 (at 16 kHz)
Mel frames: ~500
Expected sequence length after preprocessing: 500
Expected encoder shape: (1, 500, 512)
Expected latency: 50-100ms on NPU
```

**Validation**:
1. Load audio successfully
2. Mel spectrogram shape correct: (80, 500)
3. After conv: (512, 500)
4. Encoder output shape: (1, 500, 512)
5. All values in valid range (FP32 not NaN/Inf)

### Scenario 2: Medium Audio (30 seconds)
```
Input File: medium_sample.wav
Duration: 30 seconds
Samples: 480,000 (at 16 kHz)
Mel frames: ~3000
Expected sequence length after preprocessing: 1500
Expected encoder shape: (1, 1500, 512)
Expected latency: 200-400ms on NPU
```

**Validation**:
1. Load audio successfully
2. Mel spectrogram shape correct: (80, 3000)
3. After conv and downsampling: (512, 1500)
4. Encoder output shape: (1, 1500, 512)
5. Memory usage within limits
6. All values in valid range

### Scenario 3: Batch Processing (3× 10-second audio)
```
Input Files: batch_1.wav, batch_2.wav, batch_3.wav
Batch size: 3
Per sample: 10 seconds, 160,000 samples
Mel frames per sample: ~1000
Expected output shape: (3, 1000, 512)
Expected latency: 150-300ms on NPU
```

**Validation**:
1. All three audio files loaded
2. Mel spectrograms stacked correctly
3. Batch processing on NPU (fewer kernel launches)
4. Output shapes: (3, 1000, 512)
5. Batching provides speedup over sequential (>1.5x on NPU)

---

## 4. Success Criteria

### Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| **Realtime Factor** (5s audio) | >2x RTF | To Achieve |
| **Realtime Factor** (30s audio) | >3x RTF | To Achieve |
| **Mel Spectrogram Speed** | 20-30x faster than CPU | To Achieve |
| **Encoder Speed** | 10-15x faster than CPU | To Achieve |
| **Batch Speedup** | >1.5x improvement | To Achieve |
| **NPU Utilization** | >70% | To Measure |

**Calculation Example** (5-second audio):
- Audio duration: 5.0 seconds
- Processing time target: 0.5-1.0 seconds
- Realtime factor: 5-10x (2x minimum = 2.5s max)

### Accuracy Thresholds

| Metric | Threshold | Notes |
|--------|-----------|-------|
| **Mel Spectrogram Correlation** | > 0.99 | Must match librosa precisely |
| **Encoder Output Correlation** | > 0.95 | BF16 quantization allows small variance |
| **Layer-by-layer Accuracy** | Each layer > 0.90 | Monitor at each transformer layer |
| **No NaN/Inf Values** | 0 occurrences | Invalid computations caught immediately |

**Validation Method**:
```python
# Pseudo-code
cpu_mel = librosa_mel_spectrogram(audio)
npu_mel = npu_mel_compute(audio)
correlation = np.corrcoef(cpu_mel.flatten(), npu_mel.flatten())[0,1]
assert correlation > 0.99

cpu_encoder_out = pytorch_encoder(cpu_mel)
npu_encoder_out = npu_encoder(npu_mel)
correlation = np.corrcoef(cpu_out.flatten(), npu_out.flatten())[0,1]
assert correlation > 0.95
```

### Memory Usage Limits

| Component | Target | Rationale |
|-----------|--------|-----------|
| **Weights (BF16)** | 39.27 MB | Fixed: Whisper Base quantized |
| **Activations** | < 200 MB | (1500, 512) × FP32 × 2 buffers |
| **NPU Local Memory** | < 64 MB | Phoenix NPU tile memory pool |
| **Host DMA Buffers** | < 256 MB | Input/output staging |

---

## 5. Known Limitations

### Not Yet Implemented
1. **Decoder**: Only encoder tested (outputs hidden states, not text)
2. **Text Generation**: No token generation or beam search
3. **Real Transcription**: Outputs encoder features, not actual text
4. **Streaming**: No streaming/chunking optimization yet
5. **Quantization Tuning**: Using default BF16, not optimized INT8

### Decoder Roadmap
- **Phase 1**: Test encoder accuracy and performance
- **Phase 2**: Integrate NPU-optimized decoder (separate)
- **Phase 3**: Full end-to-end text output (future)

### Architectural Constraints
- **Encoder-only focus**: Decoder is separate 26-layer module
- **Batch-first processing**: All dimensions are (batch, seq_len, features)
- **Fixed sequence length**: 1500 frames expected from preprocessing
- **No attention caching**: Full attention matrices computed per forward pass

---

## 6. Test Implementation Structure

### Directory Layout
```
whisper_encoder_kernels/
├── END_TO_END_TEST_PLAN.md ← This file
├── test_e2e_mel.py ────────── Test mel spectrogram kernel
├── test_e2e_encoder.py ────── Test encoder layers
├── test_e2e_full.py ──────── Full pipeline test with all scenarios
├── test_audio_data/ ──────── Test audio files
│   ├── short_5sec.wav ───── 5-second sample
│   ├── medium_30sec.wav ──── 30-second sample
│   └── batch_10sec_*.wav ─── Batch test audio
└── test_results/ ────────── Output metrics and validation
    ├── mel_spectrogram_results.json
    ├── encoder_layer_results.json
    └── performance_benchmarks.csv
```

### Test Execution Flow

```python
# Pseudo-code for main test runner

class WhisperNPUEndToEndTest:

    def __init__(self):
        self.load_cpu_reference()      # Load librosa, pytorch
        self.initialize_npu()           # Load XCLBIN, kernels
        self.load_weights()             # BF16 encoder weights

    def test_scenario_1_short_audio(self):
        audio = load_wav("short_5sec.wav")
        mel_npu = compute_mel_spectrogram_npu(audio)
        mel_cpu = compute_mel_spectrogram_cpu(audio)

        assert mel_npu.shape == (80, 500)
        assert correlation(mel_npu, mel_cpu) > 0.99

        hidden_npu = encoder_forward_npu(mel_npu)
        hidden_cpu = encoder_forward_cpu(mel_cpu)

        assert hidden_npu.shape == (1, 500, 512)
        assert correlation(hidden_npu, hidden_cpu) > 0.95

        measure_latency(mel_npu, hidden_npu)

    def test_scenario_2_medium_audio(self):
        # Similar to scenario 1, but with 30-second audio

    def test_scenario_3_batch_processing(self):
        # Test 3 × 10-second audio samples in batch

    def run_all_tests(self):
        self.test_scenario_1_short_audio()
        self.test_scenario_2_medium_audio()
        self.test_scenario_3_batch_processing()
        self.generate_report()
```

---

## 7. Component Checklist

### Prerequisites (Must Have)
- [x] NPU hardware (AMD Phoenix)
- [x] XRT 2.20.0 runtime
- [x] MLIR-AIE2 toolchain (for kernel compilation)
- [x] XCLBIN files compiled and available
- [x] Whisper encoder weights (BF16 format)
- [x] WhisperWeightLoader class implemented
- [x] Test audio samples prepared

### Kernel Components (Build Status)
- [ ] `mel_spectrogram.xclbin` - FFT + mel filterbank on NPU
- [x] Matrix multiply kernels - Available (64×64, 32×32 variants)
- [x] Attention kernels - Available (documented but may need tuning)
- [x] GELU/LayerNorm kernels - Available

### Test Infrastructure (Required)
- [ ] Audio loading utilities (librosa or similar)
- [ ] Mel spectrogram reference implementation (librosa)
- [ ] PyTorch encoder reference model
- [ ] Correlation/distance metrics for validation
- [ ] Latency measurement framework
- [ ] Memory profiling utilities

---

## 8. Next Steps After Successful Test

### Immediate (1-2 weeks)
1. Validate test results pass all success criteria
2. Profile per-layer accuracy and performance
3. Identify any bottlenecks in pipeline
4. Fine-tune kernel parameters

### Short-term (2-4 weeks)
1. Integrate NPU decoder module
2. Implement token generation
3. Test full end-to-end transcription (audio → text)
4. Validate against OpenAI Whisper reference

### Medium-term (4-8 weeks)
1. Implement streaming/chunked processing
2. Optimize INT8 quantization for weights
3. Add multi-batch pipelining
4. Production-ready hardening

### Long-term (8+ weeks)
1. Server integration with API endpoints
2. Containerization and deployment
3. Performance optimization to 220x target
4. Integration with WhisperX for diarization

---

## 9. Test Failure Scenarios

### If Mel Spectrogram Fails
**Symptoms**: Correlation < 0.99 with CPU reference
- Check FFT scaling factors (known issue in October 2025 fixes)
- Verify mel filterbank implementations (HTK vs Slaney)
- Validate frequency range (0-8000 Hz standard)
- **Fix**: Apply FIX_GUIDE in documentation (scaling corrections)

### If Encoder Accuracy Low
**Symptoms**: Correlation < 0.95 with CPU reference
- Check weight loading (correct BF16 conversion)
- Verify quantization isn't causing numerical issues
- Validate layer-by-layer outputs (identify which layer fails)
- Check attention softmax numerical stability
- **Fix**: Review ATTENTION_ACCURACY_FINDINGS.md documentation

### If Performance Below Target
**Symptoms**: Latency > 1.0s for 5-second audio
- Profile kernel execution (may need optimization)
- Check DMA transfer overhead
- Verify batch processing is actually batching
- Measure per-layer compute time
- **Fix**: Refer to DMA_OPTIMIZATION_SUMMARY.md and kernel profiling guides

### If Memory Overflow
**Symptoms**: OOM errors during encoder forward pass
- Reduce batch size (test with batch_size=1 first)
- Check buffer allocation in XCLBIN
- Verify sequence length limits
- Monitor memory usage during inference
- **Fix**: Reduce input sequence to < 1500 frames if needed

---

## 10. Documentation References

**Related Documents** (in same directory):
- `BOTH_FIXES_COMPLETE_OCT28.md` - FFT and mel filterbank accuracy fixes
- `MASTER_CHECKLIST_OCT28.md` - Current status of all components
- `DMA_OPTIMIZATION_SUMMARY.md` - Memory transfer optimizations
- `ATTENTION_ACCURACY_FINDINGS.md` - Accuracy tuning insights
- `MATMUL_INTEGRATION_PLAN.md` - Matrix multiply kernel details
- `whisper_weight_loader.py` - Weight loading implementation

**Code Files** (required for test):
- `whisper_weight_loader.py` - Load BF16 weights
- `npu_whisper_integration_example.py` - Integration example
- `test_mel_production.py` - Mel spectrogram test reference
- Kernel source files (C, MLIR) in respective subdirectories

---

## 11. Success Metrics Summary

### Test Completion Criteria
- [ ] All 3 test scenarios pass (short, medium, batch)
- [ ] Mel spectrogram accuracy > 0.99 correlation
- [ ] Encoder output accuracy > 0.95 correlation
- [ ] No NaN/Inf values in outputs
- [ ] Performance > 2x realtime for any audio length
- [ ] Memory usage within 256 MB bounds

### Deliverables
1. Test results JSON with metrics
2. Accuracy validation report
3. Performance benchmarks by scenario
4. Optimized kernel parameters (if tuning needed)
5. Integration guide for decoder phase

---

**End of End-to-End Test Plan**

*This plan assumes all NPU components are available and XCLBIN files have been compiled. Actual implementation may require adjustments based on hardware and kernel status.*
