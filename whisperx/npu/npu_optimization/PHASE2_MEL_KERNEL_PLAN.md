# Phase 2: Mel Spectrogram Kernel Implementation Plan
**Date**: October 27, 2025
**Status**: Planning
**Goal**: 20-30x realtime Whisper transcription with NPU mel extraction

---

## 🎯 Objective

Replace the CPU-based librosa mel spectrogram extraction with a custom NPU kernel to achieve the first major performance improvement in the Whisper pipeline.

**Current Performance**: librosa on CPU (~0.3s for 55s audio, 5.8% of total time)
**Target Performance**: NPU mel kernel @ 20-30x realtime (~0.015s for 55s audio)
**Expected Speedup**: ~20x faster than current CPU implementation

---

## 📊 Mel Spectrogram Overview

### What It Does
Converts raw audio waveform into a 2D time-frequency representation optimized for speech:

```
Raw Audio (16kHz) → Windowing → FFT → Mel Filterbank → Log → Mel Spectrogram
[16000 samples]      [400 samp]  [256]   [80 filters]    [dB]   [80×3000]
```

### Whisper's Requirements
- **Input**: 16kHz audio, arbitrary length
- **Window**: 25ms (400 samples) with 10ms hop (160 samples)
- **FFT Size**: 512 (zero-padded from 400)
- **Mel Filters**: 80 filters (128 Hz to 8 kHz)
- **Output**: 80×N INT8 array (N = number of frames)

---

## 🚀 Phased Implementation Strategy

### Phase 2.1: Simplified Prototype (Week 1)
**Goal**: Prove NPU can process audio and produce output

**Scope**:
- Simple magnitude spectrum (skip mel filtering)
- Basic FFT implementation
- No INT8 quantization yet (use FP16)
- Process one frame at a time

**Deliverables**:
1. `mel_simple.c` - C++ kernel for basic FFT
2. `mel_simple.mlir` - MLIR wrapper
3. `mel_simple.xclbin` - Compiled NPU binary
4. Test script showing NPU execution

**Success Criteria**:
- ✅ NPU kernel compiles
- ✅ Executes without errors
- ✅ Produces magnitude spectrum output
- ⚠️ Accuracy doesn't matter yet

### Phase 2.2: Add Mel Filterbank (Week 2)
**Goal**: Generate actual mel features

**Additions**:
- 80 triangular mel filters
- Filterbank weights precomputed
- Mel-scale frequency warping
- Energy summation

**Deliverables**:
1. `mel_filterbank.c` - Mel filter application
2. Precomputed filter weights
3. Updated MLIR with larger buffers

**Success Criteria**:
- ✅ Produces 80-bin mel features
- ✅ Comparable to librosa output (within 10%)
- ✅ Still runs on NPU

### Phase 2.3: INT8 Optimization (Week 3)
**Goal**: Match Whisper's INT8 precision requirements

**Optimizations**:
- INT8 quantization of weights
- INT8 arithmetic in kernels
- Fast log approximation (lookup table)
- Dynamic range compression

**Deliverables**:
1. Quantization scheme
2. INT8 lookup tables
3. Updated kernel with INT8 ops

**Success Criteria**:
- ✅ Output matches Whisper's expectations
- ✅ WER (Word Error Rate) < 1% degradation
- ✅ 4x memory bandwidth reduction

### Phase 2.4: Multi-Frame Processing (Week 4)
**Goal**: Process entire audio files efficiently

**Optimizations**:
- Batch multiple frames
- Overlap-add for windows
- Streaming mode for live audio
- Efficient DMA transfers

**Deliverables**:
1. Batched processing kernel
2. Streaming interface
3. Integration with Whisper pipeline

**Success Criteria**:
- ✅ Processes 1-hour audio in < 5 seconds
- ✅ 20-30x realtime performance
- ✅ Ready for encoder integration

---

## 🔧 Technical Details

### FFT Implementation Options

**Option 1: Radix-2 FFT (Simplest)**
- Easiest to implement
- Good for learning
- Not optimal for AIE2

**Option 2: Radix-4 FFT (UC-Meeting-Ops approach)**
- Better performance on AIE2
- More complex butterfly operations
- Proven to work

**Option 3: Use CMSIS-DSP Library (If available)**
- Pre-optimized for ARM/AIE
- May not be available for AIE2
- Worth investigating

**Recommendation**: Start with **Radix-2** for Phase 2.1, switch to **Radix-4** in Phase 2.2

### Memory Layout

```
AIE Tile Memory (64KB):
├── Input Buffer (1KB)      - Raw audio samples
├── Window Buffer (1KB)     - Windowed samples
├── FFT Scratch (4KB)       - Complex FFT intermediate
├── Filter Weights (4KB)    - 80 mel filters
├── Output Buffer (2KB)     - Mel features
└── Code + Stack (52KB)     - Kernel code
```

### Data Flow

```
CPU → NPU Input FIFO → AIE Tile Buffer → Process → Output Buffer → Output FIFO → CPU
[Audio]  [DMA S2MM]      [1KB chunk]      [FFT]    [80 mels]    [DMA MM2S]   [Features]
```

---

## 📁 File Structure

```
npu_optimization/
├── mel_kernels/                    # New directory for mel kernel
│   ├── mel_simple.c               # Phase 2.1: Simple FFT
│   ├── mel_simple.mlir            # MLIR wrapper
│   ├── mel_filterbank.c           # Phase 2.2: Add filters
│   ├── mel_int8.c                 # Phase 2.3: INT8 version
│   ├── mel_batched.c              # Phase 2.4: Multi-frame
│   ├── fft_radix2.c               # Radix-2 FFT implementation
│   ├── fft_radix4.c               # Radix-4 FFT (later)
│   ├── mel_filters_80.h           # Precomputed filter coefficients
│   └── test_mel_kernel.py         # Test script
├── build/
│   ├── mel_simple.o               # Compiled kernel
│   ├── mel_simple.xclbin          # NPU binary
│   └── ...
└── PHASE2_MEL_KERNEL_PLAN.md      # This document
```

---

## 🎯 Success Metrics

### Performance Targets

| Metric | Current (CPU) | Phase 2.1 | Phase 2.2 | Phase 2.3 | Phase 2.4 (Target) |
|--------|---------------|-----------|-----------|-----------|-------------------|
| **Processing Time** | 0.3s | 0.5s | 0.2s | 0.1s | **0.015s** |
| **Realtime Factor** | ~180x | ~110x | ~275x | ~550x | **~3700x** |
| **Power** | 45W | 15W | 12W | 10W | **8W** |
| **Memory** | 500MB | 100MB | 50MB | 25MB | **25MB** |
| **Accuracy** | Baseline | N/A | ±10% | ±1% | **±0.1%** |

### Integration Milestones

- [  ] Phase 2.1: NPU kernel executes (proof of concept)
- [  ] Phase 2.2: Mel features generated (functional)
- [  ] Phase 2.3: INT8 precision (Whisper-compatible)
- [  ] Phase 2.4: Full pipeline integration (production-ready)

---

## 🚧 Known Challenges

### 1. FFT Complexity
**Challenge**: FFT is mathematically complex
**Solution**: Use well-documented Radix-2 algorithm first
**Resources**: Cooley-Tukey FFT, FFTW papers

### 2. Mel Filter Calculation
**Challenge**: Mel-scale frequency mapping is non-trivial
**Solution**: Precompute all filter weights in Python, store as const arrays
**Tool**: librosa.filters.mel() for reference weights

### 3. INT8 Quantization
**Challenge**: Maintaining accuracy with reduced precision
**Solution**:
- Dynamic range analysis
- Per-channel quantization
- Lookup tables for non-linear ops (log, exp)

### 4. DMA Synchronization
**Challenge**: Coordinating data movement with computation
**Solution**: Double buffering (2 input, 2 output buffers)

### 5. AIE2 Vector Operations
**Challenge**: Need to use 256-bit SIMD efficiently
**Solution**: Process 32 samples at a time (32×INT8 = 256 bits)

---

## 📚 Resources

### UC-Meeting-Ops Reference
- **File**: `mlir_aie2_kernels.mlir` lines 129-268
- **Features**: Complete mel + FFT implementation
- **Approach**: Radix-4 FFT with INT8
- **Status**: Never compiled to XCLBIN (we can be first!)

### MLIR-AIE Examples
- **Location**: `/home/ucadmin/mlir-aie-source/test/`
- **Relevant**: DMA examples, vector operations
- **API**: Check for FFT intrinsics

### FFT Algorithms
- **Radix-2 Cooley-Tukey**: Classic, simple
- **Radix-4 Butterfly**: Better for power-of-4 sizes (512 = 4^4.something... wait 512 = 2^9, not power of 4)
- **Split-Radix**: Optimal for power-of-2

### Whisper Mel Spec Reference
- **librosa**: `librosa.feature.melspectrogram()`
- **Parameters**:
  - sr=16000
  - n_fft=512
  - hop_length=160
  - win_length=400
  - n_mels=80
  - fmin=0, fmax=8000

---

## 🔄 Development Workflow

### Iteration Cycle (for each phase)

1. **Design** (1-2 days)
   - Write C++ kernel
   - Create MLIR wrapper
   - Design test cases

2. **Compile** (< 1 hour with working pipeline)
   - Peano compile C++ → ELF
   - aie-translate MLIR → CDO
   - bootgen CDO → PDI
   - xclbinutil PDI → XCLBIN

3. **Test** (1 hour)
   - Load XCLBIN on NPU
   - Execute with test data
   - Verify output

4. **Debug** (variable)
   - Check output correctness
   - Profile performance
   - Fix issues

5. **Optimize** (2-3 days)
   - Vectorize operations
   - Reduce memory transfers
   - Tune tile utilization

### Estimated Timeline

- **Phase 2.1**: 3-5 days (learning curve)
- **Phase 2.2**: 5-7 days (filter implementation)
- **Phase 2.3**: 4-6 days (quantization)
- **Phase 2.4**: 5-7 days (integration)

**Total**: **17-25 days** (3-5 weeks) to production-ready mel kernel

---

## 🎯 Phase 2.1 Next Steps (Immediate)

### Step 1: Create Simple FFT Kernel (Today)
```c
// mel_simple.c
#include <stdint.h>

// Simple 512-point FFT (radix-2, decimation-in-time)
void fft_512(int16_t* input, int32_t* output_real, int32_t* output_imag) {
    // Bit-reversal reordering
    // Butterfly operations
    // Twiddle factor multiplication
}

// Main kernel entry
int main() {
    return 0;
}
```

### Step 2: Create MLIR Wrapper (Tomorrow)
- Based on `passthrough_step3.mlir` structure
- Input: 512 samples
- Output: 256 complex values (real + imag)
- Buffer sizes calculated

### Step 3: Compile and Test (Day 3)
- Use proven compilation pipeline
- Test with synthetic sine wave
- Verify FFT output (expect peak at sine frequency)

---

## 💡 Quick Wins

1. **Use Existing Passthrough Infrastructure**
   - Copy `passthrough_step3.mlir` as template
   - Adjust buffer sizes for FFT
   - Minimal changes needed

2. **Reference Implementations**
   - FFT: CMSIS-DSP or Numerical Recipes
   - Mel Filters: librosa weights
   - INT8: ONNX Runtime quantization scheme

3. **Incremental Testing**
   - Test FFT separately
   - Test mel filters separately
   - Combine when both work

---

**Status**: Ready to begin Phase 2.1
**Next Session**: Start with simple FFT kernel implementation
**Expected Outcome**: Working NPU-based mel spectrogram in 3-5 weeks

🚀 Let's build the world's fastest Whisper transcription system!

