# Mel Spectrogram NPU Kernels - Phase 2

Custom MLIR-AIE2 kernels for accelerated mel spectrogram computation on AMD Phoenix NPU.

**Goal**: Achieve 220x realtime Whisper transcription by offloading mel extraction and transformer operations to NPU.

## Current Status: Phase 2.1 - Simple FFT Kernel

**Implementation Complete** ✅
- Basic Radix-2 FFT (512-point)
- Magnitude spectrum computation
- MLIR wrapper with ObjectFIFO data movement
- Compilation and test scripts

**Performance Target**: 20-30x realtime (Phase 2.1)

## Files

### C Kernels
- **fft_radix2.c** - Radix-2 Decimation-in-Time FFT implementation
  - 512-point FFT for Whisper (25ms window at 16kHz)
  - Cooley-Tukey algorithm
  - Bit-reversal permutation
  - Butterfly operations with twiddle factors
  - Magnitude spectrum output

- **mel_simple.c** - Main mel spectrogram kernel (Phase 2.1)
  - Hann windowing
  - Zero-padding (400 → 512 samples)
  - FFT processing
  - Frame-by-frame processing
  - Entry point for AIE core

### MLIR Configuration
- **mel_simple.mlir** - MLIR-AIE2 wrapper
  - Device: AMD Phoenix NPU (npu1)
  - Tiles: ShimNOC (0,0) + Compute (0,2)
  - ObjectFIFO data movement pattern
  - DMA configuration for host ↔ NPU transfer
  - Buffer management with locks

### Build Scripts
- **compile_mel_simple.sh** - Complete compilation pipeline
  - C++ → AIE2 ELF (Peano compiler)
  - MLIR lowering (aie-opt)
  - CDO generation (aie-translate)
  - PDI packaging (bootgen)
  - XCLBIN creation (xclbinutil)

- **test_mel_simple.py** - NPU execution test
  - Load XCLBIN to NPU
  - Generate 1kHz sine wave test audio
  - Execute FFT on NPU
  - Verify magnitude spectrum output
  - Check for peak at expected frequency bin

## Phase Roadmap

### Phase 2.1: Simple FFT (Current) ✅
**Goal**: Prove NPU can process audio

**Features**:
- Basic Radix-2 FFT
- Magnitude spectrum only
- FP32 arithmetic
- Single frame processing

**Success Criteria**:
- ✅ C kernels written
- ✅ MLIR wrapper created
- ✅ Compilation pipeline ready
- ⏳ NPU execution verified (pending test)

### Phase 2.2: Mel Filterbank (Week 2)
**Goal**: Generate actual mel features

**Additions**:
- 80 triangular mel filters
- Precomputed filter weights
- Mel-scale frequency warping
- Energy summation per band

**Target**: Output compatible with Whisper encoder

### Phase 2.3: INT8 Optimization (Week 3)
**Goal**: Match Whisper's INT8 precision

**Optimizations**:
- INT8 quantization of weights
- INT8 arithmetic in kernels
- Fast log approximation (lookup table)
- Dynamic range compression

**Target**: Match CPU accuracy, 4x bandwidth reduction

### Phase 2.4: Multi-Frame Processing (Week 4)
**Goal**: Process entire audio files efficiently

**Optimizations**:
- Batch multiple frames
- Overlap-add for windows
- Streaming mode for live audio
- Efficient DMA transfers

**Target**: 20-30x realtime, ready for integration

## Quick Start

### 1. Verify Prerequisites

```bash
# Check NPU device
ls -l /dev/accel/accel0

# Check XRT
/opt/xilinx/xrt/bin/xrt-smi examine

# Check MLIR-AIE source build
ls -l /home/ucadmin/mlir-aie-source/build/bin/aie-opt
ls -l /home/ucadmin/mlir-aie-source/ironenv/lib/python3.13/site-packages/llvm-aie/bin/clang++
```

### 2. Compile Kernel

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels
./compile_mel_simple.sh
```

**Expected Output**:
```
✅ C kernel compiled: build/mel_simple.o
✅ MLIR lowered: build/mel_simple_lowered.mlir
✅ CDO files generated
✅ PDI generated: build/mel_simple.pdi
✅ XCLBIN packaged: build/mel_simple.xclbin
```

### 3. Test on NPU

```bash
./test_mel_simple.py
```

**Expected Output**:
```
✅ NPU device opened
✅ XCLBIN registered
✅ Kernel executed successfully!
✅ Output data read: 256 magnitude values
```

## Technical Details

### FFT Algorithm: Radix-2 Cooley-Tukey

**Input**: 512 int16 samples (windowed audio)

**Process**:
1. Bit-reversal permutation
2. Log₂(512) = 9 stages of butterfly operations
3. Complex twiddle factor multiplication: W_N^k = e^(-2πik/N)
4. In-place computation for memory efficiency

**Output**: 256 complex frequency bins (0 to Nyquist)

### Whisper Mel Spectrogram Parameters

- **Sample Rate**: 16 kHz
- **Window Size**: 400 samples (25ms)
- **Hop Size**: 160 samples (10ms)
- **FFT Size**: 512 (zero-padded)
- **Mel Bins**: 80 (128 Hz to 8 kHz)
- **Output Format**: INT8 (Phase 2.3+)

### Memory Layout (AIE Tile)

```
64KB Tile Memory:
├── Input Buffer (1KB)      - Windowed audio samples
├── FFT Scratch (4KB)       - Complex intermediate values
├── Output Buffer (1KB)     - Magnitude spectrum
├── Hann Window (1KB)       - Precomputed coefficients
└── Code + Stack (57KB)     - Kernel executable
```

### Data Flow

```
CPU → ShimNOC DMA → ObjectFIFO → Compute Tile Buffer
                                       ↓
                                  FFT Kernel
                                       ↓
                    ObjectFIFO ← Compute Tile Buffer
                         ↓
           ShimNOC DMA ← CPU
```

## Performance Expectations

### Phase 2.1 (Current)
- **Processing**: Basic FFT only
- **Target**: 20-30x realtime
- **Power**: ~10W
- **Latency**: ~15ms per frame

### Phase 2.4 (Complete)
- **Processing**: Full mel extraction on NPU
- **Target**: 20-30x realtime
- **Power**: ~8W
- **Latency**: ~5ms per frame

### Full Pipeline (Phase 5)
- **With custom encoder/decoder**: 200-220x realtime
- **Proof**: UC-Meeting-Ops achieved 220x on same hardware
- **Timeline**: 10-12 weeks total

## Troubleshooting

### Compilation Errors

**"Peano not found"**:
```bash
# Verify Peano compiler location
find /home/ucadmin/mlir-aie-source -name "clang++" | grep llvm-aie
```

**"aie-opt: command not found"**:
```bash
# Use full path to source-built tools
export PATH="/home/ucadmin/mlir-aie-source/build/bin:$PATH"
```

### Runtime Errors

**"Cannot open /dev/accel/accel0"**:
```bash
# Check NPU permissions
sudo chmod a+rw /dev/accel/accel0
```

**"Output is all zeros"**:
- Check MLIR ObjectFIFO configuration
- Verify core is actually executing (not just placeholder)
- Debug with simpler passthrough first

## Next Steps

After Phase 2.1 verification:

1. **Implement Mel Filters** (Phase 2.2)
   - Create mel filter coefficient generator
   - Add 80-band triangular filterbank
   - Integrate with FFT output

2. **Add INT8 Support** (Phase 2.3)
   - Quantize filter weights
   - Implement fast log approximation
   - Match Whisper encoder input format

3. **Batch Processing** (Phase 2.4)
   - Process multiple frames in parallel
   - Add streaming mode
   - Optimize DMA for throughput

## References

- **UC-Meeting-Ops**: Achieved 220x with MLIR kernels (proof of concept)
- **MLIR-AIE Examples**: `/home/ucadmin/mlir-aie-source/test/`
- **XRT Documentation**: `/opt/xilinx/xrt/share/doc/`
- **Whisper Reference**: librosa mel spectrogram parameters

---

**Created**: October 27, 2025
**Status**: Phase 2.1 Implementation Complete
**Next**: Compile and test on NPU
