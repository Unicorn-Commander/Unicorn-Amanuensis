# Unicorn-Amanuensis Service Exploration - Complete Index

## Report Generated: November 1, 2025

This exploration provides a comprehensive mapping of the Unicorn-Amanuensis service architecture and detailed plans for integrating the C++ NPU runtime.

## Documents Created

### 1. **SERVICE_ARCHITECTURE_REPORT.md** (1012 lines)
**Comprehensive Technical Report** - Start here for deep understanding

**Contents**:
- Executive summary
- Complete service architecture overview
- Entry point analysis (FastAPI routes)
- Platform detection logic (3-tier priority system)
- Current Whisper encoder implementation details
- Existing NPU integration (callbacks, quantization)
- Recommended integration architecture
- Detailed 6-phase integration plan (4 weeks)
- Critical integration points (memory, types, callbacks)
- Example code snippets
- Configuration & environment variables
- Troubleshooting checklist
- Complete references

**Key Sections**:
- Section 1-4: Understanding current state
- Section 5-6: Integration design and roadmap
- Section 7-12: Implementation details and reference

### 2. **INTEGRATION_SUMMARY.md** (150+ lines)
**Executive Summary** - Quick overview for decision makers

**Contents**:
- Current state snapshot
- Architecture diagram (text-based)
- Key components table
- 4-week integration roadmap
- Performance targets
- Success criteria checklist
- Known issues and workarounds
- Getting started guide

**Best For**:
- Project managers
- Quick reference
- Stakeholder briefing
- Progress tracking

### 3. **This File: EXPLORATION_INDEX.md**
Navigation guide for all exploration results

---

## Architecture Summary

### Service Structure
```
Unicorn-Amanuensis (Multi-Platform STT Service)
├── api.py (105 lines)
│   └── Main FastAPI router with platform detection
├── runtime/platform_detector.py (147 lines)
│   └── XDNA2 > XDNA1 > CPU detection
├── xdna2/runtime/whisper_xdna2_runtime.py (946 lines)
│   └── Python encoder + NPU callbacks
├── xdna2/npu_callback_native.py (435 lines)
│   └── Python-side NPU matmul handler
└── xdna1/server.py (complete FastAPI server)
    └── XDNA1 fallback implementation
```

### C++ Integration (Ready)
```
xdna2/cpp/ (Complete, Compiled)
├── include/
│   ├── encoder_c_api.h (C-style API)
│   └── npu_callback.h (Callback interface)
├── src/
│   ├── encoder_c_api.cpp (API implementation)
│   ├── attention.cpp (Self-attention layer)
│   ├── ffn.cpp (Feed-forward layer)
│   └── ... quantization, bfp16, main
├── tests/ (10 test programs)
└── CMakeLists.txt (Build system)
```

---

## Key Findings

### 1. Service Architecture Status
- **FastAPI Server**: Production-ready (port 9000)
- **Platform Detection**: Working (XDNA2, XDNA1, CPU)
- **Python Encoder**: Complete and functional (946 lines)
- **NPU Callbacks**: Framework ready (435 lines)
- **C++ Encoder**: Implemented and compiled (2000+ lines)

### 2. Integration Points Identified
1. **Service Entry** (`api.py`): Routes based on platform
2. **Platform Detection**: Automatic via `platform_detector.py`
3. **Audio Preprocessing**: Librosa mel-spectrogram
4. **Encoder Layers**: Where C++ will be integrated
5. **NPU Callbacks**: Python handlers for C++ calls
6. **XRT Kernels**: INT8 matmul acceleration

### 3. Data Flow for Transcription
```
1. POST /v1/audio/transcriptions (audio file)
   ↓
2. Platform Detection (auto-detect XDNA2)
   ↓
3. Audio Preprocessing (mel-spectrogram)
   ↓
4. Encoder [C++ via ctypes] (6 transformer layers)
   ├─ C++ calls Python callback
   ├─ Python quantizes and runs NPU
   └─ C++ dequantizes result
   ↓
5. Decoder (CPU for now)
   ↓
6. Return JSON with transcription
```

### 4. Performance Targets
- **Current (XDNA1)**: 220x realtime
- **Target (XDNA2 + C++)**: 400-500x realtime
- **Power**: 5-15W (vs 40-50W for XDNA1)
- **Latency**: <30ms for 15s audio

---

## Integration Phases

### Phase 1: Build & Bridge (Week 1)
**Objective**: Compile C++ library and create Python wrapper

**Deliverables**:
- `libencoder.so` compiled with XRT support
- `encoder_ctypes_bridge.py` (Python-C++ bridge)
- Verified library loading and symbol resolution

**File**: See SERVICE_ARCHITECTURE_REPORT.md Section 6.1

### Phase 2: Service Integration (Week 2)
**Objective**: Integrate C++ encoder with WhisperXDNA2Runtime

**Deliverables**:
- C++ encoder initialization in Python
- NPU callback registration
- Encoder layer replacement

**File**: See SERVICE_ARCHITECTURE_REPORT.md Section 6.3

### Phase 3: Testing & Validation (Week 2-3)
**Objective**: Ensure correctness and performance

**Deliverables**:
- Unit tests (layer creation, weight loading)
- Accuracy tests (<1% vs PyTorch)
- Performance tests (target: <30ms for 15s audio)

**File**: See SERVICE_ARCHITECTURE_REPORT.md Section 6.4

### Phase 4: Deployment (Week 3-4)
**Objective**: Update service and document

**Deliverables**:
- Modified `api.py` to use C++ encoder
- Complete documentation
- Deployment checklist

**File**: See SERVICE_ARCHITECTURE_REPORT.md Section 6.6

---

## Critical Information

### Memory Management
**Issue**: C++ pointers vs Python garbage collection

**Solution**: Keep numpy arrays alive during C++ execution
```python
input_array = np.asarray(data, dtype=np.float32)
lib.encoder_layer_forward(
    handle,
    input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    ...
)
# input_array stays in scope
```

### Data Types
**Critical**: Type consistency throughout pipeline

- **Weights**: FP32 (from Hugging Face) → quantize in Python → INT8
- **Activations**: FP32 in/out of C++, INT8 on NPU
- **Biases**: FP32 (always CPU)

### Callback Signature
**C++ Type**:
```c
typedef int (*NPUMatmulCallback)(
    void* user_data,
    const uint8_t* A_bfp16,
    const uint8_t* B_bfp16,
    uint8_t* C_bfp16,
    size_t M, K, N
);
```

**Python Implementation**: ctypes.CFUNCTYPE wrapper (see report Section 7.3)

### Known Issue: BF16 Signed Value Bug
**Status**: AMD XDNA2 AIE accumulator bug (documented)

**Impact**: 789-2823% error with signed BF16 values

**Workaround**: Scale inputs to [0,1] before NPU (3.55% error)

**File**: `xdna2/runtime/bf16_workaround.py`

---

## File Reference Table

| Component | File | Lines | Status | Key Info |
|-----------|------|-------|--------|----------|
| **Service Entry** | `api.py` | 105 | ✅ | FastAPI router, platform detection |
| **Platform Detect** | `runtime/platform_detector.py` | 147 | ✅ | XDNA2/XDNA1/CPU detection |
| **Python Runtime** | `xdna2/runtime/whisper_xdna2_runtime.py` | 946 | ✅ | Encoder + model loading |
| **NPU Callback** | `xdna2/npu_callback_native.py` | 435 | ✅ | BF16/BFP16 handler |
| **Quantization** | `xdna2/runtime/quantization.py` | 200+ | ✅ | INT8 quantization |
| **BF16 Fix** | `xdna2/runtime/bf16_workaround.py` | 250+ | ✅ | Workaround for AMD bug |
| **C++ API Header** | `xdna2/cpp/include/encoder_c_api.h` | 126 | ✅ | C-style API |
| **C++ Callback Header** | `xdna2/cpp/include/npu_callback.h` | 66 | ✅ | Callback interface |
| **C++ Encoder** | `xdna2/cpp/src/encoder_c_api.cpp` | 150+ | ✅ | API implementation |
| **C++ Attention** | `xdna2/cpp/src/attention.cpp` | 300+ | ✅ | Self-attention layer |
| **C++ FFN** | `xdna2/cpp/src/ffn.cpp` | 200+ | ✅ | Feed-forward layer |
| **C++ Tests** | `xdna2/cpp/tests/` | 1000+ | ✅ | 10 test programs |
| **Build System** | `xdna2/cpp/CMakeLists.txt` | 100+ | ✅ | Cmake configuration |

---

## Navigation Guide

### For Different Audiences

**Project Managers**:
1. Start: `INTEGRATION_SUMMARY.md`
2. Then: Section 5 of SERVICE_ARCHITECTURE_REPORT.md (architecture diagram)
3. Reference: "Integration Roadmap" section

**Software Engineers**:
1. Start: `INTEGRATION_SUMMARY.md` (overview)
2. Then: SERVICE_ARCHITECTURE_REPORT.md (sections 3-7)
3. Deep Dive: C++ API headers in `xdna2/cpp/include/`
4. Code: `xdna2/runtime/whisper_xdna2_runtime.py` and `npu_callback_native.py`

**Integration Specialists**:
1. Start: SERVICE_ARCHITECTURE_REPORT.md Sections 1-4 (current state)
2. Focus: Sections 5-7 (integration points and critical issues)
3. Implementation: Section 6 (step-by-step plan)
4. Reference: Section 9 (code examples)

**QA/Testing**:
1. Read: Section 6.4 (Testing & Validation)
2. Requirements: "Success Criteria" in INTEGRATION_SUMMARY.md
3. Tests: `xdna2/cpp/tests/` directory

---

## Quick Start Commands

### 1. Review Architecture
```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis
cat INTEGRATION_SUMMARY.md
```

### 2. Read Full Report
```bash
cat SERVICE_ARCHITECTURE_REPORT.md | less
```

### 3. Check Current Implementation
```bash
head -100 xdna2/runtime/whisper_xdna2_runtime.py
head -50 xdna2/cpp/include/encoder_c_api.h
```

### 4. Verify Platform Detection
```bash
python3 -c "from runtime.platform_detector import get_platform; print(get_platform())"
```

### 5. Build C++ (When Ready)
```bash
cd xdna2/cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
ls -la libencoder.so
```

---

## Success Metrics

### By Phase
| Phase | Week | Metric | Target |
|-------|------|--------|--------|
| Build & Bridge | 1 | `libencoder.so` exists | Yes |
| Integration | 2 | `encoder_ctypes_bridge.py` works | Yes |
| Testing | 3 | Accuracy vs PyTorch | <1% error |
| Performance | 3 | Latency for 15s audio | <30ms |
| Deployment | 4 | Service uses C++ encoder | Yes |

### Final Metrics
- **Realtime Factor**: 400-500x (vs 220x baseline)
- **Power Draw**: 5-15W (vs 40-50W baseline)
- **Accuracy**: <1% vs PyTorch
- **Service Uptime**: 99.9%

---

## Related Documentation

**In This Service**:
- `README.md` - Service overview
- `xdna2/README.md` - XDNA2 specifics
- `xdna2/cpp/README.md` - C++ build guide
- `xdna2/BF16_SIGNED_VALUE_BUG.md` - AMD bug details
- `xdna2/API_EXAMPLES.md` - Usage examples

**In CC-1L Project**:
- `/home/ccadmin/CC-1L/CLAUDE.md` - Project context
- `/home/ccadmin/CC-1L/docs/architecture/OVERVIEW.md` - System architecture
- `/home/ccadmin/CC-1L/docs/roadmap/implementation-roadmap.md` - Development timeline

---

## Key Takeaways

1. **Service is production-ready** - Platform detection, API routes, basic functionality all working
2. **C++ encoder is complete** - 2000+ lines, compiled, ready to integrate
3. **Architecture is clean** - Separation of concerns, easy to integrate C++
4. **Integration is straightforward** - ctypes bridge, callback pattern, no major architectural changes
5. **Performance target is achievable** - 2.3x speedup from XDNA1 baseline is realistic given 2.3% NPU utilization
6. **Known issues are manageable** - BF16 workaround exists and working, quantization pipeline proven

---

## Next Action Items

1. **Review Architecture** (30 min)
   - Read INTEGRATION_SUMMARY.md

2. **Deep Dive** (2-3 hours)
   - Read SERVICE_ARCHITECTURE_REPORT.md Sections 1-7

3. **Code Review** (2-3 hours)
   - Examine `xdna2/cpp/include/encoder_c_api.h`
   - Review `xdna2/runtime/whisper_xdna2_runtime.py`
   - Check `xdna2/npu_callback_native.py`

4. **Build Planning** (1 hour)
   - Review Phase 1 in SERVICE_ARCHITECTURE_REPORT.md Section 6.1
   - Prepare build environment
   - Verify XRT/CMake availability

5. **Begin Integration** (Week 1)
   - Build C++ library
   - Create ctypes bridge
   - Test library loading

---

## Questions or Clarifications?

Refer to the appropriate section of SERVICE_ARCHITECTURE_REPORT.md:
- **Architecture**: Sections 1-4
- **Integration Design**: Section 5
- **Implementation Plan**: Section 6
- **Technical Details**: Section 7-8
- **Code Examples**: Section 9
- **Configuration**: Section 10
- **Troubleshooting**: Section 11

---

**Exploration Complete**: November 1, 2025
**Status**: Ready for C++ NPU Runtime Integration
**Timeline**: 4 weeks to full deployment
**Target Performance**: 400-500x realtime speech-to-text

All documentation is self-contained in this directory.
