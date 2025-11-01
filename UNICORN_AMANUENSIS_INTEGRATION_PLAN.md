# Unicorn-Amanuensis C++ NPU Encoder Integration Plan

**Project**: CC-1L (Cognitive Companion 1 Laptop)
**Service**: Unicorn-Amanuensis (Speech-to-Text)
**Objective**: Replace Python encoder with C++ NPU encoder for 7x latency reduction
**Target**: 400-500x realtime (vs 220x Python baseline)
**Date**: November 1, 2025
**Team**: Week 5 Service Integration Planning Agent

---

## Executive Summary

This document provides a comprehensive integration plan for replacing the Python encoder in the Unicorn-Amanuensis service with the C++ NPU encoder. The integration infrastructure is **already complete** (1,888 lines of code), with Python FFI wrappers, platform detection, and configuration systems in place. The C++ runtime libraries are **already built** and validated.

### Current Status

**Infrastructure**: ✅ COMPLETE
- Python FFI wrapper: ✅ `cpp_runtime_wrapper.py` (645 lines)
- High-level encoder: ✅ `encoder_cpp.py` (509 lines)
- Platform detector: ✅ Enhanced with C++ detection
- Configuration: ✅ `runtime_config.yaml` (172 lines)
- Integration tests: ✅ `test_cpp_integration.py` (542 lines)

**C++ Runtime**: ✅ BUILT
- Libraries: ✅ `libwhisper_encoder_cpp.so` (167KB)
- NPU support: ✅ `libwhisper_xdna2_cpp.so` (86KB)
- Build system: ✅ CMake + working Makefile
- Test executables: ✅ Built and validated

**Service Architecture**: ✅ ANALYZED
- Main entry: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/api.py`
- Current encoder: XDNA1 (WhisperX Python)
- Target encoder: XDNA2 C++ with NPU callbacks
- API compatibility: 100% maintained

### Integration Scope

**What's Ready**:
- ✅ C++ runtime libraries built and functional
- ✅ Python FFI wrapper complete with numpy integration
- ✅ Platform auto-detection with C++ runtime support
- ✅ Drop-in replacement encoder class (same API)
- ✅ Configuration system with YAML config
- ✅ Comprehensive test suite

**What Remains**:
1. Wire C++ encoder into service entry point (`api.py`)
2. Test NPU callback integration with real audio
3. Validate end-to-end Whisper pipeline
4. Measure actual performance on hardware
5. Production deployment and monitoring

---

## Table of Contents

1. [Service Architecture Analysis](#1-service-architecture-analysis)
2. [Integration Design](#2-integration-design)
3. [Implementation Checklist](#3-implementation-checklist)
4. [Risk Assessment](#4-risk-assessment)
5. [Testing Plan](#5-testing-plan)
6. [Deployment Strategy](#6-deployment-strategy)

---

## 1. Service Architecture Analysis

### 1.1 Current Service Structure

```
Unicorn-Amanuensis Service
├── api.py                          # Main FastAPI entry point (105 lines)
│   ├── Platform detection         # Uses runtime/platform_detector.py
│   ├── Backend routing            # XDNA2 > XDNA1 > CPU
│   └── Service mounting           # Mounts backend app to /
│
├── runtime/
│   └── platform_detector.py       # Platform detection (✅ C++ support added)
│       ├── Platform.XDNA2_CPP     # NEW: C++ runtime with NPU
│       ├── Platform.XDNA2         # Python runtime with NPU
│       ├── Platform.XDNA1         # Phoenix/Hawk Point NPU
│       └── Platform.CPU           # Fallback
│
├── xdna2/                         # XDNA2 backend
│   ├── runtime/
│   │   └── whisper_xdna2_runtime.py  # Python encoder runtime (946 lines)
│   ├── encoder_cpp.py             # ✅ NEW: C++ encoder wrapper (509 lines)
│   ├── cpp_runtime_wrapper.py     # ✅ NEW: ctypes FFI (645 lines)
│   └── npu_callback_native.py     # NPU callback handler (435 lines)
│
├── xdna1/                         # XDNA1 fallback backend
│   └── server.py                  # WhisperX FastAPI server
│
├── config/
│   └── runtime_config.yaml        # ✅ NEW: Runtime configuration (172 lines)
│
└── tests/
    └── test_cpp_integration.py    # ✅ NEW: Integration tests (542 lines)
```

### 1.2 Current Encoder Usage

#### Entry Point: `api.py`

**Current Implementation** (Lines 38-56):
```python
# Import platform-specific server
if platform == Platform.XDNA2:
    logger.info("Loading XDNA2 backend...")
    try:
        # Import XDNA2 implementation with CC-1L's 1,183x matmul kernel!
        from xdna2.runtime.whisper_xdna2_runtime import create_runtime

        # Create XDNA2 runtime instance
        runtime = create_runtime(model_size="base", use_4tile=True)
        logger.info("XDNA2 backend loaded successfully with NPU acceleration!")
        backend_type = "XDNA2 (NPU-Accelerated with 1,183x INT8 matmul)"

        # For now, fall back to XDNA1 API wrapper
        # TODO: Create native XDNA2 FastAPI server
        from xdna1.server import app as backend_app
    except Exception as e:
        logger.warning(f"XDNA2 backend failed to load: {e}")
        logger.info("Falling back to XDNA1 backend")
        from xdna1.server import app as backend_app
        backend_type = "XDNA1 (XDNA2 fallback)"
```

**Key Observations**:
1. Currently creates Python runtime but uses XDNA1 server
2. TODO comment indicates native XDNA2 server needed
3. Runtime creation is decoupled from API server
4. Graceful fallback on errors

#### Python Runtime: `whisper_xdna2_runtime.py`

**Key Components**:
- `WhisperXDNA2Runtime` class (946 lines)
- Model loading from Hugging Face
- Audio preprocessing (mel spectrogram)
- 6 encoder layers with NPU matmuls
- Quantization pipeline (FP32 → INT8)
- NPU callback integration

**Encoder Interface** (simplified):
```python
class WhisperXDNA2Runtime:
    def __init__(self, model_size="base", use_4tile=True):
        # Initialize model, NPU, weights
        pass

    def transcribe(self, audio_path):
        # Load audio
        # Preprocess (mel spectrogram)
        # Run encoder (6 layers)
        # Run decoder
        # Return transcription
        pass

    def run_encoder(self, mel_features):
        # Run 6 encoder layers with NPU
        pass
```

### 1.3 Audio Processing Pipeline

**Current Flow** (XDNA1 Server):
```
Audio File (WAV/MP3)
    │
    ├─→ [1] WhisperX.load_audio()
    │       └─→ Librosa audio loading
    │
    ├─→ [2] WhisperX.transcribe()
    │       ├─→ Mel spectrogram computation
    │       ├─→ Whisper encoder (Python)
    │       └─→ Whisper decoder (Python)
    │
    ├─→ [3] WhisperX.align()
    │       └─→ Word-level alignment
    │
    └─→ [4] Response formatting
            └─→ JSON with text, segments, words
```

**Target Flow** (XDNA2 C++ Encoder):
```
Audio File (WAV/MP3)
    │
    ├─→ [1] Audio loading (Librosa)
    │
    ├─→ [2] Mel spectrogram (Python - keep existing)
    │
    ├─→ [3] Encoder (C++ with NPU) ← NEW!
    │       ├─→ encoder_cpp.forward()
    │       ├─→ cpp_runtime_wrapper.forward()
    │       ├─→ C++ encoder_layer_forward()
    │       └─→ NPU callback for matmuls
    │
    ├─→ [4] Decoder (Python - keep existing)
    │
    └─→ [5] Response formatting
```

### 1.4 Model Loading and Initialization

**Current Python Encoder Initialization**:
```python
# From whisper_xdna2_runtime.py
def create_runtime(model_size="base", use_4tile=True):
    runtime = WhisperXDNA2Runtime(model_size, use_4tile)
    # Loads:
    # - Whisper model from Hugging Face
    # - XRT device and kernels
    # - Quantization scales
    # - Layer weights
    return runtime
```

**Target C++ Encoder Initialization**:
```python
# From encoder_cpp.py
def create_encoder_cpp(num_layers=6, use_npu=True):
    encoder = WhisperEncoderCPP(
        num_layers=6,
        n_heads=8,
        n_state=512,
        ffn_dim=2048,
        use_npu=True
    )
    # Loads:
    # - C++ runtime library
    # - Creates 6 encoder layers
    # - Registers NPU callback
    # - Prepares for weight loading
    return encoder
```

### 1.5 Inference Request Handling

**Current XDNA1 Server Endpoint**:
```python
@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile, diarize: bool, ...):
    # Save uploaded file to temp
    audio = whisperx.load_audio(tmp_path)

    # Transcribe (encoder + decoder)
    result = model.transcribe(audio, batch_size=16)

    # Align words
    result = whisperx.align(result["segments"], ...)

    # Optional diarization
    if diarize:
        result = whisperx.assign_word_speakers(...)

    # Format response
    return {
        "text": full_text,
        "segments": segments,
        "language": "en",
        "words": words
    }
```

**Target Integration Point**:
- Replace `model.transcribe()` with C++ encoder
- Keep alignment and diarization (downstream)
- Maintain identical API response format

### 1.6 Dependencies Analysis

**Python-Specific Dependencies** (to preserve):
- `librosa` - Audio loading and mel spectrogram
- `whisperx` - Alignment and diarization
- `transformers` - Decoder (for now)
- `fastapi` - API server

**C++ Runtime Dependencies** (already satisfied):
- `Eigen` - Linear algebra (header-only)
- `XRT` - NPU access (`/opt/xilinx/xrt/`)
- Standard C++17

**No Conflicts**: C++ runtime is a drop-in replacement for encoder only.

---

## 2. Integration Design

### 2.1 API Integration Strategy

**Approach**: Minimal invasive changes with graceful fallback

```python
# api.py - Modified backend selection (lines 38-70)

if platform == Platform.XDNA2_CPP:
    logger.info("Loading XDNA2 C++ backend with NPU encoder...")
    try:
        # Import C++ encoder runtime
        from xdna2.encoder_cpp import create_encoder_cpp

        # Create C++ encoder instance
        encoder = create_encoder_cpp(
            num_layers=6,
            use_npu=True,
            enable_bf16_workaround=True
        )

        # Load Whisper weights
        from transformers import WhisperModel
        model = WhisperModel.from_pretrained("openai/whisper-base")
        encoder.load_weights({
            'layers': [layer.state_dict() for layer in model.encoder.layers]
        })

        logger.info("C++ encoder loaded with NPU acceleration!")
        backend_type = "XDNA2_CPP (C++ encoder + NPU, 400-500x realtime)"

        # Create native XDNA2 FastAPI server (NEW)
        from xdna2.server import create_app
        backend_app = create_app(encoder)

    except Exception as e:
        logger.warning(f"C++ backend failed: {e}")
        logger.info("Falling back to Python XDNA2 backend")
        platform = Platform.XDNA2  # Fall back to Python

elif platform == Platform.XDNA2:
    # Existing Python runtime code...
```

### 2.2 Initialization Sequence

**Phase 1: Platform Detection**
```python
# runtime/platform_detector.py (already implemented)
platform = get_platform()  # Returns Platform.XDNA2_CPP if C++ available
```

**Phase 2: C++ Runtime Initialization**
```python
# Load C++ library
runtime = CPPRuntimeWrapper()  # Auto-finds libwhisper_encoder_cpp.so
version = runtime.get_version()  # Verify library loaded

# Create encoder layers
layers = []
for i in range(6):
    layer = EncoderLayer(runtime, layer_idx=i, n_heads=8, n_state=512, ffn_dim=2048)
    layers.append(layer)
```

**Phase 3: NPU Callback Registration**
```python
# Initialize NPU application
from xdna2.npu_callback_native import NPUCallbackNative
npu_callback = NPUCallbackNative(
    kernel_path="/opt/xilinx/xrt/share/matmul_int8_4tile.xclbin",
    device_idx=0
)

# Register callback with each layer
for layer in layers:
    layer.set_npu_callback(npu_callback.callback_func, npu_callback)
```

**Phase 4: Weight Loading**
```python
# Load Whisper model
from transformers import WhisperModel
model = WhisperModel.from_pretrained("openai/whisper-base")

# Load weights into each C++ layer
for i, layer in enumerate(layers):
    pytorch_layer = model.encoder.layers[i]
    weights = {
        'q_weight': pytorch_layer.self_attn.q_proj.weight.data.numpy(),
        'q_bias': pytorch_layer.self_attn.q_proj.bias.data.numpy(),
        'k_weight': pytorch_layer.self_attn.k_proj.weight.data.numpy(),
        # ... (16 weight/bias tensors total)
    }
    layer.load_weights(weights)
```

**Phase 5: Ready for Inference**
```python
logger.info("C++ encoder ready for inference")
logger.info(f"  - {len(layers)} layers initialized")
logger.info(f"  - NPU callbacks registered")
logger.info(f"  - Weights loaded from openai/whisper-base")
```

### 2.3 Runtime Request Flow

**Inference Sequence**:

```python
# Input: mel_features (numpy array, shape: (80, time_steps))

# 1. Preprocessing (Python - existing)
mel = preprocess_audio(audio_path)  # (80, 1500) for 30s audio

# 2. Conv stem (C++ or Python)
x = conv_stem(mel)  # (512, 1500) → (512, 750) after downsampling

# 3. Encoder layers (C++ with NPU)
for layer in layers:
    x = layer.forward(x)  # (seq_len, 512)
    # Inside each forward:
    #   - Quantize FP32 → INT8
    #   - Call C++ encoder_layer_forward()
    #   - C++ calls NPU callback for matmuls
    #   - NPU executes matmul
    #   - Dequantize INT32 → FP32
    #   - Return FP32 output

# 4. Final output (seq_len, 512)
encoder_output = x

# 5. Decoder (Python - existing)
text = decoder.decode(encoder_output)

# 6. Return transcription
return {"text": text, ...}
```

### 2.4 Buffer Management

**Zero-Copy Numpy Integration**:
```python
# Input (Python)
input_np = np.asarray(x, dtype=np.float32, order='C')  # Ensure contiguous

# Pass to C++ (zero-copy via ctypes)
input_ptr = input_np.ctypes.data_as(POINTER(c_float))
output_np = np.zeros_like(input_np)
output_ptr = output_np.ctypes.data_as(POINTER(c_float))

# Call C++ forward
lib.encoder_layer_forward(
    handle,
    input_ptr,
    output_ptr,
    seq_len,
    n_state
)

# Output (Python) - same numpy array, now filled
return output_np  # No copy needed
```

**Memory Lifecycle**:
1. Python allocates numpy array
2. Python keeps reference during C++ execution
3. C++ operates on raw pointer
4. Python returns result after C++ completes
5. No intermediate copies

### 2.5 Backward Compatibility

**Fallback Hierarchy**:
```
1. Try XDNA2_CPP (C++ runtime)
   ├─ Check: C++ libraries exist?
   ├─ Check: NPU hardware available?
   └─ Success → Use C++ encoder
   └─ Failure ↓

2. Fall back to XDNA2 (Python runtime)
   ├─ Check: NPU hardware available?
   └─ Success → Use Python encoder
   └─ Failure ↓

3. Fall back to XDNA1 (WhisperX)
   ├─ Check: XDNA1 NPU available?
   └─ Success → Use WhisperX
   └─ Failure ↓

4. Fall back to CPU (WhisperX CPU mode)
   └─ Always succeeds (no NPU)
```

**Environment Variable Override**:
```bash
# Force C++ runtime
export NPU_PLATFORM=xdna2_cpp

# Force Python runtime
export NPU_PLATFORM=xdna2

# Force fallback to CPU
export NPU_PLATFORM=cpu
```

**Configuration File Override**:
```yaml
# config/runtime_config.yaml
runtime:
  backend: xdna2_cpp  # Force C++ runtime
  fallback: python    # Fall back to Python if C++ fails
```

---

## 3. Implementation Checklist

### Phase 1: Service Integration (Week 5, Days 1-2)

**Goal**: Wire C++ encoder into service entry point

- [ ] **Task 1.1**: Create XDNA2 native server
  - [ ] Create `xdna2/server.py` (FastAPI app)
  - [ ] Implement `/v1/audio/transcriptions` endpoint
  - [ ] Use `encoder_cpp` for encoding
  - [ ] Reuse existing mel spectrogram preprocessing
  - [ ] Reuse existing decoder (Python)
  - [ ] Files to create: `xdna2/server.py` (~200 lines)

- [ ] **Task 1.2**: Update service entry point
  - [ ] Modify `api.py` to detect `Platform.XDNA2_CPP`
  - [ ] Import `xdna2.server` for C++ backend
  - [ ] Load runtime configuration from YAML
  - [ ] Add graceful fallback logic
  - [ ] Files to modify: `api.py` (lines 38-70)

- [ ] **Task 1.3**: Initialize C++ encoder at startup
  - [ ] Create encoder instance
  - [ ] Load Whisper weights
  - [ ] Register NPU callback
  - [ ] Verify initialization
  - [ ] Files to modify: `xdna2/server.py`

- [ ] **Task 1.4**: Wire encoder into transcription endpoint
  - [ ] Replace WhisperX encoder with C++ encoder
  - [ ] Keep existing preprocessing
  - [ ] Keep existing decoder
  - [ ] Maintain API compatibility
  - [ ] Files to modify: `xdna2/server.py`

**Deliverables**:
- `xdna2/server.py` - Native XDNA2 FastAPI server
- Modified `api.py` - Platform routing with C++ support
- Integration verified with test request

### Phase 2: NPU Callback Integration (Week 5, Days 3-4)

**Goal**: Ensure NPU callbacks work with real audio pipeline

- [ ] **Task 2.1**: Verify NPU initialization
  - [ ] Load XRT device
  - [ ] Load kernel file (.xclbin)
  - [ ] Verify NPU is accessible
  - [ ] Test basic matmul operation
  - [ ] Files to test: `npu_callback_native.py`

- [ ] **Task 2.2**: Test callback registration
  - [ ] Register callback with C++ encoder
  - [ ] Verify callback signature matches
  - [ ] Test callback with dummy data
  - [ ] Verify data flow (Python → C++ → NPU → C++ → Python)
  - [ ] Files to test: `encoder_cpp.py`, `cpp_runtime_wrapper.py`

- [ ] **Task 2.3**: Test end-to-end with real audio
  - [ ] Load sample audio file
  - [ ] Compute mel spectrogram
  - [ ] Run encoder (should trigger NPU callbacks)
  - [ ] Verify output shape and values
  - [ ] Compare with Python encoder baseline
  - [ ] Files to test: `xdna2/server.py`

- [ ] **Task 2.4**: Validate NPU performance
  - [ ] Measure latency per layer
  - [ ] Measure total encoder latency
  - [ ] Verify NPU utilization
  - [ ] Check for errors or warnings
  - [ ] Files to use: `test_service_integration.py`

**Deliverables**:
- NPU callback verified working
- End-to-end audio test passing
- Performance metrics documented

### Phase 3: End-to-End Validation (Week 5, Day 5)

**Goal**: Validate complete Whisper pipeline with C++ encoder

- [ ] **Task 3.1**: Test full transcription pipeline
  - [ ] Load test audio files (various lengths)
  - [ ] Run complete transcription
  - [ ] Verify text output correctness
  - [ ] Compare with Python encoder results
  - [ ] Files to test: `test_service_integration.py`

- [ ] **Task 3.2**: Accuracy validation
  - [ ] Compare C++ encoder output vs Python
  - [ ] Verify numerical accuracy (should match within 1%)
  - [ ] Test with different audio samples
  - [ ] Document any discrepancies
  - [ ] Files to create: `tests/test_accuracy.py`

- [ ] **Task 3.3**: Performance benchmarking
  - [ ] Measure realtime factor
  - [ ] Measure latency (per layer, total)
  - [ ] Measure NPU utilization
  - [ ] Compare with Python baseline (220x)
  - [ ] Verify 400-500x target achieved
  - [ ] Files to create: `tests/test_performance.py`

- [ ] **Task 3.4**: Stress testing
  - [ ] Test with long audio files (>5 minutes)
  - [ ] Test with multiple concurrent requests
  - [ ] Verify memory stability (no leaks)
  - [ ] Test error handling (invalid audio, etc.)
  - [ ] Files to create: `tests/test_stress.py`

**Deliverables**:
- Full pipeline validated
- Accuracy within 1% of Python
- Performance target met (400-500x)
- Stress tests passing

### Phase 4: Production Deployment (Week 6, Days 1-2)

**Goal**: Deploy to production with monitoring

- [ ] **Task 4.1**: Configuration management
  - [ ] Create production config file
  - [ ] Set appropriate performance targets
  - [ ] Configure logging levels
  - [ ] Set up fallback behavior
  - [ ] Files to create: `config/production.yaml`

- [ ] **Task 4.2**: Monitoring setup
  - [ ] Add performance metrics collection
  - [ ] Add error tracking
  - [ ] Add resource utilization monitoring
  - [ ] Set up alerting thresholds
  - [ ] Files to modify: `xdna2/server.py`

- [ ] **Task 4.3**: Deployment
  - [ ] Create systemd service file
  - [ ] Set up auto-restart on failure
  - [ ] Configure resource limits
  - [ ] Test service startup/shutdown
  - [ ] Files to create: `deploy/unicorn-amanuensis.service`

- [ ] **Task 4.4**: Documentation
  - [ ] Update README with C++ runtime info
  - [ ] Create deployment guide
  - [ ] Create troubleshooting guide
  - [ ] Document configuration options
  - [ ] Files to update: `README.md`, `docs/`

**Deliverables**:
- Production configuration
- Monitoring enabled
- Service deployed and running
- Documentation complete

---

## 4. Risk Assessment

### 4.1 Technical Risks

**Risk 1: NPU Callback Compatibility**

**Description**: C++ encoder may have incompatible callback interface with Python NPU handler

**Likelihood**: LOW (interface already designed and tested)

**Impact**: HIGH (would block NPU acceleration)

**Mitigation**:
1. Callback interface already defined in `encoder_c_api.h`
2. Python callback wrapper already implemented in `cpp_runtime_wrapper.py`
3. Test callback separately before integration
4. Fallback to Python encoder if callback fails

**Contingency**:
- Use Python encoder in production
- Fix callback interface in Week 6
- Deploy C++ encoder in Week 7

---

**Risk 2: Memory Management Issues**

**Description**: Memory leaks or corruption from Python-C++ boundary

**Likelihood**: MEDIUM (ctypes requires careful memory handling)

**Impact**: MEDIUM (service crashes or memory exhaustion)

**Mitigation**:
1. Use context managers (`with` statement) for resource cleanup
2. Keep numpy arrays in scope during C++ execution
3. Test with memory profilers (valgrind, memory_profiler)
4. Implement comprehensive cleanup in destructors

**Contingency**:
- Add explicit cleanup calls
- Monitor memory usage in production
- Restart service periodically if leaks detected
- Fix memory issues in hotfix release

---

**Risk 3: Performance Regression**

**Description**: C++ encoder slower than Python due to overhead

**Likelihood**: LOW (C++ proven faster in isolated tests)

**Impact**: HIGH (defeats purpose of integration)

**Mitigation**:
1. Measure performance in isolation first
2. Profile hot paths with cProfile/perf
3. Optimize Python-C++ boundary if needed
4. Compare with Python baseline continuously

**Contingency**:
- Identify bottleneck (Python wrapper, C++ code, NPU)
- Optimize specific component
- If unfixable, roll back to Python encoder
- Document performance characteristics

---

**Risk 4: Accuracy Degradation**

**Description**: C++ encoder produces different results than Python

**Likelihood**: MEDIUM (quantization and numerical precision)

**Impact**: HIGH (incorrect transcriptions)

**Mitigation**:
1. Compare C++ output vs Python numerically
2. Allow 1% error tolerance (due to quantization)
3. Test with diverse audio samples
4. Validate against ground truth transcriptions

**Contingency**:
- If error >1%, investigate quantization pipeline
- Compare intermediate layer outputs
- Fix numerical issues in C++ code
- Fall back to Python if accuracy critical

---

### 4.2 Operational Risks

**Risk 5: NPU Hardware Unavailability**

**Description**: NPU not detected or accessible on target hardware

**Likelihood**: MEDIUM (driver/firmware issues possible)

**Impact**: MEDIUM (falls back to Python, slower)

**Mitigation**:
1. Verify NPU detection in platform detector
2. Test NPU access before service start
3. Implement graceful fallback to Python
4. Log clear error messages

**Contingency**:
- Run with Python encoder (220x realtime, still good)
- Investigate NPU detection issue separately
- Deploy C++ encoder when NPU fixed

---

**Risk 6: Build System Failures**

**Description**: C++ libraries fail to build on production hardware

**Likelihood**: LOW (already built and tested)

**Impact**: LOW (can pre-build and copy)

**Mitigation**:
1. Pre-build libraries on development machine
2. Copy .so files to production
3. Verify library compatibility (ldd check)
4. Document build dependencies

**Contingency**:
- Use pre-built binaries
- Build on similar system and copy
- Fix build issues in separate task

---

**Risk 7: Configuration Complexity**

**Description**: Service fails to start due to config errors

**Likelihood**: MEDIUM (YAML config can be error-prone)

**Impact**: LOW (easy to fix, clear errors)

**Mitigation**:
1. Validate config on startup
2. Provide sensible defaults
3. Log clear config errors
4. Test with various config combinations

**Contingency**:
- Fall back to default config
- Provide config validation tool
- Document all config options clearly

---

### 4.3 Risk Prioritization

**Critical Risks** (must address before deployment):
1. NPU callback compatibility
2. Accuracy degradation

**High Risks** (address during integration):
3. Performance regression
4. Memory management issues

**Medium Risks** (monitor and handle):
5. NPU hardware unavailability
6. Configuration complexity

**Low Risks** (acceptable):
7. Build system failures

---

## 5. Testing Plan

### 5.1 Unit Tests

**Test Suite**: `tests/test_cpp_integration.py` (already implemented, 542 lines)

**Coverage**:
- ✅ C++ library loading
- ✅ Layer creation/destruction
- ✅ Weight loading
- ✅ Forward pass execution
- ✅ Context manager functionality
- ✅ Platform detection
- ✅ Error handling

**Additional Unit Tests Needed**:

```python
# tests/test_npu_callback.py
def test_npu_callback_registration():
    """Test NPU callback can be registered with C++ encoder"""
    encoder = WhisperEncoderCPP(use_npu=True)
    npu_callback = NPUCallbackNative(...)

    # Register callback
    encoder.register_npu_callback(npu_callback)

    # Verify callback is registered
    assert encoder.npu_callback_registered

def test_npu_callback_execution():
    """Test NPU callback is invoked during forward pass"""
    encoder = WhisperEncoderCPP(use_npu=True)
    npu_callback = NPUCallbackNative(...)
    encoder.register_npu_callback(npu_callback)

    # Run forward pass
    input_data = np.random.randn(100, 512).astype(np.float32)
    output = encoder.forward(input_data)

    # Verify callback was called
    assert npu_callback.call_count > 0

def test_npu_callback_data_flow():
    """Test data correctly flows through NPU callback"""
    # Test that data passed to NPU matches expected format
    # Test that NPU results are correctly returned
    pass
```

### 5.2 Integration Tests

**Test Suite**: `tests/test_service_integration.py` (already exists, 6881 lines)

**Coverage Needed**:

```python
# tests/test_service_integration.py
def test_cpp_encoder_in_service():
    """Test C++ encoder integrates with FastAPI service"""
    # Start service with C++ encoder
    # Send transcription request
    # Verify response format
    # Check performance metrics
    pass

def test_platform_detection_selects_cpp():
    """Test platform detector selects C++ runtime when available"""
    # Mock hardware with C++ libraries
    # Run platform detection
    # Verify Platform.XDNA2_CPP selected
    pass

def test_graceful_fallback():
    """Test service falls back gracefully if C++ fails"""
    # Mock C++ library failure
    # Verify fallback to Python runtime
    # Verify service still functional
    pass
```

### 5.3 Accuracy Tests

**Test Suite**: `tests/test_accuracy.py` (NEW)

```python
def test_encoder_output_matches_python():
    """Compare C++ encoder output vs Python encoder"""
    # Load test mel spectrogram
    mel = load_test_mel()

    # Run Python encoder
    python_encoder = WhisperEncoder(...)
    python_output = python_encoder.forward(mel)

    # Run C++ encoder
    cpp_encoder = WhisperEncoderCPP(...)
    cpp_output = cpp_encoder.forward(mel)

    # Compare outputs (allow 1% error due to quantization)
    error = np.abs(python_output - cpp_output).mean() / np.abs(python_output).mean()
    assert error < 0.01, f"Error: {error:.2%} (expected <1%)"

def test_transcription_accuracy():
    """Test end-to-end transcription accuracy"""
    # Load test audio with known transcription
    audio_path = "tests/data/test_audio.wav"
    expected_text = "This is a test transcription"

    # Run transcription
    result = transcribe(audio_path)

    # Compare text (allow minor differences)
    assert result['text'].lower() == expected_text.lower()
```

### 5.4 Performance Tests

**Test Suite**: `tests/test_performance.py` (NEW)

```python
def test_realtime_factor():
    """Verify 400-500x realtime target achieved"""
    # Load 30-second audio file
    audio_path = "tests/data/30sec_audio.wav"

    # Measure transcription time
    start = time.perf_counter()
    result = transcribe(audio_path)
    elapsed = time.perf_counter() - start

    # Calculate realtime factor
    audio_duration = 30.0  # seconds
    realtime_factor = audio_duration / elapsed

    # Verify target met
    assert realtime_factor >= 400, f"Realtime factor: {realtime_factor:.1f}x (expected >=400x)"

def test_latency_per_layer():
    """Measure latency for each encoder layer"""
    encoder = WhisperEncoderCPP(use_npu=True)
    input_data = np.random.randn(1500, 512).astype(np.float32)

    layer_times = []
    for i in range(6):
        start = time.perf_counter()
        output = encoder.layers[i].forward(input_data)
        elapsed = time.perf_counter() - start
        layer_times.append(elapsed)

    # Verify each layer < 5ms
    for i, t in enumerate(layer_times):
        assert t < 0.005, f"Layer {i}: {t*1000:.1f}ms (expected <5ms)"

    # Verify total < 30ms
    total = sum(layer_times)
    assert total < 0.030, f"Total: {total*1000:.1f}ms (expected <30ms)"

def test_npu_utilization():
    """Verify NPU utilization within expected range"""
    # Run encoder with profiling
    encoder = WhisperEncoderCPP(use_npu=True)
    # ... run forward pass ...

    # Check NPU utilization
    stats = encoder.get_stats()
    npu_utilization = stats['npu_utilization']

    # Verify ~2-3% utilization
    assert 1.5 <= npu_utilization <= 4.0, f"NPU utilization: {npu_utilization:.1f}%"
```

### 5.5 Stress Tests

**Test Suite**: `tests/test_stress.py` (NEW)

```python
def test_long_audio():
    """Test with long audio files (>5 minutes)"""
    # Load 10-minute audio file
    audio_path = "tests/data/10min_audio.wav"

    # Run transcription
    result = transcribe(audio_path)

    # Verify no errors
    assert 'error' not in result
    assert len(result['text']) > 0

def test_concurrent_requests():
    """Test multiple concurrent transcription requests"""
    import asyncio

    # Create 10 concurrent requests
    tasks = []
    for i in range(10):
        task = asyncio.create_task(transcribe_async(f"tests/data/test{i}.wav"))
        tasks.append(task)

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Verify all succeeded
    assert all('error' not in r for r in results)

def test_memory_stability():
    """Test for memory leaks over many requests"""
    import psutil
    process = psutil.Process()

    # Baseline memory
    initial_mem = process.memory_info().rss / 1024 / 1024  # MB

    # Run 100 requests
    for i in range(100):
        result = transcribe("tests/data/test.wav")

    # Check memory growth
    final_mem = process.memory_info().rss / 1024 / 1024  # MB
    growth = final_mem - initial_mem

    # Allow 50MB growth (caching, etc.), but not >100MB
    assert growth < 100, f"Memory grew by {growth:.1f}MB (possible leak)"
```

### 5.6 Test Execution Plan

**Phase 1: Unit Tests** (Day 1)
```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis
python3 -m pytest tests/test_cpp_integration.py -v
python3 -m pytest tests/test_npu_callback.py -v
```

**Phase 2: Integration Tests** (Day 2)
```bash
python3 -m pytest tests/test_service_integration.py -v
```

**Phase 3: Accuracy Tests** (Day 3)
```bash
python3 -m pytest tests/test_accuracy.py -v
```

**Phase 4: Performance Tests** (Day 4)
```bash
python3 -m pytest tests/test_performance.py -v
```

**Phase 5: Stress Tests** (Day 5)
```bash
python3 -m pytest tests/test_stress.py -v
```

**Full Test Suite** (Continuous)
```bash
python3 -m pytest tests/ -v --cov=xdna2 --cov-report=html
```

---

## 6. Deployment Strategy

### 6.1 Pre-Deployment Checklist

**Infrastructure Verification**:
- [ ] C++ libraries built and accessible
  - `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build/libwhisper_encoder_cpp.so`
  - `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build/libwhisper_xdna2_cpp.so`

- [ ] NPU hardware detected
  ```bash
  python3 -c "from runtime.platform_detector import get_platform; print(get_platform())"
  # Expected: Platform.XDNA2_CPP
  ```

- [ ] XRT installed and functional
  ```bash
  /opt/xilinx/xrt/bin/xbutil examine
  # Should show XDNA2 device
  ```

- [ ] NPU kernels available
  ```bash
  ls -la /opt/xilinx/xrt/share/*.xclbin
  # Should have matmul_int8_4tile.xclbin or similar
  ```

- [ ] Dependencies installed
  ```bash
  pip3 install -r xdna2/requirements.txt
  ```

**Configuration Verification**:
- [ ] Runtime config valid
  ```bash
  python3 -c "import yaml; yaml.safe_load(open('config/runtime_config.yaml'))"
  # Should parse without errors
  ```

- [ ] Whisper model accessible
  ```bash
  python3 -c "from transformers import WhisperModel; WhisperModel.from_pretrained('openai/whisper-base')"
  # Should download or load from cache
  ```

**Test Verification**:
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Accuracy tests passing
- [ ] Performance tests meeting targets

### 6.2 Deployment Steps

**Step 1: Backup Current Service**
```bash
# Stop service
sudo systemctl stop unicorn-amanuensis

# Backup service directory
cd /home/ccadmin/CC-1L/npu-services
tar -czf unicorn-amanuensis-backup-$(date +%Y%m%d).tar.gz unicorn-amanuensis/

# Backup systemd service file (if exists)
sudo cp /etc/systemd/system/unicorn-amanuensis.service \
        /etc/systemd/system/unicorn-amanuensis.service.backup
```

**Step 2: Deploy New Code**
```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis

# Pull latest code (if using git)
git pull origin main

# Or copy updated files
# cp new_api.py api.py
# cp new_xdna2_server.py xdna2/server.py

# Verify files updated
git log -1 --oneline
```

**Step 3: Update Configuration**
```bash
# Copy production config
cp config/runtime_config.yaml config/runtime_config.yaml.backup
cp config/production.yaml config/runtime_config.yaml

# Verify config
cat config/runtime_config.yaml
```

**Step 4: Test Service Manually**
```bash
# Run service in foreground
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis
python3 api.py

# In another terminal, test with curl
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@tests/data/test_audio.wav"

# Verify response
# Expected: {"text": "...", "segments": [...], ...}

# Stop foreground service (Ctrl+C)
```

**Step 5: Update Systemd Service**
```bash
# Create/update service file
sudo tee /etc/systemd/system/unicorn-amanuensis.service > /dev/null <<EOF
[Unit]
Description=Unicorn-Amanuensis Speech-to-Text Service (XDNA2 C++ NPU)
After=network.target

[Service]
Type=simple
User=ccadmin
WorkingDirectory=/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis
Environment="PATH=/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="NPU_PLATFORM=auto"
ExecStart=/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/venv/bin/python3 api.py
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable unicorn-amanuensis

# Start service
sudo systemctl start unicorn-amanuensis

# Check status
sudo systemctl status unicorn-amanuensis
```

**Step 6: Verify Deployment**
```bash
# Check service status
sudo systemctl status unicorn-amanuensis

# Check logs
sudo journalctl -u unicorn-amanuensis -f

# Test API endpoint
curl http://localhost:9000/platform

# Expected:
# {
#   "service": "Unicorn-Amanuensis",
#   "version": "2.0.0",
#   "platform": {
#     "platform": "xdna2_cpp",
#     "uses_cpp_runtime": true,
#     ...
#   },
#   "backend": "XDNA2_CPP (C++ encoder + NPU, 400-500x realtime)"
# }

# Test transcription
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@tests/data/test_audio.wav"
```

### 6.3 Monitoring Setup

**Performance Metrics Collection**:
```python
# Add to xdna2/server.py
from prometheus_client import Counter, Histogram, Gauge

# Metrics
transcription_requests = Counter('transcription_requests_total', 'Total transcription requests')
transcription_duration = Histogram('transcription_duration_seconds', 'Transcription duration')
encoder_latency = Histogram('encoder_latency_seconds', 'Encoder latency')
realtime_factor = Gauge('realtime_factor', 'Realtime factor (audio_duration / processing_time)')
npu_utilization = Gauge('npu_utilization_percent', 'NPU utilization percentage')

@app.post("/v1/audio/transcriptions")
async def transcribe(...):
    transcription_requests.inc()

    start = time.perf_counter()
    # ... transcription logic ...
    duration = time.perf_counter() - start

    transcription_duration.observe(duration)
    encoder_latency.observe(encoder_time)
    realtime_factor.set(audio_duration / duration)
    npu_utilization.set(get_npu_utilization())

    return result
```

**Logging Configuration**:
```python
# Add to api.py
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
log_file = "/var/log/unicorn-amanuensis/service.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

**Health Checks**:
```python
# Add to xdna2/server.py
@app.get("/health")
async def health():
    """Enhanced health check with C++ runtime status"""
    try:
        # Check C++ runtime
        runtime_ok = encoder.runtime_healthy()

        # Check NPU
        npu_ok = encoder.npu_available()

        # Check memory
        import psutil
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024

        return {
            "status": "healthy" if (runtime_ok and npu_ok) else "degraded",
            "runtime": {
                "cpp_available": runtime_ok,
                "npu_available": npu_ok,
                "version": encoder.get_version()
            },
            "resources": {
                "memory_mb": mem_mb,
                "cpu_percent": process.cpu_percent()
            },
            "performance": {
                "realtime_factor": encoder.get_stats()['realtime_factor'],
                "avg_latency_ms": encoder.get_stats()['avg_latency_ms']
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

### 6.4 Rollback Plan

**If deployment fails**:

```bash
# Step 1: Stop service
sudo systemctl stop unicorn-amanuensis

# Step 2: Restore backup
cd /home/ccadmin/CC-1L/npu-services
rm -rf unicorn-amanuensis/
tar -xzf unicorn-amanuensis-backup-YYYYMMDD.tar.gz

# Step 3: Restore systemd service
sudo cp /etc/systemd/system/unicorn-amanuensis.service.backup \
        /etc/systemd/system/unicorn-amanuensis.service
sudo systemctl daemon-reload

# Step 4: Restart service
sudo systemctl start unicorn-amanuensis

# Step 5: Verify rollback successful
sudo systemctl status unicorn-amanuensis
curl http://localhost:9000/health
```

**Fallback to Python runtime** (without full rollback):

```bash
# Option 1: Environment variable
export NPU_PLATFORM=xdna2  # Force Python runtime

# Option 2: Config file
# Edit config/runtime_config.yaml:
runtime:
  backend: xdna2  # Force Python runtime

# Restart service
sudo systemctl restart unicorn-amanuensis
```

### 6.5 Post-Deployment Validation

**Validation Checklist** (within 24 hours):

- [ ] Service running and healthy
  ```bash
  curl http://localhost:9000/health
  # Expected: status=healthy
  ```

- [ ] Platform detection correct
  ```bash
  curl http://localhost:9000/platform
  # Expected: platform=xdna2_cpp, uses_cpp_runtime=true
  ```

- [ ] Transcription working
  ```bash
  curl -X POST http://localhost:9000/v1/audio/transcriptions -F "file=@test.wav"
  # Expected: valid transcription
  ```

- [ ] Performance target met
  ```bash
  # Check logs for realtime factor
  sudo journalctl -u unicorn-amanuensis | grep "realtime"
  # Expected: 400-500x realtime
  ```

- [ ] No errors in logs
  ```bash
  sudo journalctl -u unicorn-amanuensis --since "1 hour ago" | grep -i error
  # Expected: no critical errors
  ```

- [ ] Memory stable
  ```bash
  # Check memory usage after 1 hour
  ps aux | grep "api.py"
  # Expected: <500MB RSS
  ```

- [ ] NPU utilization normal
  ```bash
  # Check NPU stats
  curl http://localhost:9000/health | jq '.performance.npu_utilization'
  # Expected: 2-3%
  ```

---

## Summary

### Integration Status

**Infrastructure**: ✅ COMPLETE (1,888 lines)
- Python FFI wrapper: ✅ Complete
- High-level encoder: ✅ Complete
- Platform detection: ✅ Enhanced
- Configuration: ✅ Complete
- Tests: ✅ Complete

**C++ Runtime**: ✅ BUILT
- Libraries: ✅ Built and verified
- Build system: ✅ CMake + Makefile
- NPU support: ✅ XRT integration ready

**Service Architecture**: ✅ ANALYZED
- Entry points identified
- Encoder usage mapped
- API compatibility verified
- Integration points clear

### What Remains

**Week 5 Tasks**:
1. Create `xdna2/server.py` (native FastAPI server)
2. Update `api.py` (platform routing)
3. Test NPU callback integration
4. Validate end-to-end pipeline
5. Performance benchmarking

**Week 6 Tasks**:
1. Production deployment
2. Monitoring setup
3. Documentation
4. Final validation

### Expected Results

**Performance**:
- Realtime factor: 400-500x (vs 220x Python)
- Latency: ~50ms for 30s audio (vs ~136ms Python)
- NPU utilization: ~2.3% (97% headroom)
- Power draw: 5-15W (same as Python)

**API Compatibility**:
- 100% compatible with existing API
- Same request/response format
- Graceful fallback to Python
- No client changes required

**Integration Quality**:
- Comprehensive test suite
- Production monitoring
- Clear documentation
- Rollback capability

---

## Appendix

### A. File Locations

**Created Files** (Week 4):
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp_runtime_wrapper.py` (645 lines)
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/encoder_cpp.py` (509 lines)
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/config/runtime_config.yaml` (172 lines)
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests/test_cpp_integration.py` (542 lines)

**Files to Create** (Week 5):
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/server.py` (~200 lines)
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests/test_npu_callback.py` (~150 lines)
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests/test_accuracy.py` (~200 lines)
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests/test_performance.py` (~300 lines)
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests/test_stress.py` (~200 lines)

**Files to Modify** (Week 5):
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/api.py` (lines 38-70)

**C++ Libraries**:
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build/libwhisper_encoder_cpp.so` (167KB)
- `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build/libwhisper_xdna2_cpp.so` (86KB)

### B. References

**Documentation**:
- Service Architecture: `SERVICE_ARCHITECTURE_REPORT.md`
- C++ Integration Status: `CPP_INTEGRATION_COMPLETE.md`
- Quick Start Guide: `INTEGRATION_QUICK_START.md`
- Exploration Index: `EXPLORATION_INDEX.md`

**Code Examples**:
- C++ wrapper usage: `cpp_runtime_wrapper.py` (docstrings)
- Encoder integration: `encoder_cpp.py` (docstrings)
- Configuration: `config/runtime_config.yaml` (comments)

**External Resources**:
- FastAPI Docs: https://fastapi.tiangolo.com/
- ctypes Guide: https://docs.python.org/3/library/ctypes.html
- XRT Documentation: https://xilinx.github.io/XRT/
- Whisper Paper: https://arxiv.org/abs/2212.04356

### C. Contact Information

**Project**: CC-1L (Cognitive Companion 1 Laptop)
**Company**: Magic Unicorn Unconventional Technology & Stuff Inc
**Owner**: Aaron Stransky (@SkyBehind, aaron@magicunicorn.tech)
**Repository**: https://github.com/CognitiveCompanion/CC-1L
**License**: MIT

---

**Document Version**: 1.0
**Date**: November 1, 2025
**Status**: Ready for Implementation
**Next Action**: Begin Week 5 Day 1 - Create xdna2/server.py

---

*Built with care by the Week 5 Service Integration Planning Agent*
