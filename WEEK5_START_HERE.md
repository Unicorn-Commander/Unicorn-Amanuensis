# Week 5 Integration Planning - START HERE

**Service**: Unicorn-Amanuensis (Speech-to-Text)
**Planning Agent**: Service Integration Planning Agent
**Date**: November 1, 2025
**Status**: Planning Complete - Ready for Implementation

---

## Welcome to Week 5!

This document is your starting point for integrating the C++ NPU encoder into the Unicorn-Amanuensis service.

---

## Quick Navigation

### 1. Executive Summary (Read First)
**Document**: `WEEK5_INTEGRATION_SUMMARY.md` (361 lines, 5 min read)

**What's Inside**:
- TL;DR of what's done and what remains
- Quick stats (infrastructure 100% complete!)
- Week 5 checklist (5 days, ~15 tasks)
- Success criteria

**Why Read This**: Get oriented quickly, understand scope, see the big picture.

---

### 2. Complete Integration Plan (Main Reference)
**Document**: `UNICORN_AMANUENSIS_INTEGRATION_PLAN.md` (1,577 lines, 30 min read)

**What's Inside**:
1. **Service Architecture Analysis** - Where encoder is used, how it works
2. **Integration Design** - API changes, initialization, runtime behavior
3. **Implementation Checklist** - Step-by-step tasks for Week 5-6
4. **Risk Assessment** - What could go wrong and how to handle it
5. **Testing Plan** - Unit, integration, accuracy, performance, stress tests
6. **Deployment Strategy** - Pre-deployment checks, deployment steps, monitoring, rollback

**Why Read This**: Comprehensive guide with all details needed for implementation.

---

### 3. Supporting Documentation

#### C++ Integration Status
**Document**: `CPP_INTEGRATION_COMPLETE.md` (18KB, 10 min read)

**What's Inside**:
- Files created/modified in Week 4
- Architecture diagrams
- Integration status (infrastructure 100% done)
- Testing commands
- Troubleshooting guide

**Why Read This**: Understand what infrastructure is already built.

---

#### Quick Start Guide
**Document**: `INTEGRATION_QUICK_START.md` (9.1KB, 5 min read)

**What's Inside**:
- Quick test commands
- File locations
- How to use the C++ encoder
- Configuration options
- API compatibility examples

**Why Read This**: Fast hands-on testing of existing components.

---

#### Service Architecture
**Document**: `SERVICE_ARCHITECTURE_REPORT.md` (32KB, 15 min read)

**What's Inside**:
- Complete service architecture analysis
- Entry points and interfaces
- Current encoder implementation
- Recommended integration architecture
- Step-by-step integration plan (6 phases)

**Why Read This**: Deep dive into service internals and integration strategy.

---

## What You Have (Week 4 Complete)

### Infrastructure (1,888 lines of code)

```
✅ Python FFI Wrapper (645 lines)
   └─ xdna2/cpp_runtime_wrapper.py
      - ctypes-based C API wrapper
      - Zero-copy numpy integration
      - Context managers for resource cleanup
      - Comprehensive error handling

✅ High-Level Encoder (509 lines)
   └─ xdna2/encoder_cpp.py
      - Drop-in replacement for Python encoder
      - Same API as xdna2.encoder
      - NPU callback integration
      - BF16 workaround support

✅ Platform Detection (enhanced)
   └─ runtime/platform_detector.py
      - Platform.XDNA2_CPP enum
      - _has_cpp_runtime() method
      - Auto-selects C++ when available

✅ Configuration System (172 lines)
   └─ config/runtime_config.yaml
      - Runtime selection (auto, xdna2_cpp, xdna2, xdna1, cpu)
      - NPU configuration
      - Quantization settings
      - Performance targets

✅ Integration Tests (542 lines)
   └─ tests/test_cpp_integration.py
      - Library loading tests
      - Layer creation/destruction tests
      - Weight loading tests
      - Forward pass tests
      - Platform detection tests
      - Performance benchmarks
```

### C++ Runtime (254KB total)

```
✅ C++ Encoder Library (167KB)
   └─ xdna2/cpp/build/libwhisper_encoder_cpp.so
      - 6 encoder layers
      - Attention + FFN
      - INT8 quantization
      - BFP16 converter

✅ NPU Support Library (86KB)
   └─ xdna2/cpp/build/libwhisper_xdna2_cpp.so
      - XRT integration
      - NPU callback interface
      - BF16 workaround
```

### Test Results

```
✅ Library loads successfully
✅ Layers create/destroy without leaks
✅ Weights load (FP32 → INT8)
✅ Forward pass executes (CPU)
✅ Platform detection works
⏳ NPU callback (requires hardware testing)
⏳ End-to-end pipeline (Week 5)
⏳ Performance validation (Week 5)
```

---

## What You Need to Do (Week 5)

### Day 1-2: Service Integration (~6 hours)

**Deliverable**: Native XDNA2 FastAPI server

**Tasks**:
1. Create `xdna2/server.py` (~200 lines)
   - FastAPI app with `/v1/audio/transcriptions` endpoint
   - Use `encoder_cpp.forward()` for encoding
   - Reuse mel spectrogram preprocessing (Python)
   - Reuse decoder (Python)

2. Update `api.py` (lines 38-70)
   - Detect `Platform.XDNA2_CPP`
   - Import `xdna2.server` for C++ backend
   - Load runtime configuration
   - Add graceful fallback logic

3. Test basic integration
   - Service starts without errors
   - Platform detector selects C++ runtime
   - Encoder initializes successfully
   - Weights load correctly

**Reference**: Integration Plan Section 2.1 (API Integration Strategy)

---

### Day 3-4: NPU Callback Testing (~6 hours)

**Deliverable**: Working NPU integration

**Tasks**:
1. Verify NPU initialization
   - Load XRT device
   - Load kernel file (.xclbin)
   - Test basic matmul operation

2. Test callback registration
   - Register callback with C++ encoder
   - Verify callback signature
   - Test data flow: Python → C++ → NPU → Python

3. Test with real audio
   - Load sample audio file
   - Run encoder (triggers NPU callbacks)
   - Verify output correctness
   - Compare with Python baseline

4. Measure NPU performance
   - Latency per layer
   - Total encoder latency
   - NPU utilization

**Reference**: Integration Plan Section 3 (Implementation Checklist, Phase 2)

---

### Day 5: End-to-End Validation (~6 hours)

**Deliverable**: Validated pipeline meeting performance target

**Tasks**:
1. Full pipeline testing
   - Complete transcription with C++ encoder
   - Verify text output accuracy
   - Compare with Python results

2. Performance benchmarking
   - Measure realtime factor (target: 400-500x)
   - Measure latency (target: <60ms for 30s audio)
   - Verify NPU utilization (~2.3%)

3. Stress testing
   - Long audio files (>5 minutes)
   - Concurrent requests
   - Memory stability (no leaks)

4. Create test reports
   - Accuracy validation results
   - Performance benchmark results
   - Stress test results

**Reference**: Integration Plan Section 5 (Testing Plan)

---

## Performance Targets

| Metric | Python Baseline | C++ Target | How to Measure |
|--------|-----------------|------------|----------------|
| Realtime Factor | 220x | 400-500x | `audio_duration / processing_time` |
| Latency (30s) | ~136ms | ~50-60ms | `time.perf_counter()` around encoder |
| Per-Layer Time | ~23ms | ~8-10ms | Time each layer forward pass |
| NPU Utilization | ~5% | ~2.3% | `encoder.get_stats()['npu_utilization']` |
| Memory (RSS) | ~300MB | ~350MB | `psutil.Process().memory_info().rss` |

---

## Success Criteria

### Must Have (Critical)
- [ ] Service starts with C++ encoder
- [ ] Platform detector selects `XDNA2_CPP`
- [ ] NPU callbacks execute successfully
- [ ] Transcription accuracy within 1% of Python
- [ ] Realtime factor ≥400x
- [ ] No memory leaks over 100 requests

### Nice to Have (Important)
- [ ] Detailed performance metrics logged
- [ ] All test suites passing (unit, integration, performance)
- [ ] Documentation updated
- [ ] Monitoring enabled

### Bonus (Optional)
- [ ] Prometheus metrics endpoint
- [ ] Grafana dashboard
- [ ] Alert rules configured

---

## Quick Commands

### Test Existing Infrastructure

```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis

# Test C++ runtime wrapper
python3 xdna2/cpp_runtime_wrapper.py

# Test high-level encoder
python3 xdna2/encoder_cpp.py

# Run integration tests
python3 tests/test_cpp_integration.py

# Test platform detection
python3 -c "from runtime.platform_detector import get_platform_info; \
            import json; print(json.dumps(get_platform_info(), indent=2))"
```

### Start Service (After Week 5 Implementation)

```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis

# Manual start (foreground, for testing)
python3 api.py

# Test endpoint
curl http://localhost:9000/platform
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@tests/data/test_audio.wav"

# Production start (systemd)
sudo systemctl start unicorn-amanuensis
sudo systemctl status unicorn-amanuensis
sudo journalctl -u unicorn-amanuensis -f
```

### Performance Testing

```bash
# Run performance tests
python3 tests/test_performance.py

# Measure realtime factor
python3 -c "
from xdna2.server import transcribe
import time

audio_path = 'tests/data/30sec_audio.wav'
start = time.perf_counter()
result = transcribe(audio_path)
elapsed = time.perf_counter() - start

print(f'Realtime factor: {30.0 / elapsed:.1f}x')
print(f'Latency: {elapsed * 1000:.1f}ms')
"
```

---

## Troubleshooting

### Issue: C++ Libraries Not Found

**Symptom**: `CPPRuntimeError: C++ runtime library not found`

**Solution**:
```bash
# Check if libraries exist
ls -la /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build/*.so

# If missing, rebuild
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp
./build.sh
```

---

### Issue: Platform Detector Doesn't Select C++

**Symptom**: Platform detector selects `xdna2` instead of `xdna2_cpp`

**Solution**:
```bash
# Force C++ runtime
export NPU_PLATFORM=xdna2_cpp

# Or edit config
vim config/runtime_config.yaml
# Set: runtime.backend: xdna2_cpp
```

---

### Issue: NPU Callback Fails

**Symptom**: NPU callback initialization fails

**Check**:
```bash
# Verify XRT installed
/opt/xilinx/xrt/bin/xbutil examine

# Check kernel files
ls -la /opt/xilinx/xrt/share/*.xclbin

# Test without NPU (CPU only)
# In encoder creation:
encoder = WhisperEncoderCPP(use_npu=False)
```

---

### Issue: Performance Below Target

**Symptom**: Realtime factor <400x

**Debug**:
```bash
# Profile hot paths
python3 -m cProfile -o profile.stats api.py
python3 -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Check NPU utilization
curl http://localhost:9000/health | jq '.performance'

# Enable detailed timing
# Edit config/runtime_config.yaml:
# performance.monitoring.detailed_timing: true
```

---

## Files to Create This Week

| File | Lines | Purpose | Priority |
|------|-------|---------|----------|
| `xdna2/server.py` | ~200 | Native FastAPI server | P0 (Critical) |
| `tests/test_npu_callback.py` | ~150 | NPU callback tests | P0 (Critical) |
| `tests/test_accuracy.py` | ~200 | Accuracy validation | P1 (High) |
| `tests/test_performance.py` | ~300 | Performance benchmarks | P1 (High) |
| `tests/test_stress.py` | ~200 | Stress tests | P2 (Medium) |

---

## Documentation Index

| Document | Size | Purpose | Read Time |
|----------|------|---------|-----------|
| `WEEK5_START_HERE.md` | This file | Navigation hub | 5 min |
| `WEEK5_INTEGRATION_SUMMARY.md` | 361 lines | Executive summary | 5 min |
| `UNICORN_AMANUENSIS_INTEGRATION_PLAN.md` | 1,577 lines | Complete plan | 30 min |
| `CPP_INTEGRATION_COMPLETE.md` | 665 lines | Week 4 status | 10 min |
| `INTEGRATION_QUICK_START.md` | 404 lines | Quick testing guide | 5 min |
| `SERVICE_ARCHITECTURE_REPORT.md` | 1,011 lines | Service internals | 15 min |

**Total Documentation**: ~4,400 lines (~70 minutes to read everything)

**Recommended Reading Order**:
1. This file (WEEK5_START_HERE.md) - 5 min
2. WEEK5_INTEGRATION_SUMMARY.md - 5 min
3. Skim UNICORN_AMANUENSIS_INTEGRATION_PLAN.md - 15 min
4. Read specific sections as needed - on demand

---

## Contact & Resources

**Project**: CC-1L (Cognitive Companion 1 Laptop)
**Company**: Magic Unicorn Unconventional Technology & Stuff Inc
**Owner**: Aaron Stransky (@SkyBehind, aaron@magicunicorn.tech)
**Repository**: https://github.com/CognitiveCompanion/CC-1L
**License**: MIT

**External Resources**:
- FastAPI Docs: https://fastapi.tiangolo.com/
- ctypes Guide: https://docs.python.org/3/library/ctypes.html
- XRT Documentation: https://xilinx.github.io/XRT/
- Whisper Paper: https://arxiv.org/abs/2212.04356

---

## Next Steps

### Right Now (5 minutes)
1. Read `WEEK5_INTEGRATION_SUMMARY.md`
2. Skim `UNICORN_AMANUENSIS_INTEGRATION_PLAN.md` Table of Contents
3. Test existing infrastructure with quick commands above

### Tomorrow (Day 1)
1. Read Integration Plan Section 2.1 (API Integration Strategy)
2. Create `xdna2/server.py`
3. Update `api.py`
4. Test basic integration

### This Week (Days 1-5)
1. Days 1-2: Service Integration
2. Days 3-4: NPU Callback Testing
3. Day 5: End-to-End Validation

### Next Week (Week 6)
1. Days 1-2: Production Deployment
2. Days 3-5: Validation & Monitoring

---

## Summary

**Where We Are**:
- ✅ C++ runtime libraries built (254KB)
- ✅ Python FFI wrapper complete (645 lines)
- ✅ Integration infrastructure ready (1,888 lines)
- ✅ Platform detection enhanced
- ✅ Tests written (542 lines)

**What's Next**:
- Create native XDNA2 server (~200 lines)
- Test NPU callbacks with real audio
- Validate end-to-end pipeline
- Measure performance (target: 400-500x)
- Deploy to production

**Expected Outcome**:
- 400-500x realtime performance
- 2.3x lower latency (~50ms vs ~136ms)
- 100% API compatibility
- Production-ready service

**Timeline**: 5 days (Week 5) + 2 days (Week 6 deployment)

---

**Status**: Ready to Begin Week 5 Implementation

**First Task**: Read `WEEK5_INTEGRATION_SUMMARY.md`

**Good luck!** 🚀

---

*Service Integration Planning Agent - Week 5*
*CC-1L Project - Phase 7*
*November 1, 2025*
