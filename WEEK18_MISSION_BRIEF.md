# Week 18 Mission Brief

**Date**: November 2, 2025
**Duration Budget**: 6-8 hours (3 parallel teams)
**Mission**: Decoder Optimization & Performance Breakthrough

---

## Mission Context

**Week 17 Critical Discovery**: NPU is working perfectly and fast (50-80ms, 6-10% of total time). The bottleneck is the Python Whisper decoder taking 500-600ms (62-75% of total time).

**Current Performance**: 1.6-11.9× realtime
**Target Performance**: 400-500× realtime
**Gap**: ~50× slower than target

**Root Cause**: Decoder running on CPU in Python takes 500-600ms per inference.

---

## Week 18 Objective

**Primary Goal**: Reduce total processing time from 800-900ms to 10-20ms (50× speedup)

**Target Milestone**: Achieve 100-200× realtime factor (intermediate toward 400-500×)

**Success Criteria**:
- [ ] Decoder time reduced from 500-600ms to <50ms (10× speedup minimum)
- [ ] 30-second audio clips supported (buffer fix)
- [ ] Performance profiling framework operational
- [ ] Multi-stream testing validated (4+ concurrent streams)
- [ ] Documentation updated with optimization results

---

## Team Structure

### Team 1: Decoder Optimization (Priority P0) 🔥
**Team Lead**: Decoder Optimization Specialist
**Duration**: 3-4 hours
**Budget**: Critical path

**Mission**: Replace or optimize Python Whisper decoder to reduce processing time from 500-600ms to <50ms.

**Objectives**:
1. **Research decoder options** (1 hour):
   - C++ Whisper.cpp integration
   - ONNX Runtime optimized inference
   - PyTorch optimizations
   - Batch processing strategies

2. **Implement optimization** (2 hours):
   - Integrate fastest solution
   - Benchmark performance improvement
   - Validate accuracy (maintain >95%)

3. **Validate integration** (1 hour):
   - End-to-end transcription tests
   - Accuracy validation
   - Performance measurement

**Success Metrics**:
- Decoder time: <50ms (vs 500-600ms current)
- Accuracy: >95% (maintain quality)
- Realtime factor: 100-200× (intermediate milestone)

**Deliverables**:
- Optimized decoder implementation
- Performance benchmark results
- Integration test results
- Technical report with recommendations

---

### Team 2: Buffer Management & Long-form Audio (Priority P1)
**Team Lead**: Buffer Management Specialist
**Duration**: 2-3 hours

**Mission**: Fix 30-second audio buffer size limit and validate long-form transcription.

**Objectives**:
1. **Fix buffer pool size** (30 minutes):
   - Increase GlobalBufferManager pool size
   - Support 30+ second audio clips
   - Test with 30s, 60s, 120s audio

2. **Optimize buffer management** (1 hour):
   - Implement buffer reuse
   - Add buffer pooling strategies
   - Reduce memory footprint

3. **Long-form testing** (1 hour):
   - Create long-form test suite
   - Validate 30s, 60s, 120s audio
   - Performance impact analysis

**Success Metrics**:
- 30+ second audio: Working
- Buffer pool size: Configurable and optimized
- Memory usage: <100 MB for 120s audio

**Deliverables**:
- Buffer size configuration changes
- Long-form audio test suite
- Buffer optimization recommendations
- Performance impact report

---

### Team 3: Performance Profiling & Multi-stream (Priority P2)
**Team Lead**: Performance Engineering Specialist
**Duration**: 2-3 hours

**Mission**: Create comprehensive performance profiling framework and validate multi-stream execution.

**Objectives**:
1. **Performance profiling** (1.5 hours):
   - Implement detailed timing instrumentation
   - Profile all pipeline stages (mel, encoder, decoder)
   - Identify remaining bottlenecks
   - Create performance visualization

2. **Multi-stream testing** (1 hour):
   - Implement concurrent request testing
   - Test 4, 8, 16 concurrent streams
   - Measure throughput and latency
   - Validate NPU resource sharing

3. **Optimization roadmap** (30 minutes):
   - Identify next optimization opportunities
   - Create Week 19-20 roadmap
   - Document path to 400-500× target

**Success Metrics**:
- Profiling framework: Detailed timing for all stages
- Multi-stream: 4+ concurrent streams working
- Throughput: Combined >100× realtime with 4 streams

**Deliverables**:
- Performance profiling tool
- Multi-stream test results
- Performance visualization
- Week 19-20 optimization roadmap

---

## Team Coordination

### Communication Protocol
- **Shared Resources**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/`
- **Test Data**: `tests/test_*.wav` (1s, 5s, 30s, silence)
- **Results**: `tests/results/week18_*.json`

### Dependencies
- **Team 1 → Team 3**: Decoder optimization impacts profiling results
- **Team 2 → Team 3**: Buffer changes affect multi-stream testing
- **All teams**: Independent until integration phase

### Risk Management
- **Team 1 critical path**: If decoder optimization blocked, fall back to PyTorch optimizations
- **Team 2 fallback**: If buffer issues arise, focus on streaming approach
- **Team 3 independent**: Can proceed regardless of Team 1/2 status

---

## Technical Approach

### Team 1: Decoder Options Analysis

**Option 1: Whisper.cpp Integration** (RECOMMENDED)
- **Pros**: C++ implementation, 10-50× faster than Python
- **Cons**: Integration effort, need Python bindings
- **Estimated speedup**: 10-20×
- **Estimated effort**: 2-3 hours
- **Risk**: Medium (integration complexity)

**Option 2: ONNX Runtime**
- **Pros**: Optimized inference, GPU support
- **Cons**: Model conversion required
- **Estimated speedup**: 5-10×
- **Estimated effort**: 1-2 hours
- **Risk**: Low (well-documented)

**Option 3: PyTorch Optimizations**
- **Pros**: Easy to implement, low risk
- **Cons**: Limited speedup potential
- **Estimated speedup**: 2-3×
- **Estimated effort**: 1 hour
- **Risk**: Very low

**Option 4: Batch Processing**
- **Pros**: Amortize overhead, higher throughput
- **Cons**: Increases latency per request
- **Estimated speedup**: 3-5×
- **Estimated effort**: 2 hours
- **Risk**: Low

**Recommendation**: Try Option 1 (Whisper.cpp) first, fall back to Option 2 (ONNX) if integration blocked.

---

### Team 2: Buffer Strategy

**Current Issue**: Buffer pool sized for ~10 second audio clips
**Root Cause**: Conservative initial sizing

**Solution Approach**:
1. Make buffer pool size configurable (environment variable)
2. Implement buffer reuse (release after processing)
3. Add buffer streaming (process in chunks)

**Configuration Changes**:
```python
# Current
MAX_BUFFER_SIZE = 10 * 16000  # 10 seconds

# Proposed
MAX_BUFFER_SIZE = int(os.getenv('MAX_AUDIO_DURATION', 120)) * 16000
```

---

### Team 3: Profiling Architecture

**Profiling Levels**:
1. **Coarse**: Total time per stage (mel, encoder, decoder)
2. **Medium**: Substage timing (encode layers, decoder attention, etc.)
3. **Fine**: Per-operation timing (matmul, softmax, etc.)

**Tools**:
- Python `time.perf_counter()` for timing
- cProfile for detailed profiling
- Custom instrumentation for NPU operations

**Metrics to Track**:
- Execution time (mean, p50, p95, p99)
- Memory usage (peak, average)
- NPU utilization
- Throughput (audio seconds per wall-clock second)
- Latency (wall-clock time per audio second)

---

## Expected Outcomes

### Conservative Scenario (75% confidence)
- **Decoder**: 100-150ms (3-5× speedup) via PyTorch optimizations
- **Buffer**: 30s audio working
- **Multi-stream**: 4 streams validated
- **Realtime factor**: 20-40× (progress toward target)

### Target Scenario (60% confidence)
- **Decoder**: 50-80ms (7-10× speedup) via ONNX Runtime
- **Buffer**: 120s audio working
- **Multi-stream**: 8 streams working
- **Realtime factor**: 100-150× (significant milestone)

### Stretch Scenario (40% confidence)
- **Decoder**: 20-40ms (15-25× speedup) via Whisper.cpp
- **Buffer**: Unlimited (streaming approach)
- **Multi-stream**: 16+ streams working
- **Realtime factor**: 200-300× (approaching target)

---

## Week 19-20 Preview

**If Week 18 achieves target scenario**:
- Week 19: Batch processing, multi-tile NPU scaling
- Week 20: Final optimization, production deployment

**If Week 18 achieves conservative scenario**:
- Week 19: Continue decoder optimization (Whisper.cpp)
- Week 20: NPU scaling, batch processing

**Path to 400-500× target**:
1. Week 18: Decoder optimization → 100-200×
2. Week 19: Batch processing + multi-tile → 250-350×
3. Week 20: Final tuning → 400-500×

**Confidence**: 85% achievable by end of Week 20

---

## Timeline

### Hour 0-1: Setup & Planning
- All teams: Review mission brief
- All teams: Set up testing environment
- All teams: Create task breakdown

### Hour 1-4: Implementation (Parallel)
- **Team 1**: Research + implement decoder optimization
- **Team 2**: Fix buffer + test long-form audio
- **Team 3**: Build profiling framework

### Hour 4-6: Integration & Testing
- **Team 1**: Validate decoder integration
- **Team 2**: Long-form stress testing
- **Team 3**: Multi-stream validation

### Hour 6-8: Documentation & Consolidation
- All teams: Write final reports
- All teams: Create consolidated Week 18 summary
- All teams: Commit and push results

---

## Success Criteria Checklist

### Must Have (P0)
- [ ] Decoder time <100ms (>5× speedup)
- [ ] End-to-end transcription still working
- [ ] Accuracy >95% maintained

### Should Have (P1)
- [ ] 30s audio working
- [ ] Realtime factor >50×
- [ ] Performance profiling operational

### Nice to Have (P2)
- [ ] Decoder time <50ms (>10× speedup)
- [ ] 120s audio working
- [ ] Multi-stream 8+ concurrent
- [ ] Realtime factor >100×

---

## Risk Mitigation

### Technical Risks
1. **Decoder integration complexity** (High)
   - Mitigation: Multiple options (Whisper.cpp, ONNX, PyTorch)
   - Fallback: PyTorch optimizations (guaranteed some speedup)

2. **Accuracy degradation** (Medium)
   - Mitigation: Comprehensive accuracy testing
   - Fallback: Revert to Python decoder if accuracy <95%

3. **Buffer issues** (Low)
   - Mitigation: Simple configuration change
   - Fallback: Streaming approach

### Schedule Risks
1. **Team 1 takes longer** (Medium)
   - Mitigation: 4-hour budget with fallback options
   - Impact: Other teams can proceed independently

2. **Integration issues** (Low)
   - Mitigation: Comprehensive testing framework ready
   - Fallback: Staged rollout

---

## Resource Requirements

### Compute Resources
- **NPU**: Available (50 TOPS, <5% utilized)
- **CPU**: Available (16C/32T)
- **RAM**: 120GB (sufficient)
- **Storage**: 953GB free (sufficient)

### Software Dependencies
- **Whisper.cpp**: May need installation (30 minutes)
- **ONNX Runtime**: Available via pip (5 minutes)
- **Test audio**: Already created (1.28 MB)

### Data Requirements
- **Test audio**: ✅ Already available (1s, 5s, 30s, silence)
- **Long-form audio**: Need to create 60s, 120s clips (10 minutes)
- **Reference transcriptions**: Manual validation required

---

## Documentation Deliverables

### Team 1 Reports
- `WEEK18_DECODER_OPTIMIZATION_REPORT.md`
- `DECODER_PERFORMANCE_BENCHMARK.md`
- Code changes with comments

### Team 2 Reports
- `WEEK18_BUFFER_MANAGEMENT_REPORT.md`
- `LONG_FORM_AUDIO_TEST_RESULTS.md`
- Configuration guide

### Team 3 Reports
- `WEEK18_PERFORMANCE_PROFILING_REPORT.md`
- `WEEK18_MULTI_STREAM_RESULTS.md`
- `WEEK19_OPTIMIZATION_ROADMAP.md`

### Consolidated Reports
- `WEEK18_COMPLETE.md` (all teams summary)
- `WEEK18_EXECUTIVE_SUMMARY.md` (high-level results)
- `WEEK18_FINAL_STATUS.md` (status and next steps)

---

## Team Assignments

### Team 1: Decoder Optimization
**Lead**: Decoder Optimization Specialist (subagent)
**Focus**: Replace/optimize decoder for 10-25× speedup
**Priority**: P0 (critical path)
**Timeline**: 3-4 hours

### Team 2: Buffer Management
**Lead**: Buffer Management Specialist (subagent)
**Focus**: Long-form audio support, buffer optimization
**Priority**: P1 (high)
**Timeline**: 2-3 hours

### Team 3: Performance Engineering
**Lead**: Performance Engineering Specialist (subagent)
**Focus**: Profiling, multi-stream, optimization roadmap
**Priority**: P2 (medium)
**Timeline**: 2-3 hours

---

## Pre-Mission Checklist

### Environment Verification
- [ ] NPU operational (Week 16 verified ✅)
- [ ] End-to-end transcription working (Week 17 verified ✅)
- [ ] Test audio files available (Week 17 created ✅)
- [ ] Python environment active (`mlir-aie/ironenv`)
- [ ] XRT environment loaded (`/opt/xilinx/xrt/setup.sh`)

### Code Baseline
- [ ] Week 17 code committed and pushed ✅
- [ ] Clean git status
- [ ] All tests passing from Week 17

### Documentation Baseline
- [ ] Week 17 reports complete ✅
- [ ] Performance baseline documented ✅
- [ ] Architecture documented ✅

---

## Post-Mission Checklist

### Code & Testing
- [ ] All code changes committed
- [ ] Integration tests passing
- [ ] Performance benchmarks recorded
- [ ] Accuracy validation complete

### Documentation
- [ ] All team reports written
- [ ] Week 18 consolidated summary created
- [ ] Performance graphs/visualizations
- [ ] Week 19 roadmap documented

### Git Operations
- [ ] Submodule committed and pushed
- [ ] Main repository committed and pushed
- [ ] Tag created: `week18-decoder-optimization`

---

## Notes for Team Leads

### Communication
- Document all findings in detail
- Include code examples and benchmarks
- Explain technical decisions and trade-offs
- Highlight any blockers or risks encountered

### Testing
- Run comprehensive tests before declaring success
- Validate accuracy doesn't degrade
- Measure performance improvements quantitatively
- Test edge cases (long audio, silence, noise)

### Documentation
- Write for future developers
- Include setup instructions
- Document any configuration changes
- Create troubleshooting guides

### Coordination
- Be aware of other teams' progress
- Share findings that impact other teams
- Escalate blockers early
- Celebrate wins together

---

## Success Definition

**Week 18 is successful if**:
1. Decoder processing time reduced by ≥5× (500ms → ≤100ms)
2. End-to-end transcription still working with >95% accuracy
3. 30-second audio clips supported
4. Performance profiling framework operational
5. Path to 400-500× target clearly documented

**Stretch success**:
- Decoder time ≤50ms (≥10× speedup)
- Realtime factor ≥100×
- Multi-stream 8+ concurrent working
- 120-second audio supported

---

## Let's Ship It! 🚀

**Mission Start**: Now
**Mission End**: 6-8 hours from now
**Expected Outcome**: 100-200× realtime factor (halfway to 400-500× target)

**Week 17 Momentum**: End-to-end transcription working, NPU executing perfectly
**Week 18 Focus**: Optimize the decoder bottleneck
**Week 19 Target**: Final push to 400-500× with batching and multi-tile

**Confidence**: 85% - We know the bottleneck, we have the solution options, and the infrastructure is solid.

---

**Built with 🦄 by Magic Unicorn Unconventional Technology & Stuff Inc**
