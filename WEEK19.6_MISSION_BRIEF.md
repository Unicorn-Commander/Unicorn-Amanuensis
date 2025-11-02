# Week 19.6 Mission Brief: Stabilization & Recovery

**Date**: November 2, 2025
**Duration Budget**: 4-6 hours (3 parallel teams)
**Mission**: Rollback from Week 19.5 regression and stabilize Week 18 baseline
**Priority**: P0 - CRITICAL (blocking all future work)

---

## Mission Context

**Current Situation**: Week 19.5 architecture fix resulted in catastrophic regression:
- Performance: 7.9× → 2.7× realtime (66% WORSE)
- Multi-stream success: 100% → 26% (74% failures)
- Accuracy: Degraded (empty transcriptions, hallucinations)
- 30s audio: Still broken

**Week 18 Baseline** (stable, proven):
- Performance: 7.9× realtime average
- Multi-stream: 100% success rate
- Accuracy: Maintained
- Known issue: 30s audio fails (buffer size limit)

**Mission Objective**: Restore Week 18 stability and fix known issues to create solid foundation for Week 20 batch processing.

---

## Week 19.6 Objectives

**Primary Goal**: Restore Week 18 performance + fix buffer pool exhaustion + add instrumentation

**Success Criteria**:
- [ ] Performance ≥ 7.9× realtime (Week 18 parity)
- [ ] Multi-stream 100% success rate (no buffer exhaustion)
- [ ] 30s audio working (buffer pool fix)
- [ ] Component timing instrumentation operational
- [ ] All tests passing (4/5 from Week 17, plus 30s)

---

## Team Structure

### Team 1: Rollback & Buffer Pool Fix (Priority P0) 🔥
**Team Lead**: Rollback Specialist
**Duration**: 1-2 hours
**Budget**: Critical path

**Mission**: Revert Week 19.5 changes and fix buffer pool exhaustion

**Objectives**:
1. **Rollback Week 19.5** (30 minutes):
   - Option A: Git revert Week 19.5 commits
   - Option B: Disable via environment variables (cleaner for research)
   - Verify Week 18 code is active

2. **Fix Buffer Pool Exhaustion** (30 minutes):
   - Increase audio buffer pool from 5 to 50
   - Add buffer pool growth strategy
   - Configure for multi-stream workloads
   - Test with 4, 8, 16 concurrent streams

3. **Fix 30s Audio Support** (30 minutes):
   - Verify MAX_AUDIO_DURATION=30 configuration
   - Test 30s, 60s audio clips
   - Ensure buffer sizes scale correctly

**Deliverables**:
- Rollback implementation (code or config)
- Buffer pool configuration changes
- Multi-stream validation tests
- Long-form audio validation

**Success Metrics**:
- Performance: ≥7.9× realtime
- Multi-stream: 100% success (0% failures)
- 30s audio: Working
- Buffer pool: No exhaustion errors

---

### Team 2: Timing Instrumentation (Priority P0)
**Team Lead**: Performance Engineering Specialist
**Duration**: 2-3 hours

**Mission**: Add component-level timing to debug future regressions

**Objectives**:
1. **Design Timing Framework** (30 minutes):
   - Hierarchical timing (total → stages → substages)
   - Statistical collection (mean, p50, p95, p99)
   - Minimal overhead (<5ms)
   - JSON export for analysis

2. **Instrument Pipeline Stages** (1.5 hours):
   - Audio loading and preprocessing
   - Mel spectrogram computation
   - NPU encoder execution
   - Decoder execution
   - Post-processing
   - Total end-to-end time

3. **Add API Response Timing** (30 minutes):
   - Return timing breakdown in transcription responses
   - Optional verbose mode (detailed substages)
   - Timing visualization recommendations

**Implementation**:
```python
class ComponentTimer:
    """Hierarchical timing with minimal overhead"""

    def __init__(self):
        self.timings = {}
        self.stack = []

    @contextmanager
    def time(self, component: str):
        """Time a component"""
        start = time.perf_counter()
        self.stack.append(component)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            path = '.'.join(self.stack)
            if path not in self.timings:
                self.timings[path] = []
            self.timings[path].append(elapsed)
            self.stack.pop()

    def get_breakdown(self) -> Dict:
        """Get timing breakdown"""
        return {
            component: {
                'mean': statistics.mean(times),
                'p50': statistics.median(times),
                'p95': statistics.quantiles(times, n=20)[18],
                'count': len(times)
            }
            for component, times in self.timings.items()
        }

# Usage in transcription_pipeline.py
timer = ComponentTimer()

async def transcribe_audio(audio: np.ndarray):
    with timer.time('total'):
        with timer.time('mel'):
            mel = compute_mel(audio)
        with timer.time('encoder'):
            features = npu_encode(mel)
        with timer.time('decoder'):
            text = decode(features)

    return {
        'text': text,
        'timing': timer.get_breakdown()  # NEW!
    }
```

**API Response Format**:
```json
{
  "text": " Ooh.",
  "timing": {
    "total": {"mean": 328, "p50": 320, "p95": 350, "count": 1},
    "total.mel": {"mean": 150, "p50": 150, "p95": 155, "count": 1},
    "total.encoder": {"mean": 20, "p50": 20, "p95": 22, "count": 1},
    "total.decoder": {"mean": 158, "p50": 158, "p95": 160, "count": 1}
  }
}
```

**Deliverables**:
- ComponentTimer class implementation
- Pipeline instrumentation
- API response updates
- Timing visualization script

**Success Metrics**:
- All pipeline stages instrumented
- Overhead: <5ms
- Timing data in API responses
- Statistical aggregation working

---

### Team 3: Validation & Testing (Priority P1)
**Team Lead**: QA & Validation Specialist
**Duration**: 2-3 hours

**Mission**: Validate Week 18 baseline is restored and all fixes working

**Objectives**:
1. **Baseline Validation** (1 hour):
   - Run Week 17 test suite (1s, 5s, 30s, silence)
   - Verify performance ≥ Week 18
   - Verify accuracy maintained
   - Document results

2. **Multi-Stream Testing** (1 hour):
   - Test 4, 8, 16 concurrent streams
   - Verify 100% success rate (no buffer exhaustion)
   - Measure throughput and latency
   - Statistical validation

3. **Long-Form Audio** (30 minutes):
   - Test 30s, 60s audio clips
   - Verify no failures
   - Measure performance scaling
   - Document memory usage

4. **Regression Testing** (30 minutes):
   - Verify Week 19.5 issues are resolved
   - No empty transcriptions
   - No hallucinations on silence
   - Consistent results across runs

**Test Suite**:
```python
# tests/week19_6_validation.py

async def test_week18_baseline():
    """Verify Week 18 performance restored"""
    results = await run_baseline_tests()

    assert results['1s_audio']['realtime_factor'] >= 3.0
    assert results['5s_audio']['realtime_factor'] >= 10.0
    assert results['30s_audio']['status'] == 'PASS'  # NEW!
    assert results['silence']['text'] == ''

async def test_multi_stream_reliability():
    """Verify buffer pool fixes"""
    results = await run_concurrent_tests(
        streams=[4, 8, 16],
        duration=1.0
    )

    for stream_count, result in results.items():
        assert result['success_rate'] == 100.0  # No failures!
        assert 'buffer pool exhausted' not in result['errors']

async def test_component_timing():
    """Verify timing instrumentation"""
    result = await transcribe('tests/audio/test_5s.wav')

    assert 'timing' in result
    assert 'total' in result['timing']
    assert 'total.mel' in result['timing']
    assert 'total.encoder' in result['timing']
    assert 'total.decoder' in result['timing']
```

**Deliverables**:
- Week 19.6 validation test suite
- Baseline performance report
- Multi-stream reliability report
- Regression test results

**Success Metrics**:
- All baseline tests passing
- Performance ≥ Week 18
- Multi-stream: 100% success
- 30s audio: Working

---

## Team Coordination

### Communication Protocol
- **Shared Resources**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/`
- **Test Data**: `tests/audio/test_*.wav`
- **Results**: `tests/results/week19_6_*.json`

### Dependencies
- **Team 1 → Team 3**: Rollback must complete before validation
- **Team 2 → Team 3**: Instrumentation needed for timing validation
- **Teams can start in parallel**: Team 1 & 2 independent

### Risk Management
- **Rollback issues**: Git revert may have conflicts → use environment variables
- **Buffer pool changes**: May affect memory → monitor carefully
- **Instrumentation overhead**: May slow down service → measure impact

---

## Technical Approach

### Team 1: Rollback Strategy

**Option A: Git Revert** (Clean but destructive)
```bash
# Revert Week 19.5 commits
git revert <week19.5_commit_hash>

# Keep Week 19 batch processor for future use
git checkout <week19_commit_hash> -- xdna2/batch_processor.py
```

**Option B: Environment Variable Disable** (RECOMMENDED - preserves research)
```bash
# xdna2/server.py
USE_CUSTOM_DECODER = os.getenv('USE_CUSTOM_DECODER', 'false').lower() == 'true'
USE_FASTER_WHISPER = os.getenv('USE_FASTER_WHISPER', 'false').lower() == 'true'

# Start service with Week 18 config
USE_CUSTOM_DECODER=false USE_FASTER_WHISPER=false python -m uvicorn xdna2.server:app
```

**Recommendation**: Option B (preserves code for future debugging)

### Team 1: Buffer Pool Configuration

**Current (Week 18)**:
```python
# xdna2/server.py
buffer_manager.configure({
    'audio': {
        'size': MAX_AUDIO_SAMPLES * 4,
        'shape': (MAX_AUDIO_SAMPLES,),
        'pool_size': 5,  # TOO SMALL!
        'dtype': np.float32
    }
})
```

**Fixed (Week 19.6)**:
```python
# xdna2/server.py
AUDIO_BUFFER_POOL_SIZE = int(os.getenv('AUDIO_BUFFER_POOL_SIZE', '50'))

buffer_manager.configure({
    'audio': {
        'size': MAX_AUDIO_SAMPLES * 4,
        'shape': (MAX_AUDIO_SAMPLES,),
        'pool_size': AUDIO_BUFFER_POOL_SIZE,  # Increased from 5 to 50
        'dtype': np.float32,
        'growth_strategy': 'auto',  # Auto-grow if needed
        'max_pool_size': 100  # Safety limit
    },
    'mel': {
        'size': MAX_MEL_FRAMES * 80 * 4,
        'shape': (MAX_MEL_FRAMES, 80),
        'pool_size': 50,  # Increased from 10
        'dtype': np.float32
    },
    'encoder': {
        'size': MAX_ENCODER_FRAMES * 1500 * 4,
        'shape': (MAX_ENCODER_FRAMES, 1500),
        'pool_size': 50,  # Increased from 5
        'dtype': np.float32
    }
})
```

**Impact**: Support 50+ concurrent streams (vs 4-5 before)

---

## Expected Outcomes

### Conservative Scenario (90% confidence)
- **Performance**: 7.9× realtime (Week 18 parity)
- **Multi-stream**: 100% success at 4-8 streams
- **30s audio**: Working
- **Timing**: Basic instrumentation operational

### Target Scenario (70% confidence)
- **Performance**: 7.9× realtime (Week 18 parity)
- **Multi-stream**: 100% success at 16+ streams
- **30s audio**: Working (60s, 120s also working)
- **Timing**: Comprehensive instrumentation with statistics

### Stretch Scenario (40% confidence)
- **Performance**: 8-10× realtime (slight improvement via fixes)
- **Multi-stream**: 100% success at 32+ streams
- **30s audio**: Working (120s+ working)
- **Timing**: Real-time visualization dashboard

---

## Week 20 Preview

**After Week 19.6 stabilization complete**:

**Week 20: Batch Processing** (3-5 days)
- Test Week 19 Team 2 batch processor implementation
- Expected: 2-3× throughput improvement
- Target: 15-25× realtime average
- Confidence: 75% (proven technology, lower risk)

**Path to 400-500× target**:
```
Week 19.6 (stabilize): 7.9× realtime ✅ Foundation solid
Week 20 (batch): 15-25× realtime (2-3× improvement)
Week 21 (decoder opt): 60-150× realtime (4-6× improvement)
Week 22 (multi-tile): 240-1,200× realtime (4-8× improvement)
```

**Confidence in target**: ✅ **70%** (realistic, incremental path)

---

## Timeline

### Hour 0-1: Setup & Planning
- All teams: Review mission brief
- All teams: Set up testing environment
- Team 1: Decide rollback strategy (Option A vs B)

### Hour 1-3: Implementation (Parallel)
- **Team 1**: Rollback + buffer pool fix
- **Team 2**: Design + implement timing framework
- **Team 3**: Prepare validation test suite

### Hour 3-5: Integration & Testing
- **Team 1**: Validate buffer pool fixes
- **Team 2**: Integrate timing into pipeline
- **Team 3**: Run comprehensive validation

### Hour 5-6: Documentation & Consolidation
- All teams: Write final reports
- PM: Create consolidated Week 19.6 summary
- Commit and push results

---

## Success Criteria Checklist

### Must Have (P0)
- [ ] Performance ≥ 7.9× realtime (Week 18 parity)
- [ ] Multi-stream 100% success (no buffer exhaustion)
- [ ] 30s audio working
- [ ] Timing instrumentation operational

### Should Have (P1)
- [ ] 60s audio working
- [ ] Multi-stream 100% success at 16 streams
- [ ] Component timing in API responses
- [ ] Statistical timing aggregation

### Nice to Have (P2)
- [ ] 120s audio working
- [ ] Multi-stream 100% success at 32+ streams
- [ ] Real-time timing visualization
- [ ] Performance dashboard

---

## Risk Mitigation

### Technical Risks
1. **Rollback conflicts** (Medium)
   - Mitigation: Use environment variables instead of git revert
   - Fallback: Manual code restoration from Week 18

2. **Buffer pool memory usage** (Low)
   - Mitigation: Monitor memory usage, set max_pool_size
   - Fallback: Reduce pool size if memory issues

3. **Instrumentation overhead** (Low)
   - Mitigation: Use perf_counter, minimize allocations
   - Fallback: Make instrumentation optional

### Schedule Risks
1. **Teams take longer** (Low)
   - Mitigation: Simple tasks, well-defined scope
   - Impact: Week 20 delayed by 1-2 days

---

## Resource Requirements

### Compute Resources
- **NPU**: Available (operational since Week 14)
- **CPU**: 16C/32T (sufficient)
- **RAM**: 120GB (sufficient, even with 50 buffer pools)
- **Storage**: 953GB free (sufficient)

### Software Dependencies
- **All installed**: Week 18 codebase has all dependencies

### Data Requirements
- **Test audio**: ✅ Available (1s, 5s, 30s, silence)
- **Long-form audio**: ✅ Created in Week 18 (30s, 60s, 120s)

---

## Documentation Deliverables

### Team 1 Reports
- `WEEK19.6_ROLLBACK_REPORT.md`
- `WEEK19.6_BUFFER_POOL_FIX.md`
- Configuration guide

### Team 2 Reports
- `WEEK19.6_TIMING_INSTRUMENTATION_REPORT.md`
- ComponentTimer documentation
- API timing specification

### Team 3 Reports
- `WEEK19.6_VALIDATION_REPORT.md`
- Performance comparison (Week 18 vs 19.6)
- Multi-stream test results

### Consolidated Reports
- `WEEK19.6_COMPLETE.md` (all teams summary)
- `WEEK19.6_EXECUTIVE_SUMMARY.md` (high-level results)

---

## Pre-Mission Checklist

### Environment Verification
- [ ] Week 19.5 code committed ✅ (bf46924)
- [ ] NPU operational ✅ (Week 14 verified)
- [ ] Test audio files available ✅
- [ ] Python environment active

### Code Baseline
- [ ] Week 19 + 19.5 committed ✅
- [ ] Week 18 baseline documented ✅
- [ ] Rollback strategy selected

### Documentation Baseline
- [ ] Week 19.5 performance report ✅
- [ ] Week 19.5 architecture analysis ✅
- [ ] Recommendations documented ✅

---

## Post-Mission Checklist

### Code & Testing
- [ ] Rollback implemented and validated
- [ ] Buffer pool configuration updated
- [ ] Timing instrumentation integrated
- [ ] All tests passing

### Documentation
- [ ] All team reports written
- [ ] Week 19.6 consolidated summary
- [ ] Performance comparison graphs
- [ ] Week 20 roadmap updated

### Git Operations
- [ ] Changes committed
- [ ] Pushed to GitHub
- [ ] Tag created: `week19.6-stabilization`

---

## Notes for Team Leads

### Communication
- Document all findings in detail
- Include before/after comparisons
- Explain technical decisions
- Highlight any issues or blockers

### Testing
- Run comprehensive tests before declaring success
- Validate performance hasn't regressed
- Test edge cases (long audio, concurrent streams)
- Document memory usage

### Documentation
- Write for future developers
- Include configuration examples
- Document troubleshooting steps
- Create setup guides

### Coordination
- Share findings with other teams
- Escalate blockers early
- Celebrate wins
- Learn from Week 19.5 failures

---

## Success Definition

**Week 19.6 is successful if**:
1. Performance ≥ 7.9× realtime (Week 18 parity)
2. Multi-stream 100% success rate (no buffer exhaustion)
3. 30s audio working
4. Timing instrumentation operational
5. Ready for Week 20 batch processing

**Stretch success**:
- Performance > 8× realtime (slight improvement)
- Multi-stream 100% success at 32+ streams
- 120s audio working
- Real-time timing visualization

---

## Let's Recover and Build Solid Foundation! 🛠️

**Mission Start**: Now
**Mission End**: 4-6 hours from now
**Expected Outcome**: Stable Week 18 baseline + buffer pool fix + timing instrumentation

**Week 19.5 Lesson**: Don't deploy without thorough testing
**Week 19.6 Focus**: Stabilize foundation for future optimization
**Week 20 Target**: Batch processing for 2-3× improvement

**Confidence**: 90% - Simple, well-defined tasks with proven baseline

---

**Built with 🦄 by Magic Unicorn Unconventional Technology & Stuff Inc**
