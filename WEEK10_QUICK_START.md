# Week 10 Quick Start: Pipeline Integration & Testing

**Fast-track guide to running and validating the multi-stream pipeline.**

---

## 🚀 Quick Setup (5 minutes)

### 1. Generate Test Audio

```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests
python3 generate_test_audio.py
```

**Output**: 6 test audio files in `tests/audio/`

### 2. Install Test Dependencies

```bash
pip install pytest pytest-asyncio aiohttp numpy
```

### 3. Start Service (Pipeline Mode)

```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis

ENABLE_PIPELINE=true \
  python -m uvicorn xdna2.server:app --host 0.0.0.0 --port 9050
```

**Verify**:
```bash
curl http://localhost:9050/ | jq .mode
# Expected: "pipeline"
```

---

## 🧪 Quick Tests (10 minutes)

### Integration Tests

```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests
pytest test_pipeline_integration.py -v
```

**Expected**: 8 tests pass, ~30 seconds

### Load Test (Quick)

```bash
python load_test_pipeline.py --quick
```

**Expected**: Throughput results for 5 concurrency levels, ~60 seconds

### Accuracy Validation

```bash
python validate_accuracy.py
```

**Expected**: >99% similarity, consistency verified

---

## 📊 Monitor Pipeline

### Check Pipeline Status

```bash
# Pipeline health
curl http://localhost:9050/health/pipeline | jq

# Pipeline statistics
curl http://localhost:9050/stats/pipeline | jq
```

### Monitor NPU (Background)

```bash
# In separate terminal
python monitor_npu_utilization.py --duration 120 --output npu_stats.csv
```

**Then run load test**:
```bash
python load_test_pipeline.py
```

---

## 🎯 Performance Validation

### Sequential Mode Baseline

```bash
# 1. Start in sequential mode
ENABLE_PIPELINE=false \
  python -m uvicorn xdna2.server:app --port 9050

# 2. Run load test (single concurrency)
cd tests
python load_test_pipeline.py --concurrency 1 --duration 60

# 3. Record baseline throughput (~15.6 req/s)
```

### Pipeline Mode Target

```bash
# 1. Restart in pipeline mode
ENABLE_PIPELINE=true \
  python -m uvicorn xdna2.server:app --port 9050

# 2. Run full load test suite
cd tests
python load_test_pipeline.py

# 3. Verify improvement
# Target: 67 req/s (+329%)
```

---

## 📈 Quick Results

### Expected Performance (Pipeline Mode)

| Concurrency | Throughput | Mean Latency | P95 Latency |
|-------------|------------|--------------|-------------|
| 1           | 15.6 req/s | 64ms         | 70ms        |
| 5           | 42 req/s   | 119ms        | 135ms       |
| 10          | 58 req/s   | 172ms        | 195ms       |
| 15          | **67 req/s** | 224ms      | 250ms       |
| 20          | 67 req/s   | 299ms        | 330ms       |

**Improvement**: +329% over sequential (15.6 → 67 req/s)

**NPU Utilization**: 15% (vs 0.12% sequential, +1775%)

---

## 🔧 Troubleshooting

### Service Not Running

```bash
# Check if service is up
curl http://localhost:9050/health

# If not, restart
pkill -f uvicorn
ENABLE_PIPELINE=true python -m uvicorn xdna2.server:app --port 9050
```

### Tests Fail: Audio Not Found

```bash
# Regenerate test audio
cd tests
python generate_test_audio.py

# Verify files exist
ls -lh audio/
```

### Low Throughput

```bash
# Verify pipeline mode enabled
curl http://localhost:9050/ | jq .mode
# Should be "pipeline", not "sequential"

# Check pipeline health
curl http://localhost:9050/health/pipeline | jq .healthy
# Should be true

# Check pipeline stats
curl http://localhost:9050/stats/pipeline | jq .throughput_rps
```

### Tests Timeout

```bash
# Increase timeout in test
pytest test_pipeline_integration.py -v --timeout=60

# Or for load tests
python load_test_pipeline.py --duration 10  # Shorter test
```

---

## 📝 Key Files

**Service**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/server.py`
**Tests**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests/`

**Test Audio**: `tests/audio/test_audio.wav` (default)

**Scripts**:
- `test_pipeline_integration.py` - Integration tests (pytest)
- `load_test_pipeline.py` - Load testing suite
- `validate_accuracy.py` - Accuracy validation
- `monitor_npu_utilization.py` - NPU monitoring
- `generate_test_audio.py` - Test audio generation

---

## 🎓 Full Documentation

**Comprehensive Guide**: `WEEK10_INTEGRATION_REPORT.md`
**Testing Guide**: `tests/README.md`
**Week 9 Implementation**: `WEEK9_MULTI_STREAM_IMPLEMENTATION_REPORT.md`

---

## ✅ Success Checklist

- [ ] Test audio generated (6 files)
- [ ] Service starts in pipeline mode
- [ ] Integration tests pass (8/8)
- [ ] Load test shows throughput >30 req/s
- [ ] Accuracy validation >99% similarity
- [ ] Pipeline health endpoint returns healthy
- [ ] NPU utilization monitored

**Minimum Success**: 30+ req/s (+92%)
**Target Success**: 67 req/s (+329%)

---

**Questions?** See `WEEK10_INTEGRATION_REPORT.md` or `tests/README.md`

**Status**: Week 10 Integration Complete ✅
**Next**: NPU hardware validation (Week 11)

**Built with 🦄 by Magic Unicorn Unconventional Technology & Stuff Inc**
