# Multi-Stream Pipeline Testing Suite

Comprehensive testing infrastructure for Week 10 pipeline integration and performance validation.

## Overview

This test suite validates the multi-stream pipeline implementation for Unicorn-Amanuensis, ensuring:

- **Performance**: +329% throughput improvement (15.6 → 67 req/s)
- **Accuracy**: Identical outputs to sequential mode (>99% similarity)
- **Reliability**: Stable operation under sustained load
- **NPU Utilization**: 15% NPU utilization (+1775% improvement)

## Test Files

### Integration Tests

**File**: `test_pipeline_integration.py`

Pytest-based integration tests for pipeline functionality:

- Service health checks
- Pipeline mode verification
- Single request processing
- Concurrent request handling (5, 15 requests)
- Pipeline statistics endpoints
- Health monitoring endpoints
- Sequential consistency validation

**Usage**:
```bash
# Run all integration tests
pytest test_pipeline_integration.py -v

# Run specific test
pytest test_pipeline_integration.py::test_pipeline_concurrent_requests -v

# Run with coverage
pytest test_pipeline_integration.py --cov=transcription_pipeline --cov-report=html
```

**Requirements**:
- Service running on `localhost:9050`
- `ENABLE_PIPELINE=true` environment variable
- Test audio files in `tests/audio/` directory

### Load Testing

**File**: `load_test_pipeline.py`

Comprehensive load testing script with variable concurrency:

- Concurrency levels: 1, 5, 10, 15, 20 concurrent requests
- Sustained load testing (30-60 seconds)
- Throughput measurement (requests per second)
- Latency distribution (p50, p95, p99)
- Error rate tracking

**Usage**:
```bash
# Run full load test suite (5 concurrency levels, 30s each)
python load_test_pipeline.py

# Test specific concurrency level
python load_test_pipeline.py --concurrency 10 --duration 30

# Quick test (10 seconds)
python load_test_pipeline.py --quick

# Test with specific audio file
python load_test_pipeline.py --audio audio/test_5s.wav
```

**Performance Targets**:
- Sequential: 15.6 req/s (baseline)
- Pipeline: 67 req/s (+329% improvement)
- Individual latency: <70ms per request
- Error rate: <1%

### Accuracy Validation

**File**: `validate_accuracy.py`

Validates pipeline produces identical outputs to sequential mode:

- Text exact matching
- Text similarity (sequence matcher)
- Edit distance (Levenshtein)
- Segment count comparison
- Word count comparison
- Request mixing detection

**Usage**:
```bash
# Validate accuracy with default audio
python validate_accuracy.py

# Validate with specific audio
python validate_accuracy.py --audio audio/test_10s.wav

# Strict mode (fail on any difference)
python validate_accuracy.py --strict
```

**Success Criteria**:
- Text similarity: >99%
- Segment count: Exact match
- Word count: Exact match
- No request mixing

### NPU Utilization Monitoring

**File**: `monitor_npu_utilization.py`

Monitors NPU utilization during load testing:

- Real-time NPU usage sampling
- Power consumption tracking
- Temperature monitoring
- Statistical analysis (mean, max, min)

**Usage**:
```bash
# Monitor NPU for 60 seconds
python monitor_npu_utilization.py --duration 60

# Custom sampling interval
python monitor_npu_utilization.py --duration 60 --interval 0.5

# Export to CSV
python monitor_npu_utilization.py --duration 60 --output npu_stats.csv
```

**Performance Target**:
- Sequential: 0.12% NPU utilization
- Pipeline: 15% NPU utilization (+1775%)

**Monitoring Methods** (auto-detected):
1. `xrt-smi` (AMD XRT tools)
2. `sysfs` (/sys/class/accel/accel*)
3. `simulation` (fallback for testing)

### Test Audio Generation

**File**: `generate_test_audio.py`

Generates synthetic audio files for testing:

**Generated Files**:
- `test_audio.wav`: 10s speech-like (default)
- `test_1s.wav`: 1 second
- `test_5s.wav`: 5 seconds
- `test_30s.wav`: 30 seconds
- `test_silence.wav`: 5s silence
- `test_tone.wav`: 5s 440Hz sine wave

**Usage**:
```bash
# Generate all test audio files
python generate_test_audio.py

# Generate specific duration
python generate_test_audio.py --duration 15

# Custom output directory
python generate_test_audio.py --output /path/to/audio
```

**Audio Specifications**:
- Sample rate: 16kHz (Whisper standard)
- Format: WAV (16-bit PCM mono)
- Content: Speech-like (multiple harmonics + noise)

## Test Audio Directory

**Location**: `tests/audio/`

**Contents**:
```
audio/
├── test_audio.wav       # Default 10s test audio
├── test_1s.wav          # 1 second test
├── test_5s.wav          # 5 second test
├── test_30s.wav         # 30 second test
├── test_silence.wav     # 5s silence
└── test_tone.wav        # 5s 440Hz tone
```

## Testing Workflow

### 1. Setup

```bash
# Navigate to tests directory
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests

# Generate test audio (if not already done)
python generate_test_audio.py

# Install test dependencies
pip install pytest pytest-asyncio aiohttp numpy
```

### 2. Start Service

**Sequential Mode** (baseline):
```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis
ENABLE_PIPELINE=false python -m uvicorn xdna2.server:app --host 0.0.0.0 --port 9050
```

**Pipeline Mode** (target):
```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis
ENABLE_PIPELINE=true python -m uvicorn xdna2.server:app --host 0.0.0.0 --port 9050
```

### 3. Run Tests

**Integration Tests**:
```bash
pytest test_pipeline_integration.py -v
```

**Load Tests** (run in separate terminal from NPU monitoring):
```bash
# Terminal 1: NPU monitoring
python monitor_npu_utilization.py --duration 180 --output npu_stats.csv

# Terminal 2: Load testing
python load_test_pipeline.py
```

**Accuracy Validation**:
```bash
# Test pipeline mode
ENABLE_PIPELINE=true python validate_accuracy.py

# Compare with sequential (requires service restart)
ENABLE_PIPELINE=false python validate_accuracy.py
```

### 4. Compare Results

**Sequential vs Pipeline Comparison**:

1. Run load test with `ENABLE_PIPELINE=false`
2. Record baseline throughput (~15.6 req/s)
3. Restart service with `ENABLE_PIPELINE=true`
4. Run load test again
5. Calculate improvement: `(pipeline - sequential) / sequential * 100`
6. Verify: Improvement >= 329%

## Performance Targets

### Minimum Success (60% Week 9 Goals)

- ✅ Endpoint integration complete
- ✅ Pipeline processes requests successfully
- ✅ Throughput: 30+ req/s (+92% minimum)
- ✅ Accuracy: Outputs match sequential (>99% similarity)
- ✅ No crashes or deadlocks under load

### Stretch Goals (100% Week 9 Goals)

- ✅ Throughput: 67 req/s (+329% full target)
- ✅ NPU utilization: 15%
- ✅ Latency: Individual requests <70ms
- ✅ Handles 20+ concurrent requests
- ✅ Zero memory leaks (sustained load)

## Expected Results

### Sequential Mode (Baseline)

```
Concurrency    Throughput      Mean Latency    P95 Latency
-----------    ----------      ------------    -----------
1              15.6 req/s      64ms            70ms
5              15.6 req/s      320ms           350ms
10             15.6 req/s      640ms           700ms
```

### Pipeline Mode (Target)

```
Concurrency    Throughput      Mean Latency    P95 Latency
-----------    ----------      ------------    -----------
1              15.6 req/s      64ms            70ms
5              40 req/s        125ms           140ms
10             60 req/s        167ms           180ms
15             67 req/s        224ms           250ms
20             67 req/s        299ms           330ms
```

**Key Improvements**:
- Throughput: 15.6 → 67 req/s (+329%)
- NPU Utilization: 0.12% → 15% (+1775%)
- Handles 15-20 concurrent requests smoothly

## Troubleshooting

### Service Not Responding

```bash
# Check service is running
curl http://localhost:9050/health

# Check logs
tail -f /var/log/unicorn-amanuensis.log

# Restart service
pkill -f "uvicorn xdna2.server"
ENABLE_PIPELINE=true python -m uvicorn xdna2.server:app --host 0.0.0.0 --port 9050
```

### Tests Failing

**Issue**: `Test audio not found`
```bash
# Generate test audio
python generate_test_audio.py
```

**Issue**: `Connection refused`
```bash
# Verify service is running
curl http://localhost:9050/

# Check correct port
netstat -tuln | grep 9050
```

**Issue**: `Pipeline not enabled`
```bash
# Verify environment variable
curl http://localhost:9050/ | jq .mode

# Restart with pipeline enabled
ENABLE_PIPELINE=true python -m uvicorn xdna2.server:app --host 0.0.0.0 --port 9050
```

### Low Throughput

**Issue**: Throughput below target

Potential causes:
1. **NPU not utilized**: Check NPU monitoring
2. **Wrong mode**: Verify `ENABLE_PIPELINE=true`
3. **Resource bottleneck**: Check CPU, memory usage
4. **Network latency**: Test on localhost
5. **Test audio too large**: Use shorter test files

```bash
# Check service mode
curl http://localhost:9050/stats/pipeline

# Monitor NPU
python monitor_npu_utilization.py --duration 30

# Check system resources
htop
```

### Accuracy Issues

**Issue**: Pipeline produces different results

```bash
# Run accuracy validation
python validate_accuracy.py --strict

# Check for request mixing
# (different audio should produce different results)

# Verify same audio produces same results
pytest test_pipeline_integration.py::test_pipeline_sequential_consistency -v
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Pipeline Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install pytest pytest-asyncio aiohttp numpy

    - name: Generate test audio
      run: |
        cd tests
        python generate_test_audio.py

    - name: Start service
      run: |
        ENABLE_PIPELINE=true python -m uvicorn xdna2.server:app --port 9050 &
        sleep 10  # Wait for service to start

    - name: Run integration tests
      run: |
        cd tests
        pytest test_pipeline_integration.py -v

    - name: Run load tests (quick)
      run: |
        cd tests
        python load_test_pipeline.py --quick

    - name: Validate accuracy
      run: |
        cd tests
        python validate_accuracy.py
```

## Performance Benchmarking

### Reproducible Benchmark

```bash
#!/bin/bash
# benchmark.sh - Reproducible performance benchmark

echo "=== Multi-Stream Pipeline Benchmark ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Generate test audio
echo "Generating test audio..."
python generate_test_audio.py

# Start service in pipeline mode
echo "Starting service (pipeline mode)..."
ENABLE_PIPELINE=true python -m uvicorn xdna2.server:app --port 9050 &
SERVICE_PID=$!
sleep 10

# Run load test
echo "Running load test..."
python load_test_pipeline.py --duration 60 > results_pipeline.txt

# Stop service
kill $SERVICE_PID
sleep 5

# Start service in sequential mode
echo "Starting service (sequential mode)..."
ENABLE_PIPELINE=false python -m uvicorn xdna2.server:app --port 9050 &
SERVICE_PID=$!
sleep 10

# Run load test
echo "Running load test..."
python load_test_pipeline.py --duration 60 --concurrency 1 > results_sequential.txt

# Stop service
kill $SERVICE_PID

# Compare results
echo ""
echo "=== Results ==="
grep "Best Throughput" results_*.txt
echo ""
echo "Improvement: $(python -c "import re; p=float(re.search(r'(\d+\.\d+) req/s', open('results_pipeline.txt').read()).group(1)); s=float(re.search(r'(\d+\.\d+) req/s', open('results_sequential.txt').read()).group(1)); print(f'+{(p/s-1)*100:.0f}%')")"
```

## Documentation

- **Implementation**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/WEEK9_MULTI_STREAM_IMPLEMENTATION_REPORT.md`
- **Quick Start**: `/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/WEEK9_QUICK_START.md`
- **API Docs**: See service `/` endpoint for OpenAPI documentation

## Support

For issues or questions:
1. Check service logs
2. Verify test audio exists
3. Confirm service is running on correct port
4. Review error messages in test output

---

**Author**: CC-1L Multi-Stream Integration & Testing Team
**Date**: November 1, 2025
**Version**: 1.0.0
**Status**: Week 10 Integration Complete
