# Unicorn-Amanuensis Deployment Guide

**Service**: Speech-to-Text with NPU Acceleration
**Version**: 3.0.0 (Multi-Stream Pipeline + C++ NPU Encoder)
**Target Performance**: 400-500x realtime
**Last Updated**: November 2, 2025

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Performance Tuning](#performance-tuning)
7. [Operations](#operations)

---

## Prerequisites

### Hardware Requirements

**Minimum**:
- AMD Strix Halo APU (Ryzen AI MAX+ 395 or similar)
- XDNA2 NPU (50 TOPS)
- 16GB RAM
- 10GB disk space

**Recommended**:
- 32GB+ RAM (for multi-stream processing)
- 20GB+ disk space (for models and logs)
- NVMe SSD (for fast model loading)

### Software Requirements

**Operating System**:
- Ubuntu 24.10+ or 25.10 (Oracular Oriole)
- Linux kernel 6.17.0+ (for XDNA2 support)

**Core Dependencies**:
```bash
# AMD XRT Runtime
Version: 2.21.0+
Source: /opt/xilinx/xrt/

# MLIR-AIE Toolchain
Version: 20.0.0+
Python API: aie.utils.xrt (setup_aie, execute)
Location: ~/mlir-aie/

# Python
Version: 3.11+ (3.13 recommended)
Packages: numpy, transformers, whisperx, fastapi, uvicorn

# C++ Runtime
Compiler: GCC 11.4.0+
Libraries: libstdc++, libgomp (for OpenMP)
```

**Python Packages**:
```txt
# Core
numpy>=1.26.0
torch>=2.0.0
transformers>=4.30.0

# Service
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6

# Audio Processing
whisperx>=3.0.0
librosa>=0.10.0
soundfile>=0.12.0

# XRT (installed separately)
pyxrt (from XRT package)
```

### System Setup

**1. Install XRT**:
```bash
# Download XRT package for Ubuntu
wget https://www.xilinx.com/bin/public/openDownload?filename=xrt_202410.2.18.0_25.10-amd64-xrt.deb

# Install XRT
sudo apt install ./xrt_202410.2.18.0_25.10-amd64-xrt.deb

# Verify installation
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
# Expected: Device 0 (Phoenix/Hawk Point XDNA Device) detected

# Test pyxrt
python3 -c "import pyxrt; print('pyxrt OK')"
```

**2. Install MLIR-AIE** (for development/compilation only):
```bash
# Clone MLIR-AIE repository
cd ~
git clone https://github.com/Xilinx/mlir-aie.git

# Install in virtual environment
cd mlir-aie
python3 -m venv ironenv
source ironenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "from aie.utils.xrt import setup_aie; print('MLIR-AIE OK')"
```

**Note**: For runtime (service deployment), MLIR-AIE is NOT required. Only XRT is needed.

**3. Install Python Dependencies**:
```bash
# Use system Python3 (recommended for runtime)
pip3 install numpy torch transformers whisperx fastapi uvicorn python-multipart librosa soundfile

# OR: Create service virtual environment
python3 -m venv /opt/unicorn-amanuensis/venv
source /opt/unicorn-amanuensis/venv/bin/activate
pip install -r requirements.txt
```

**4. Build C++ Encoder** (if not pre-built):
```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp

# Build with CMake
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Verify library
ls -lh libwhisper_encoder_cpp.so
# Expected: ~500KB shared library

# Test library
python3 -c "import ctypes; lib = ctypes.CDLL('./libwhisper_encoder_cpp.so'); print('C++ encoder OK')"
```

---

## Installation

### Method 1: System-Wide Installation (Recommended for Production)

```bash
# 1. Clone repository
git clone https://github.com/CognitiveCompanion/CC-1L.git /opt/CC-1L
cd /opt/CC-1L/npu-services/unicorn-amanuensis

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Download Whisper model (optional - auto-downloads on first run)
python3 -c "from transformers import WhisperModel; WhisperModel.from_pretrained('openai/whisper-base')"

# 4. Create systemd service
sudo cp deployment/unicorn-amanuensis.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable unicorn-amanuensis
sudo systemctl start unicorn-amanuensis

# 5. Verify service
sudo systemctl status unicorn-amanuensis
curl http://127.0.0.1:9050/health
```

### Method 2: Development Installation

```bash
# 1. Clone repository
git clone https://github.com/CognitiveCompanion/CC-1L.git ~/CC-1L
cd ~/CC-1L/npu-services/unicorn-amanuensis

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run in development mode
source /opt/xilinx/xrt/setup.sh
export ENABLE_PIPELINE=true
uvicorn xdna2.server:app --host 127.0.0.1 --port 9050 --reload
```

### Method 3: Docker Deployment

```bash
# 1. Build Docker image
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis
docker build -t unicorn-amanuensis:latest .

# 2. Run container (with NPU access)
docker run -d \
  --name unicorn-amanuensis \
  --device /dev/dri:/dev/dri \
  -v /opt/xilinx/xrt:/opt/xilinx/xrt:ro \
  -p 9050:9050 \
  -e ENABLE_PIPELINE=true \
  -e REQUIRE_NPU=false \
  -e ALLOW_FALLBACK=true \
  -e FALLBACK_DEVICE=cpu \
  unicorn-amanuensis:latest

# 3. Check logs
docker logs -f unicorn-amanuensis

# 4. Test service
curl http://127.0.0.1:9050/health
```

---

## Configuration

### Environment Variables

**Core Settings**:
```bash
# Model Configuration
export WHISPER_MODEL=base           # base, small, medium, large
export COMPUTE_TYPE=int8            # int8, float16, float32
export BATCH_SIZE=16                # Decoder batch size

# Pipeline Configuration
export ENABLE_PIPELINE=true         # Enable multi-stream processing
export NUM_LOAD_WORKERS=4           # Audio loading workers
export NUM_DECODER_WORKERS=4        # Decoder workers
export MAX_QUEUE_SIZE=100           # Request queue size

# NPU Configuration
export REQUIRE_NPU=false            # Fail if NPU unavailable
export ALLOW_FALLBACK=false         # Allow CPU/GPU fallback (default: false)
export FALLBACK_DEVICE=none         # none, igpu, cpu
```

**NPU Fallback Behavior**:

| REQUIRE_NPU | ALLOW_FALLBACK | FALLBACK_DEVICE | Behavior |
|-------------|----------------|-----------------|----------|
| false | false | none | ✅ Start service, fail if NPU unavailable |
| false | true | cpu | ✅ Start service, use CPU if NPU unavailable |
| true | false | none | ❌ Fail if NPU unavailable |
| true | true | ignored | ❌ Fail if NPU unavailable (REQUIRE takes priority) |

**Recommended Configurations**:

**Development** (permissive):
```bash
export REQUIRE_NPU=false
export ALLOW_FALLBACK=true
export FALLBACK_DEVICE=cpu
```

**Production** (strict):
```bash
export REQUIRE_NPU=false       # Don't hard-fail on NPU missing
export ALLOW_FALLBACK=false    # But don't silently degrade to CPU
export FALLBACK_DEVICE=none    # Force explicit configuration
```

**User Preference** ("I really don't want CPU fallback"):
```bash
export REQUIRE_NPU=false       # Service can start without NPU
export ALLOW_FALLBACK=false    # But will fail if NPU not available
export FALLBACK_DEVICE=none    # No silent degradation
```

### Service Configuration File

Create `/etc/unicorn-amanuensis/config.env`:
```bash
# Model
WHISPER_MODEL=base
COMPUTE_TYPE=int8
BATCH_SIZE=16

# Pipeline
ENABLE_PIPELINE=true
NUM_LOAD_WORKERS=4
NUM_DECODER_WORKERS=4
MAX_QUEUE_SIZE=100

# NPU (User preference: no silent fallback)
REQUIRE_NPU=false
ALLOW_FALLBACK=false
FALLBACK_DEVICE=none

# Buffer Pools
MEL_BUFFER_COUNT=10
AUDIO_BUFFER_COUNT=5
ENCODER_BUFFER_COUNT=5

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/unicorn-amanuensis/service.log
```

### Systemd Service File

`/etc/systemd/system/unicorn-amanuensis.service`:
```ini
[Unit]
Description=Unicorn-Amanuensis Speech-to-Text Service
After=network.target

[Service]
Type=simple
User=unicorn
Group=unicorn
WorkingDirectory=/opt/CC-1L/npu-services/unicorn-amanuensis
EnvironmentFile=/etc/unicorn-amanuensis/config.env
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="LD_LIBRARY_PATH=/opt/xilinx/xrt/lib"
Environment="PYTHONPATH=/opt/xilinx/xrt/python"
ExecStartPre=/bin/bash -c 'source /opt/xilinx/xrt/setup.sh'
ExecStart=/usr/bin/python3 -m uvicorn xdna2.server:app --host 127.0.0.1 --port 9050
Restart=always
RestartSec=10

# Resource Limits
MemoryMax=8G
CPUQuota=400%

# Logging
StandardOutput=append:/var/log/unicorn-amanuensis/stdout.log
StandardError=append:/var/log/unicorn-amanuensis/stderr.log

[Install]
WantedBy=multi-user.target
```

---

## Verification

### 1. Service Health Check

```bash
# Check if service is running
curl http://127.0.0.1:9050/health

# Expected response:
{
  "status": "healthy",
  "uptime_seconds": 123.45,
  "encoder": {
    "type": "C++ with NPU",
    "runtime_version": "1.0.0",
    "num_layers": 6,
    "npu_enabled": true,           # ← Should be true
    "weights_loaded": true
  },
  "buffer_pools": {
    "mel": {"hit_rate": 0.95, ...},
    "audio": {"hit_rate": 0.92, ...},
    "encoder_output": {"hit_rate": 0.88, ...}
  }
}
```

### 2. NPU Verification

**Check XRT Device**:
```bash
source /opt/xilinx/xrt/setup.sh
xrt-smi examine

# Expected output:
# Device 0
#   Name: Phoenix/Hawk Point XDNA Device
#   Status: Operational
```

**Check Service Logs** (look for NPU initialization):
```bash
# Systemd logs
sudo journalctl -u unicorn-amanuensis -f

# Or log file
tail -f /var/log/unicorn-amanuensis/stdout.log

# Expected log entries:
# INFO: [Init] Loading XRT NPU application...
# INFO:   Found xclbin: matmul_1tile_bf16.xclbin
# INFO:   XRT device opened
# INFO:   xclbin registered successfully
# INFO:   Hardware context created
# INFO:   Kernel loaded: MLIR_AIE
# INFO: ✅ NPU callback registered successfully
```

### 3. Functional Test

**Test Transcription**:
```bash
# Create test audio (10 seconds of silence with tone)
ffmpeg -f lavfi -i "sine=frequency=440:duration=10" -ar 16000 test_10s.wav

# Transcribe
curl -X POST http://127.0.0.1:9050/v1/audio/transcriptions \
  -F "file=@test_10s.wav" \
  -F "model=whisper-1" \
  | jq .

# Expected response:
{
  "text": "...",                    # Transcribed text
  "performance": {
    "audio_duration_s": 10.0,
    "processing_time_s": 0.025,     # ~25ms processing
    "realtime_factor": 400.0,       # 400x realtime!
    "encoder_time_ms": 13.2,        # NPU encoder time
    "decoder_time_ms": 11.8
  }
}
```

### 4. Performance Verification

**Run Smoke Test**:
```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis
source /opt/xilinx/xrt/setup.sh

# NOTE: This test should be updated to use setup_aie() before use!
# See WEEK17_PRODUCTION_READINESS_ASSESSMENT.md for details
python3 WEEK15_NPU_SIMPLE_TEST.py

# Expected output:
# ✓ PASS
# Execution: 1.22ms
# Performance: 219.4 GFLOPS
# Accuracy: 3.74% mean error (96.26% accuracy)
```

**Run Integration Tests**:
```bash
# NOTE: Fix pyxrt import issue first!
# See WEEK17_PRODUCTION_READINESS_ASSESSMENT.md
python3 WEEK16_INTEGRATION_TEST.py

# Expected output:
# ✓ Test 1: XRT Buffer Allocation - PASSED
# ✓ Test 2: XRTApp Class Integration - PASSED
# ✓ Test 3: Encoder NPU Callback - PASSED
# Total: 3/3 tests passed
```

### 5. Load Test

**Concurrent Requests**:
```bash
# Install Apache Bench
sudo apt install apache2-utils

# 100 requests, 10 concurrent
ab -n 100 -c 10 -p test_10s.wav \
  -T "multipart/form-data; boundary=----WebKitFormBoundary" \
  http://127.0.0.1:9050/v1/audio/transcriptions

# Check for:
# - No failures (Failed requests: 0)
# - Consistent latency (~25-50ms per request)
# - High throughput (>20 req/s)
```

---

## Troubleshooting

### Common Issues

#### Issue 1: NPU Returns All Zeros

**Symptoms**:
```
[9/9] Validating results...
  Result: 512x512 (range: 0.000 to 0.000)
  ✗ FAIL - All zeros returned
```

**Root Cause**: Instruction buffer not loaded

**Solution**:
```bash
# This is BLOCKER #1 in production readiness assessment
# See WEEK17_PRODUCTION_READINESS_ASSESSMENT.md for fix

# Short version: Update load_xrt_npu_application() to use setup_aie()
# Reference: WEEK16_NPU_SOLUTION.py
```

**Verification**:
```bash
# After fix, run smoke test
python3 WEEK15_NPU_SIMPLE_TEST.py
# Expected: 96.26% accuracy (not 0%)
```

#### Issue 2: pyxrt Import Error

**Symptoms**:
```
ImportError: undefined symbol: _ZNK3xrt6module12get_cfg_uuidEv
```

**Root Cause**: pyxrt version mismatch between environments

**Solution**:
```bash
# Option A: Use system Python3 (RECOMMENDED)
# Don't use ironenv for service runtime
source /opt/xilinx/xrt/setup.sh
python3 -m uvicorn xdna2.server:app --host 127.0.0.1 --port 9050

# Option B: Reinstall pyxrt in environment
source ~/mlir-aie/ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh
pip uninstall pyxrt
# pyxrt will be available via PYTHONPATH from XRT
```

**Verification**:
```bash
python3 -c "import pyxrt; print('pyxrt OK')"
```

#### Issue 3: Service Fails to Start (NPU Not Found)

**Symptoms**:
```
❌ CRITICAL: NPU required but xclbin not found
  Set REQUIRE_NPU=false to allow fallback
```

**Root Cause**: xclbin file not in expected location

**Solution**:
```bash
# Check xclbin file exists
ls -lh /home/ccadmin/CC-1L/kernels/common/build_bf16_1tile/matmul_1tile_bf16.xclbin

# If missing, compile kernel:
cd /home/ccadmin/CC-1L/kernels/common
source ~/mlir-aie/ironenv/bin/activate
make build_bf16_1tile

# Or: Disable NPU requirement
export REQUIRE_NPU=false
export ALLOW_FALLBACK=true
export FALLBACK_DEVICE=cpu
```

#### Issue 4: Poor Performance (<100x Realtime)

**Symptoms**:
```json
{
  "performance": {
    "realtime_factor": 50.0,  // Way below 400x target
    "encoder_time_ms": 200.0  // Way above 13ms target
  }
}
```

**Root Cause**: NPU not being used (CPU fallback)

**Diagnosis**:
```bash
# Check service logs
sudo journalctl -u unicorn-amanuensis | grep NPU

# Look for:
# INFO: ✅ NPU callback registered successfully  ← Good
# WARNING: Falling back to CPU mode             ← Bad

# Check health endpoint
curl http://127.0.0.1:9050/health | jq .encoder.npu_enabled
# Should be: true
```

**Solution**:
```bash
# Fix NPU initialization (see Issue 1)
# Verify xclbin path is correct
# Check XRT device is available (xrt-smi examine)
```

#### Issue 5: Memory Leak / High Memory Usage

**Symptoms**:
```bash
# Service memory grows over time
ps aux | grep unicorn
# RSS column keeps increasing (>8GB)
```

**Diagnosis**:
```bash
# Check buffer pool stats
curl http://127.0.0.1:9050/health | jq .buffer_pools

# Look for:
# "hit_rate": 0.95  ← Good (reusing buffers)
# "hit_rate": 0.10  ← Bad (allocating new buffers)
```

**Solution**:
```bash
# Increase buffer pool sizes
export MEL_BUFFER_COUNT=20         # Default: 10
export AUDIO_BUFFER_COUNT=10       # Default: 5
export ENCODER_BUFFER_COUNT=10     # Default: 5

# Restart service
sudo systemctl restart unicorn-amanuensis
```

#### Issue 6: Service Crashes Under Load

**Symptoms**:
```bash
# Service exits unexpectedly
systemctl status unicorn-amanuensis
# Status: failed (code=exited)
```

**Diagnosis**:
```bash
# Check logs for errors
sudo journalctl -u unicorn-amanuensis -n 100 --no-pager

# Look for:
# - Segmentation fault (C++ encoder crash)
# - Out of memory errors
# - NPU timeout errors
```

**Solution**:
```bash
# Increase resource limits in systemd service
sudo nano /etc/systemd/system/unicorn-amanuensis.service

# Update:
MemoryMax=16G           # Increase from 8G
CPUQuota=800%           # Increase from 400%
Restart=always          # Always restart on crash
RestartSec=5            # Wait 5s before restart

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart unicorn-amanuensis
```

---

## Performance Tuning

### NPU Optimization

**1. Use Appropriate Kernel**:
```bash
# Single-tile (fastest load, good for development)
export XCLBIN_PATH=/home/ccadmin/CC-1L/kernels/common/build_bf16_1tile/matmul_1tile_bf16.xclbin

# Multi-tile (better performance, more GFLOPS)
export XCLBIN_PATH=/home/ccadmin/CC-1L/kernels/common/build_bf16_2tile_FIXED/matmul_2tile_bf16_xdna2_FIXED.xclbin
```

**2. Buffer Pool Sizing**:
```bash
# For high-concurrency workloads
export MEL_BUFFER_COUNT=20
export AUDIO_BUFFER_COUNT=15
export ENCODER_BUFFER_COUNT=15
export MAX_QUEUE_SIZE=200

# Trade-off: Higher memory usage for fewer allocations
```

**3. Pipeline Configuration**:
```bash
# More workers = higher throughput (up to CPU core count)
export NUM_LOAD_WORKERS=8          # Audio loading
export NUM_DECODER_WORKERS=8       # Decoder processing

# More workers = higher memory usage
# Recommendation: cores/2 for each worker pool
```

### CPU/GPU Fallback Optimization

**If NPU Unavailable**:
```bash
# Use GPU for decoder (faster than CPU)
export DEVICE=cuda
export COMPUTE_TYPE=float16

# Use CPU (lower memory)
export DEVICE=cpu
export COMPUTE_TYPE=int8
export BATCH_SIZE=32               # Larger batch for CPU efficiency
```

### Memory Optimization

**Reduce Memory Footprint**:
```bash
# Smaller model
export WHISPER_MODEL=tiny          # ~150MB vs base ~500MB

# Lower buffer counts
export MEL_BUFFER_COUNT=5
export AUDIO_BUFFER_COUNT=3
export ENCODER_BUFFER_COUNT=3

# Smaller queue
export MAX_QUEUE_SIZE=50
```

### Throughput Optimization

**Maximize Requests/Second**:
```bash
# Enable pipeline
export ENABLE_PIPELINE=true

# More workers
export NUM_LOAD_WORKERS=8
export NUM_DECODER_WORKERS=8

# Larger queue
export MAX_QUEUE_SIZE=200

# Expected: 67 req/s (vs 15.6 req/s sequential)
```

---

## Operations

### Starting the Service

**Systemd** (recommended):
```bash
sudo systemctl start unicorn-amanuensis
sudo systemctl status unicorn-amanuensis
```

**Manual** (development):
```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis
source /opt/xilinx/xrt/setup.sh
export ENABLE_PIPELINE=true
python3 -m uvicorn xdna2.server:app --host 127.0.0.1 --port 9050
```

**Docker**:
```bash
docker start unicorn-amanuensis
docker logs -f unicorn-amanuensis
```

### Stopping the Service

**Graceful Shutdown**:
```bash
# Systemd
sudo systemctl stop unicorn-amanuensis

# Docker
docker stop unicorn-amanuensis

# Manual (Ctrl+C in terminal)
```

### Monitoring

**Service Health**:
```bash
# Health check
curl http://127.0.0.1:9050/health | jq .

# Check NPU status
curl http://127.0.0.1:9050/health | jq .encoder.npu_enabled

# Check buffer pool efficiency
curl http://127.0.0.1:9050/health | jq .buffer_pools
```

**System Metrics**:
```bash
# CPU and memory usage
htop

# NPU usage (xrt-smi)
source /opt/xilinx/xrt/setup.sh
watch -n 1 xrt-smi examine

# Service logs
sudo journalctl -u unicorn-amanuensis -f
```

### Backup and Recovery

**Backup Configuration**:
```bash
# Backup config
sudo cp /etc/unicorn-amanuensis/config.env /backup/config.env.$(date +%Y%m%d)

# Backup systemd service
sudo cp /etc/systemd/system/unicorn-amanuensis.service /backup/
```

**Restore from Backup**:
```bash
# Restore config
sudo cp /backup/config.env.20251102 /etc/unicorn-amanuensis/config.env

# Restore service
sudo cp /backup/unicorn-amanuensis.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart unicorn-amanuensis
```

### Scaling Considerations

**Vertical Scaling** (more resources):
- Increase worker counts (NUM_LOAD_WORKERS, NUM_DECODER_WORKERS)
- Increase buffer pool sizes
- Increase memory limits in systemd

**Horizontal Scaling** (multiple instances):
- Run multiple service instances on different ports
- Use load balancer (nginx, HAProxy) to distribute requests
- Each instance needs access to NPU (if using NPU acceleration)

---

## Security Considerations

1. **Network Exposure**: Service binds to 127.0.0.1 (localhost only) by default
2. **File Upload**: Audio files are written to /tmp (ensure proper permissions)
3. **Resource Limits**: Use systemd MemoryMax/CPUQuota to prevent resource exhaustion
4. **NPU Access**: /dev/dri device requires proper permissions (video group)

---

## Support and Resources

**Documentation**:
- Production Readiness Assessment: `WEEK17_PRODUCTION_READINESS_ASSESSMENT.md`
- Architecture Overview: `docs/architecture/OVERVIEW.md`
- XRTApp Quick Reference: `XRTAPP_QUICK_REFERENCE.md`
- Week 16 Breakthrough Report: `WEEK16_BREAKTHROUGH_REPORT.md`

**Issue Tracker**:
- GitHub Issues: https://github.com/CognitiveCompanion/CC-1L/issues

**Contact**:
- Project Lead: Aaron Stransky (aaron@magicunicorn.tech)
- Company: Magic Unicorn Unconventional Technology & Stuff Inc

---

**Last Updated**: November 2, 2025
**Version**: 1.0.0 (Pre-Release - Blockers Pending)

**Built with 🦄 by Magic Unicorn Unconventional Technology & Stuff Inc**
