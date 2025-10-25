# Unicorn Amanuensis - NPU Installation Guide
## AMD Phoenix NPU with Custom MLIR-AIE2 Runtime

**Version**: 1.0.0
**Last Updated**: October 24, 2025

---

## 🎯 Quick Start

If you already have NPU drivers installed:

```bash
# Clone repository
git clone https://github.com/Unicorn-Commander/Unicorn-Amanuensis.git
cd Unicorn-Amanuensis/whisperx

# Start with Docker (recommended)
docker compose -f ../docker-compose-uc1-optimized.yml up -d unicorn-amanuensis

# Or run directly
pip install -r requirements_npu.txt
python3 server_npu_mlir.py
```

Access at: http://localhost:9000

---

## 📋 Prerequisites

### Hardware Requirements
- **CPU**: AMD Ryzen 7040 or 8040 series (Phoenix/Hawk Point)
- **NPU**: AMD XDNA (16 TOPS INT8)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for models and runtime

### Software Requirements
- **OS**: Ubuntu 22.04/24.04, Debian 12, or Fedora 38+
- **Kernel**: Linux 6.10+ (for amdxdna driver support)
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **Docker**: 24.0+ (for containerized deployment)

---

## 🔧 Installation Methods

### Method 1: Docker (Recommended)

**Pros**: Isolated environment, automatic dependencies, easy updates
**Cons**: Slightly larger disk usage

#### Step 1: Verify NPU Device

```bash
# Check if NPU device exists
ls -l /dev/accel/accel0

# Expected output:
# crw-rw---- 1 root render 511, 0 Oct 24 12:00 /dev/accel/accel0
```

If device doesn't exist, install drivers first (see below).

#### Step 2: Install Dependencies

```bash
# Install Docker (if not already installed)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# Install docker-compose
sudo apt install docker-compose-plugin
```

#### Step 3: Clone and Configure

```bash
# Clone repository
git clone https://github.com/Unicorn-Commander/Unicorn-Amanuensis.git
cd Unicorn-Amanuensis

# Create models directory
mkdir -p whisperx/models

# Download models (optional - will auto-download if missing)
cd whisperx
./download-models.sh
```

#### Step 4: Start Container

```bash
# Start Amanuensis with NPU support
docker compose -f docker-compose-uc1-optimized.yml up -d unicorn-amanuensis

# Check logs
docker logs -f unicorn-amanuensis

# Look for:
# ✅ NPU detected - using custom MLIR-AIE2 acceleration
# ✅ Loaded NPU model: base
```

#### Step 5: Test

```bash
# Health check
curl http://localhost:9000/health

# Should return:
# {
#   "status": "healthy",
#   "npu_enabled": true,
#   "npu_available": true,
#   "runtime": "Custom MLIR-AIE2"
# }

# Transcribe test audio
curl -X POST -F "file=@test.wav" http://localhost:9000/v1/audio/transcriptions
```

---

### Method 2: Native Installation

**Pros**: Best performance, direct hardware access
**Cons**: More complex setup, dependencies on host

#### Step 1: Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build tools
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev

# Create Python virtual environment
python3.11 -m venv ~/venv-npu
source ~/venv-npu/bin/activate
```

#### Step 2: Install NPU Drivers

```bash
# Install amdxdna kernel module
cd ~
git clone https://github.com/Unicorn-Commander/unicorn-npu-core.git
cd unicorn-npu-core

# Install using prebuilt packages (fastest - 40 seconds)
sudo bash scripts/install-npu-host-prebuilt.sh

# Or build from source (10-18 minutes)
sudo bash scripts/install-npu-host.sh

# Verify installation
ls -l /dev/accel/accel0
lsmod | grep amdxdna
```

#### Step 3: Install XRT (Xilinx Runtime)

```bash
# Using prebuilt packages from unicorn-npu-core
cd ~/unicorn-npu-core/releases/v1.0.0
sudo bash install-xrt-prebuilt.sh

# Or build from source
git clone https://github.com/AMD/xrt.git
cd xrt
./build.sh
sudo apt install ./build/Release/xrt_*.deb

# Verify XRT
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
```

#### Step 4: Install MLIR-AIE2 Tools (Optional but Recommended)

```bash
# Clone MLIR-AIE repository
cd ~
git clone --recursive https://github.com/Xilinx/mlir-aie.git
cd mlir-aie

# Build with NPU support
mkdir build && cd build
cmake .. \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DAIE_ENABLE_PYTHON_PASSES=ON \
    -DXRT_ROOT=/opt/xilinx/xrt \
    -DCMAKE_INSTALL_PREFIX=/usr/local
ninja
sudo ninja install

# Verify tools
aie-opt --version
aie-translate --version
```

#### Step 5: Install Unicorn Amanuensis

```bash
# Clone repository
cd ~
git clone https://github.com/Unicorn-Commander/Unicorn-Amanuensis.git
cd Unicorn-Amanuensis/whisperx

# Install Python dependencies
source ~/venv-npu/bin/activate
pip install -r requirements_npu.txt

# Download models
./download-models.sh

# Configure environment
export WHISPER_MODEL=base
export USE_NPU=true
export WHISPER_NPU_MODEL_PATH=/home/ucadmin/Unicorn-Amanuensis/whisperx/models/whisper-base-amd-npu-int8
export API_PORT=9000
```

#### Step 6: Run Server

```bash
# Start server
python3 server_npu_mlir.py

# Or with uvicorn directly
uvicorn server_npu_mlir:app --host 0.0.0.0 --port 9000 --reload
```

---

## 🚨 Driver Installation (If Needed)

### Install amdxdna Kernel Module

The amdxdna driver enables Linux kernel to communicate with AMD NPU hardware.

#### Option A: Use Unicorn NPU Core (Recommended)

```bash
git clone https://github.com/Unicorn-Commander/unicorn-npu-core.git
cd unicorn-npu-core

# Install with prebuilt kernel module (40 seconds)
sudo bash releases/v1.0.0/install-amdxdna-prebuilt.sh

# Verify
lsmod | grep amdxdna
dmesg | tail -20  # Should show NPU initialization
```

#### Option B: Build from AMD Source

```bash
# Install dependencies
sudo apt install -y dkms linux-headers-$(uname -r)

# Clone AMD's driver
git clone https://github.com/amd/xdna-driver.git
cd xdna-driver

# Build and install
sudo ./build.sh -install

# Load module
sudo modprobe amdxdna

# Verify
lsmod | grep amdxdna
```

#### Troubleshooting Driver Installation

```bash
# Check kernel version (must be 6.10+)
uname -r

# If kernel too old, update:
sudo apt install --install-recommends linux-generic-hwe-24.04

# Check dmesg for errors
dmesg | grep -i amdxdna
dmesg | grep -i npu

# Check device permissions
ls -l /dev/accel/accel0
# Should show: crw-rw---- 1 root render

# Add user to render group if needed
sudo usermod -aG render $USER
newgrp render
```

---

## 🔍 Verification & Testing

### Verify NPU Setup

```bash
# 1. Check device exists
ls -l /dev/accel/accel0

# 2. Check kernel module
lsmod | grep amdxdna

# 3. Check XRT
source /opt/xilinx/xrt/setup.sh
xrt-smi examine

# Expected output should show:
# - Device: AMD XDNA
# - Status: Ready
# - Version: 2.20.0
```

### Test NPU Runtime

```python
# test_npu.py
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "npu"))
from npu_runtime_aie2 import NPURuntime

# Initialize runtime
runtime = NPURuntime()

# Check availability
print(f"NPU Available: {runtime.is_available()}")

# Get device info
info = runtime.get_device_info()
print(f"Device Info: {info}")

# Load model
model_path = "/path/to/models/whisper-base-amd-npu-int8"
if runtime.load_model(model_path):
    print("✅ Model loaded successfully!")
else:
    print("❌ Model loading failed")
```

Run test:
```bash
python3 test_npu.py
```

### Test Full Transcription Pipeline

```bash
# Create test audio (5 seconds of sine wave)
python3 -c "
import numpy as np
import soundfile as sf
sr = 16000
t = np.linspace(0, 5, sr * 5)
audio = 0.5 * np.sin(2 * np.pi * 440 * t)
sf.write('test_audio.wav', audio, sr)
"

# Test transcription
curl -X POST \
  -F "file=@test_audio.wav" \
  http://localhost:9000/v1/audio/transcriptions

# Should return JSON with:
# - text: transcribed text
# - npu_accelerated: true
# - processing_time: < 1 second
# - real_time_factor: < 0.01
```

---

## 📊 Performance Benchmarking

### Benchmark Your System

```python
# benchmark.py
import time
import numpy as np
import soundfile as sf
from npu_runtime_aie2 import NPURuntime

def benchmark_npu(audio_duration=60):
    """Benchmark NPU transcription performance"""

    # Generate test audio
    sr = 16000
    audio = np.random.randn(int(audio_duration * sr)).astype(np.float32)

    # Save to file
    sf.write("benchmark_audio.wav", audio, sr)

    # Initialize runtime
    runtime = NPURuntime()
    runtime.load_model("whisper-base")

    # Benchmark
    start = time.time()
    result = runtime.transcribe("benchmark_audio.wav")
    elapsed = time.time() - start

    # Calculate metrics
    rtf = elapsed / audio_duration
    speedup = audio_duration / elapsed

    print(f"📊 Benchmark Results:")
    print(f"  Audio Duration: {audio_duration}s")
    print(f"  Processing Time: {elapsed:.2f}s")
    print(f"  Real-Time Factor: {rtf:.4f}")
    print(f"  Speedup: {speedup:.1f}x realtime")
    print(f"  NPU Accelerated: {result['npu_accelerated']}")

if __name__ == "__main__":
    benchmark_npu(60)  # 60 seconds of audio
```

Run benchmark:
```bash
python3 benchmark.py
```

**Expected Results (AMD Phoenix NPU)**:
- Real-Time Factor: 0.004 - 0.006
- Speedup: 160x - 250x realtime
- Processing 60s audio in: 0.24s - 0.38s

---

## 🐳 Docker Configuration

### Environment Variables

```yaml
# docker-compose-uc1-optimized.yml
unicorn-amanuensis:
  environment:
    # Model configuration
    WHISPER_MODEL: base                    # base, medium, large, large-v3
    USE_NPU: "true"                        # Enable NPU acceleration
    WHISPER_NPU_MODEL_PATH: /models/whisper-base-amd-npu-int8

    # Performance tuning
    BATCH_SIZE: "16"                       # Batch size for processing
    COMPUTE_TYPE: "int8"                   # Quantization: int8, float16

    # Optional features
    HF_TOKEN: ""                           # HuggingFace token for diarization
    CPU_ONLY_MODE: "false"                 # Force CPU mode

  # Device access
  devices:
    - /dev/accel/accel0:/dev/accel/accel0  # NPU device
    - /dev/dri:/dev/dri                    # DRI devices for XRT

  # Group permissions for hardware access
  group_add:
    - "44"      # video group
    - "992"     # render group

  # Volume mounts
  volumes:
    - ./whisperx/models:/models:ro         # Models directory
    - /opt/xilinx/xrt:/opt/xilinx/xrt:ro   # XRT runtime
```

### Rebuild Container After Changes

```bash
# Rebuild image
docker compose -f docker-compose-uc1-optimized.yml build unicorn-amanuensis

# Restart container
docker compose -f docker-compose-uc1-optimized.yml up -d unicorn-amanuensis

# Check logs
docker logs -f unicorn-amanuensis
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "NPU device not found"

**Symptoms**:
```
⚠️ NPU not available - using CPU fallback
```

**Solutions**:
```bash
# Check device exists
ls -l /dev/accel/accel0

# If not, install amdxdna driver
sudo modprobe amdxdna

# Check permissions
sudo chmod 666 /dev/accel/accel0  # Temporary fix
sudo usermod -aG render $USER     # Permanent fix
```

#### 2. "Could not load AIE2 driver"

**Symptoms**:
```
⚠️ AIE2 driver loading failed
```

**Solutions**:
```bash
# Check MLIR tools installed
which aie-opt
which aie-translate

# If missing, install MLIR-AIE
# See "Install MLIR-AIE2 Tools" section above

# Check Python path
python3 -c "import sys; print('\n'.join(sys.path))"
```

#### 3. "ONNX models not found"

**Symptoms**:
```
❌ ONNX models not found in /models/whisper-base-onnx-int8
```

**Solutions**:
```bash
# Download models
cd ~/Unicorn-Amanuensis/whisperx
./download-models.sh

# Or manually download
mkdir -p models/whisper-base-onnx-int8
cd models
# Download from HuggingFace or use ONNX cache
```

#### 4. "High CPU usage during transcription"

**Symptoms**:
- CPU at 400%+ during transcription
- NPU shows 0% usage

**Diagnosis**:
```bash
# Check if NPU is actually being used
watch -n 1 'sudo cat /sys/kernel/debug/amdxdna/0/status'

# Check NPU device status
curl http://localhost:9000/npu/status
```

**Solutions**:
- Ensure `USE_NPU=true` in environment
- Verify NPU drivers loaded: `lsmod | grep amdxdna`
- Check server logs for NPU initialization

#### 5. "XRT library not found"

**Symptoms**:
```
Error: libxrt_coreutil.so.2: cannot open shared object file
```

**Solutions**:
```bash
# Source XRT environment
source /opt/xilinx/xrt/setup.sh

# Add to .bashrc for permanence
echo 'source /opt/xilinx/xrt/setup.sh' >> ~/.bashrc

# Or set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
```

### Getting Help

1. **Check logs**:
   ```bash
   # Docker logs
   docker logs -f unicorn-amanuensis

   # System logs
   sudo journalctl -u docker -f
   sudo dmesg | tail -50
   ```

2. **Enable debug logging**:
   ```bash
   export LOG_LEVEL=DEBUG
   python3 server_npu_mlir.py
   ```

3. **Community support**:
   - GitHub Issues: https://github.com/Unicorn-Commander/Unicorn-Amanuensis/issues
   - Discord: https://discord.gg/unicorn-commander
   - Email: hello@magicunicorn.tech

---

## 📚 Additional Resources

### Documentation
- [NPU Runtime Documentation](./NPU_RUNTIME_DOCUMENTATION.md)
- [MLIR-AIE2 Kernel Guide](https://github.com/Unicorn-Commander/unicorn-npu-core/docs/kernels.md)
- [Unicorn NPU Core](https://github.com/Unicorn-Commander/unicorn-npu-core)

### Example Code
- [Python Examples](./examples/)
- [REST API Examples](./examples/api/)
- [Streaming Examples](./examples/streaming/)

### Performance Tuning
- [Optimization Guide](./OPTIMIZATION.md)
- [Benchmarking Tools](./tools/benchmark/)

---

## 📄 License

MIT License - See [LICENSE](./LICENSE)

---

**✨ Made with magic by Magic Unicorn Unconventional Technology & Stuff Inc.**

*Making AI impossibly fast on the hardware you already own.*
