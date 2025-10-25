# NPU Models Installation Guide

Quick guide to setting up and using the custom NPU-optimized Whisper models.

## 📦 What's Included

Three NPU-optimized INT8 Whisper models:

- **Whisper Base** (122 MB) - 220x speedup, 0.0045 RTF
- **Whisper Medium** (1.2 GB) - 125x speedup, 0.008 RTF
- **Whisper Large** (2.4 GB) - 67x speedup, 0.015 RTF

## ✅ Prerequisites

### Hardware
- AMD Ryzen 7040/8040 series (Phoenix/Hawk Point)
- AMD XDNA NPU (16 TOPS INT8)
- `/dev/accel/accel0` device must be accessible

### Software
- XRT 2.20.0 (Xilinx Runtime) installed
- unicorn-npu-core v1.0.0+
- Python 3.10+

## 🚀 Quick Installation

### 1. Install Host NPU Support

If not already done:

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis

# Install NPU host dependencies (XRT, drivers, etc.)
./install-npu-host.sh
```

### 2. Models Are Already Downloaded!

The models are already in this directory:

```bash
ls -lh npu-models/
# whisper-base-npu-int8/    - Links to existing Base model
# whisper-medium-int8/       - 1.3 GB downloaded
# whisper-large-int8/        - 2.4 GB downloaded
```

### 3. Verify NPU Access

```bash
# Check NPU device
ls -l /dev/accel/accel0

# Check XRT
/opt/xilinx/xrt/bin/xrt-smi examine

# Test NPU detection
python3 -c "
from pathlib import Path
print('NPU available:', Path('/dev/accel/accel0').exists())
"
```

## 📖 Usage

### Basic Transcription

```python
from npu_runtime_aie2 import NPURuntime

# Initialize with Whisper Base (fastest)
runtime = NPURuntime()
runtime.load_model("npu-models/whisper-base-npu-int8")

# Transcribe
result = runtime.transcribe("audio.wav")
print(result["text"])
print(f"Processed in: {result['processing_time']:.2f}s")
print(f"NPU accelerated: {result['npu_accelerated']}")
```

### With Diarization (Speaker Identification)

```python
from whisperx.npu.npu_optimization.unified_stt_diarization import UnifiedSTTDiarization

# Initialize with diarization
stt = UnifiedSTTDiarization(
    model="whisper-base-npu-int8",
    device="npu",
    diarize=True,
    num_speakers=4  # Or None for auto-detection
)

# Transcribe meeting
result = stt.transcribe("meeting.wav")

# Display with speakers
for segment in result["segments"]:
    speaker = segment.get("speaker", "UNKNOWN")
    start = segment["start"]
    text = segment["text"]
    print(f"[{start:.2f}s] {speaker}: {text}")
```

### Model Selection

Choose model based on your needs:

```python
models = {
    "fast": "npu-models/whisper-base-npu-int8",      # 220x, 16s for 1hr audio
    "balanced": "npu-models/whisper-medium-int8",    # 125x, 29s for 1hr audio
    "accurate": "npu-models/whisper-large-int8"      # 67x, 54s for 1hr audio
}

# Use the fastest for real-time
runtime.load_model(models["fast"])

# Or highest accuracy for professional work
runtime.load_model(models["accurate"])
```

## 🔧 Integration with Unicorn Amanuensis Server

The NPU models integrate automatically with the Unicorn Amanuensis server.

### Environment Variables

Set in your `.env` or docker-compose.yml:

```bash
# Enable NPU acceleration
WHISPER_DEVICE=npu

# Select NPU model
WHISPER_MODEL=base          # Uses whisper-base-npu-int8
# WHISPER_MODEL=medium      # Uses whisper-medium-int8
# WHISPER_MODEL=large-v3    # Uses whisper-large-int8

# NPU model path (optional, auto-detected)
WHISPER_NPU_MODEL_PATH=/app/npu-models
```

### Docker Deployment

When using Docker, mount the models directory:

```yaml
services:
  unicorn-amanuensis:
    image: unicorn/amanuensis:npu
    devices:
      - /dev/accel/accel0:/dev/accel/accel0
      - /dev/dri:/dev/dri
    volumes:
      - ./npu-models:/app/npu-models:ro
    environment:
      - WHISPER_DEVICE=npu
      - WHISPER_MODEL=base
```

## 📊 Performance Testing

Test NPU performance:

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis

# Create test script
cat > test_npu_performance.py << 'EOF'
#!/usr/bin/env python3
import time
from npu_runtime_aie2 import NPURuntime
import sys

# Test audio file (provide your own)
audio_file = sys.argv[1] if len(sys.argv) > 1 else "test_audio.wav"

print("🧪 Testing NPU Performance")
print("=" * 50)

# Test Base model
print("\n📊 Whisper Base NPU:")
runtime = NPURuntime()
runtime.load_model("npu-models/whisper-base-npu-int8")

start = time.time()
result = runtime.transcribe(audio_file)
elapsed = time.time() - start

audio_duration = result.get("duration", 60.0)
rtf = elapsed / audio_duration if audio_duration > 0 else 0

print(f"  Processing time: {elapsed:.2f}s")
print(f"  Audio duration: {audio_duration:.2f}s")
print(f"  RTF: {rtf:.4f}")
print(f"  Speedup: {1/rtf:.1f}x")
print(f"  Text preview: {result['text'][:100]}...")
EOF

chmod +x test_npu_performance.py

# Run test (provide your audio file)
python3 test_npu_performance.py your_audio.wav
```

## 🐛 Troubleshooting

### NPU Not Detected

```bash
# Check device
ls -l /dev/accel/accel0
# If missing, install drivers:
cd ~/unicorn-npu-core
sudo bash scripts/install-amdxdna-driver.sh

# Check XRT
/opt/xilinx/xrt/bin/xrt-smi examine
# If not found, install XRT:
sudo bash scripts/install-xrt-prebuilt.sh
```

### Permission Denied

```bash
# Add user to render/video groups
sudo usermod -aG render $USER
sudo usermod -aG video $USER

# Logout and login for groups to take effect
```

### Model Not Loading

```bash
# Verify model files exist
ls -lh npu-models/whisper-base-npu-int8/
ls -lh npu-models/whisper-medium-int8/*.onnx
ls -lh npu-models/whisper-large-int8/*.onnx

# Check paths in config
cat npu-models/whisper-base-npu-int8/npu_config.json
```

### Poor Performance

```bash
# Verify NPU is actually being used
python3 -c "
from npu_runtime_aie2 import NPURuntime
runtime = NPURuntime()
print('NPU available:', runtime.is_available())
print('Device info:', runtime.get_device_info())
"

# Check for CPU fallback in logs
# If you see 'Falling back to CPU', NPU is not being used
```

## 📚 Additional Resources

- **Main Documentation**: ../CLAUDE.md (see "AMD PHOENIX NPU CUSTOM QUANTIZATION" section)
- **NPU Models README**: README.md (this directory)
- **NPU Runtime Docs**: ../NPU_RUNTIME_DOCUMENTATION.md
- **Installation Guide**: ../install-npu-host.sh

## 🦄 Support

For issues:
1. Check CLAUDE.md troubleshooting section
2. Verify NPU device is accessible
3. Check logs for errors
4. Report at: https://github.com/Unicorn-Commander/Unicorn-Amanuensis/issues

---

**🚀 Powered by Magic Unicorn Unconventional Technology & Stuff Inc.**

*Making AI impossibly fast on the hardware you already own.*
