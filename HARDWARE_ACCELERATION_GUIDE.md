# Unicorn Amanuensis - Complete Hardware Acceleration Guide

## Overview

Unicorn Amanuensis supports multiple hardware acceleration backends for optimal transcription performance:

- **AMD NPU (XDNA)**: Phoenix/Hawk Point 16 TOPS, Strix Point 50 TOPS
- **Intel iGPU**: OpenVINO acceleration for Intel integrated graphics
- **NVIDIA GPU**: CUDA acceleration for discrete GPUs
- **CPU**: Optimized fallback with AVX2/AVX512 support

## Supported Hardware

### AMD NPU (XDNA)
- **Phoenix (Ryzen 7040 series)**: 16 TOPS XDNA1 NPU
  - Ryzen 7 7840U/HS, Ryzen 5 7640U/HS, etc.
- **Hawk Point (Ryzen 8040 series)**: 16 TOPS XDNA1 NPU
  - Ryzen 7 8840U/HS, Ryzen 5 8640U/HS, etc.
- **Strix Point (Ryzen AI 300 series)**: 50 TOPS XDNA2 NPU
  - Ryzen AI 9 HX 370, Ryzen AI 9 365, etc.

### Intel iGPU
- **Intel UHD Graphics**: Gen 11+ (Ice Lake and newer)
- **Intel Iris Xe**: Tiger Lake, Alder Lake, Raptor Lake
- **Intel Arc iGPU**: Meteor Lake and newer

### NVIDIA GPU
- **Consumer GPUs**: GTX 1060+, RTX 2060+, RTX 3060+, RTX 4060+
- **Professional GPUs**: T4, A10, A100, H100
- **Minimum VRAM**: 4GB (8GB+ recommended for large models)

## Quick Start

### Auto-Detection Mode (Recommended)

```bash
# Set to auto-detect best available hardware
export WHISPER_DEVICE=auto

# Start the service
./start-whisperx-igpu.sh
```

### Force Specific Hardware

```bash
# Force AMD NPU
export WHISPER_DEVICE=npu

# Force Intel iGPU
export WHISPER_DEVICE=igpu

# Force NVIDIA GPU
export WHISPER_DEVICE=cuda

# Force CPU only
export WHISPER_DEVICE=cpu
```

## AMD NPU Configuration

### Prerequisites

1. **Check NPU availability**:
```bash
# Check for NPU device
ls -la /dev/accel/accel0

# Check processor model
cat /proc/cpuinfo | grep "model name"
```

2. **Install AMD NPU drivers** (if not present):
```bash
# Ubuntu/Debian
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
sudo apt install ./amdgpu-install_6.0.60002-1_all.deb
sudo amdgpu-install --usecase=workstation --no-32
```

3. **Set permissions**:
```bash
# Add user to render group
sudo usermod -aG render $USER
# Log out and back in
```

### NPU-Specific Environment Variables

```bash
# .env file for NPU
WHISPER_DEVICE=npu
WHISPER_MODEL=base  # Start with base for testing
COMPUTE_TYPE=int8   # NPU optimized for INT8
BATCH_SIZE=8        # Lower batch size for NPU
NPU_TILES=1         # Number of NPU tiles to use
```

### NPU Performance Optimization

The NPU uses custom MLIR kernels and binary compilation for maximum performance:

1. **Model Quantization**: Automatically converts to INT8
2. **Kernel Fusion**: Combines operations for efficiency
3. **Memory Optimization**: Uses zero-copy transfers
4. **Power Efficiency**: ~10W for 16 TOPS performance

### Testing NPU Acceleration

```bash
# Test NPU availability
curl http://localhost:9000/gpu-status | jq '.npu'

# Benchmark NPU vs CPU
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@test.wav" \
  -F "response_format=json" | jq '.performance'
```

## Intel iGPU Configuration

### Prerequisites

1. **Check Intel GPU**:
```bash
# Check for Intel GPU
lspci | grep -i "VGA.*Intel"

# Check VA-API support
vainfo

# Check OpenCL support
clinfo | grep Intel
```

2. **Install Intel GPU drivers**:
```bash
# Add Intel graphics repository
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo apt-key add -
echo "deb [arch=amd64] https://repositories.intel.com/gpu/ubuntu focal/lts/2350 unified" | \
  sudo tee /etc/apt/sources.list.d/intel-graphics.list

# Install drivers
sudo apt update
sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2
```

### OpenVINO Optimization

```bash
# .env file for Intel iGPU
WHISPER_DEVICE=igpu
COMPUTE_TYPE=int8    # INT8 for best iGPU performance
OV_CACHE_DIR=/app/models/cache
```

## NVIDIA GPU Configuration

### Prerequisites

1. **Check NVIDIA GPU**:
```bash
nvidia-smi
```

2. **Install CUDA drivers** (if needed):
```bash
# Ubuntu/Debian
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
```

### CUDA Optimization

```bash
# .env file for NVIDIA GPU
WHISPER_DEVICE=cuda
COMPUTE_TYPE=float16  # FP16 for Tensor Cores
BATCH_SIZE=32         # Higher batch size for GPU
```

## Docker Configuration

### Complete docker-compose with all hardware support:

```yaml
version: '3.8'

services:
  whisperx:
    build:
      context: ./whisperx
      dockerfile: Dockerfile.whisperx_openvino
    container_name: unicorn-amanuensis
    environment:
      WHISPER_DEVICE: ${WHISPER_DEVICE:-auto}
      WHISPER_MODEL: ${WHISPER_MODEL:-base}
      COMPUTE_TYPE: ${COMPUTE_TYPE:-int8}
      BATCH_SIZE: ${BATCH_SIZE:-16}
      HF_TOKEN: ${HF_TOKEN}
    devices:
      # Intel/AMD iGPU
      - /dev/dri:/dev/dri
      # AMD NPU
      - /dev/accel:/dev/accel
    group_add:
      - video    # For GPU access
      - render   # For NPU/GPU access
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              count: all
    volumes:
      - ./models:/app/models
      - ./cache:/app/cache
    ports:
      - "9000:9000"
      - "9001:9001"
```

## Performance Comparison

| Hardware | Model | RTF* | Power | Quality |
|----------|-------|------|-------|---------|
| AMD NPU (16 TOPS) | base | 0.05x | 10W | 95% |
| AMD NPU (50 TOPS) | large | 0.03x | 15W | 98% |
| Intel iGPU (Xe) | base | 0.08x | 15W | 95% |
| Intel iGPU (Arc) | large | 0.06x | 25W | 98% |
| NVIDIA RTX 4060 | large | 0.02x | 60W | 98% |
| CPU (Ryzen 7840U) | base | 0.3x | 25W | 95% |

*RTF = Real-Time Factor (lower is better, 0.1x = 10x faster than real-time)

## API Usage

The API is the same regardless of hardware backend:

### Basic Transcription
```bash
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "language=en"
```

### With Hardware Info
```bash
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@audio.mp3" | jq '.config.device'
```

### Check Hardware Status
```bash
# Get current hardware backend
curl http://localhost:9000/health | jq '.backend'

# Detailed hardware info
curl http://localhost:9000/gpu-status
```

## Troubleshooting

### AMD NPU Issues

1. **NPU not detected**:
```bash
# Check kernel module
lsmod | grep amdxdna

# Load module if needed
sudo modprobe amdxdna

# Check device permissions
ls -la /dev/accel/
```

2. **NPU performance issues**:
- Ensure INT8 compute type
- Use smaller models (base/small)
- Check NPU utilization: `sudo cat /sys/kernel/debug/accel/accel0/usage`

### Intel iGPU Issues

1. **OpenVINO not detecting GPU**:
```bash
# Set environment variables
export OCL_ICD_VENDORS=/etc/OpenCL/vendors/
export InferenceEngine_DIR=/usr/local/lib/python3.10/dist-packages/openvino

# Test OpenVINO devices
python3 -c "import openvino; print(openvino.Core().available_devices)"
```

2. **VA-API errors**:
```bash
# Set correct driver
export LIBVA_DRIVER_NAME=iHD
export LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
```

### General Issues

1. **Auto-detection choosing wrong device**:
- Force specific device with `WHISPER_DEVICE` environment variable
- Check detection order in logs

2. **Out of memory**:
- Reduce batch size
- Use smaller model
- Enable memory growth (NVIDIA): `export TF_FORCE_GPU_ALLOW_GROWTH=true`

3. **Slow performance**:
- Check if correct backend is being used
- Verify compute type matches hardware (INT8 for NPU/iGPU, FP16 for NVIDIA)
- Monitor hardware utilization

## Advanced Configuration

### Multi-Device Setup

For systems with multiple acceleration options:

```python
# Priority order in auto-detection:
1. AMD NPU (if available)     # Most power efficient
2. Intel iGPU (if available)  # Good balance
3. NVIDIA GPU (if available)  # Highest performance
4. CPU (fallback)             # Always available
```

### Custom Model Paths

```bash
# Use custom ONNX models for NPU
WHISPER_MODEL=/path/to/custom/model.onnx

# Use Hugging Face model
WHISPER_MODEL=openai/whisper-large-v3
```

### Batch Processing

```python
import requests
import concurrent.futures

def transcribe(file_path):
    with open(file_path, 'rb') as f:
        response = requests.post(
            "http://localhost:9000/v1/audio/transcriptions",
            files={"file": f},
            data={"word_timestamps": True}
        )
    return response.json()

# Process multiple files in parallel
files = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(transcribe, files))
```

## Integration with Open-WebUI

Configure Open-WebUI to use hardware-accelerated transcription:

1. Set STT provider to "OpenAI"
2. Set API URL: `http://unicorn-amanuensis:9000/v1`
3. Model: `whisper-1` (auto-maps to configured model)

## Support

- GitHub Issues: https://github.com/Unicorn-Commander/Unicorn-Amanuensis
- Logs: `docker logs unicorn-amanuensis`
- Health: `curl http://localhost:9000/health`