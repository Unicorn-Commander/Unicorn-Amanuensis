# Unicorn Amanuensis - WhisperX with OpenVINO Intel iGPU Setup

## Overview

This setup provides WhisperX with full features (word-level timestamps, speaker diarization, language detection) accelerated using OpenVINO on Intel integrated GPUs.

## Features

- **WhisperX Integration**: Full WhisperX functionality including alignment and diarization
- **Intel iGPU Acceleration**: OpenVINO optimization for Intel integrated graphics
- **Word-level Timestamps**: Precise word timing using phoneme alignment
- **Speaker Diarization**: Identify different speakers using pyannote.audio
- **Language Detection**: Automatic language identification
- **Hardware Transcoding**: FFmpeg with Intel QSV/VA-API support
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI Whisper API

## Prerequisites

### Hardware Requirements
- Intel CPU with integrated GPU (Intel UHD, Iris Xe, Arc)
- Minimum 8GB RAM (16GB recommended for large models)
- 10GB free disk space for models

### Software Requirements
- Docker and Docker Compose
- Intel GPU drivers installed on host
- Linux kernel 5.4+ with i915 driver

## Quick Start

1. **Check Intel GPU availability**:
```bash
# Check for Intel GPU
lspci | grep -i "VGA.*Intel"

# Check DRI devices
ls -la /dev/dri/

# Test VA-API (optional)
vainfo
```

2. **Set up environment**:
```bash
cd /home/ucadmin/Unicorn-Amanuensis

# Create .env file
cat > .env << EOF
WHISPER_MODEL=large-v3
WHISPER_DEVICE=igpu
COMPUTE_TYPE=int8
BATCH_SIZE=16
ENABLE_DIARIZATION=true
MAX_SPEAKERS=10
HF_TOKEN=your_huggingface_token_here
EOF
```

3. **Start the service**:
```bash
./start-whisperx-igpu.sh
```

4. **Test the service**:
```bash
./test-whisperx.sh
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `large-v3` | Whisper model size (tiny, base, small, medium, large, large-v2, large-v3) |
| `WHISPER_DEVICE` | `igpu` | Device to use (igpu for Intel GPU) |
| `COMPUTE_TYPE` | `int8` | Compute precision (int8, float16, float32) |
| `BATCH_SIZE` | `16` | Batch size for processing |
| `ENABLE_DIARIZATION` | `true` | Enable speaker diarization |
| `MAX_SPEAKERS` | `10` | Maximum number of speakers |
| `HF_TOKEN` | `` | Hugging Face token for diarization models |

### Model Selection

Choose model based on your needs:

- **tiny**: Fastest, lowest accuracy (~1GB)
- **base**: Good balance (~1.5GB)
- **small**: Better accuracy (~2.5GB)
- **medium**: High accuracy (~5GB)
- **large-v3**: Best accuracy (~10GB)

For Intel iGPU, we recommend:
- **base** or **small** for real-time processing
- **large-v3** for best accuracy with longer processing times

### Compute Type Optimization

- **int8**: Best performance on Intel iGPU (recommended)
- **float16**: Good balance of speed and precision
- **float32**: Full precision (slowest)

## API Usage

### Basic Transcription
```bash
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "language=en"
```

### With Word Timestamps
```bash
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "word_timestamps=true" \
  -F "timestamps=true"
```

### With Speaker Diarization
```bash
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "diarize=true" \
  -F "min_speakers=2" \
  -F "max_speakers=5"
```

### Auto Language Detection
```bash
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "language=auto"
```

## Performance Optimization

### Intel iGPU Optimization

1. **Enable GPU boost**:
```bash
# Check current GPU frequency
cat /sys/class/drm/card0/gt_cur_freq_mhz

# Set to maximum (requires root)
echo performance | sudo tee /sys/class/drm/card0/power/rc6_enable
```

2. **Optimize OpenVINO cache**:
```bash
# Set cache directory in docker-compose
environment:
  OV_CACHE_DIR: "/app/models/cache"
```

3. **Memory allocation**:
```bash
# Increase GPU memory allocation
echo 2048 | sudo tee /sys/kernel/debug/dri/0/i915_gem_shrink
```

### Docker Optimization

1. **Use tmpfs for temporary files**:
```yaml
tmpfs:
  - /tmp:size=2G
```

2. **CPU affinity** (for mixed workloads):
```yaml
cpuset: "0-3"  # Limit to specific CPU cores
```

## Troubleshooting

### Intel GPU Not Detected

1. Check kernel module:
```bash
lsmod | grep i915
```

2. Check permissions:
```bash
# Add user to video and render groups
sudo usermod -aG video,render $USER
# Log out and back in
```

3. Check Docker GPU access:
```bash
docker run --rm --device=/dev/dri ubuntu ls -la /dev/dri/
```

### OpenVINO Issues

1. Check OpenVINO devices:
```bash
curl http://localhost:9000/gpu-status
```

2. Verify OpenVINO installation in container:
```bash
docker exec unicorn-amanuensis-whisperx python -c "import openvino; print(openvino.__version__)"
```

### Performance Issues

1. Monitor GPU usage:
```bash
intel_gpu_top
```

2. Check container resources:
```bash
docker stats unicorn-amanuensis-whisperx
```

3. Reduce batch size or switch to smaller model if OOM occurs

### Diarization Not Working

1. Ensure HF_TOKEN is set in .env
2. Check token permissions on Hugging Face
3. Verify pyannote.audio models download:
```bash
docker logs unicorn-amanuensis-whisperx | grep pyannote
```

## Advanced Configuration

### Multi-GPU Setup

If you have both Intel iGPU and discrete GPU:

```yaml
environment:
  WHISPER_DEVICE: "igpu"  # Use Intel for transcription
  DIARIZATION_DEVICE: "cuda"  # Use NVIDIA for diarization
```

### Custom Model Path

```yaml
volumes:
  - /path/to/custom/models:/app/models
environment:
  MODEL_PATH: "/app/models/custom-whisper"
```

### Batch Processing

For processing multiple files:

```python
import requests
import glob

files = glob.glob("*.mp3")
for file in files:
    with open(file, 'rb') as f:
        response = requests.post(
            "http://localhost:9000/v1/audio/transcriptions",
            files={"file": f},
            data={"word_timestamps": True}
        )
        print(f"{file}: {response.json()}")
```

## Monitoring

### Health Check
```bash
curl http://localhost:9000/health
```

### GPU Status
```bash
curl http://localhost:9000/gpu-status
```

### Logs
```bash
docker-compose -f docker-compose.whisperx.yml logs -f
```

### Metrics
The service provides performance metrics in each response:
- `rtf`: Real-time factor (lower is better)
- `transcribe_time`: Time spent in transcription
- `align_time`: Time spent in alignment (word timestamps)
- `diarize_time`: Time spent in diarization

## Integration with UC-1 Pro

To integrate with the main UC-1 Pro stack:

1. Add to main docker-compose.yml:
```yaml
services:
  unicorn-amanuensis:
    extends:
      file: /home/ucadmin/Unicorn-Amanuensis/docker-compose.whisperx.yml
      service: whisperx-openvino
    networks:
      - unicorn-network
```

2. Configure Open-WebUI to use the service:
- API URL: `http://unicorn-amanuensis:9000/v1/audio/transcriptions`

## Support

For issues or questions:
- GitHub: https://github.com/Unicorn-Commander/Unicorn-Amanuensis
- Check logs: `docker logs unicorn-amanuensis-whisperx`
- Test script: `./test-whisperx.sh`