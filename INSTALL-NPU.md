# Unicorn Amanuensis - AMD Phoenix NPU Installation

## Quick Start with Docker Hub (Recommended)

The easiest way to install Unicorn Amanuensis on AMD Phoenix NPU:

```bash
# 1. Pull from Docker Hub
docker pull magicunicorn/unicorn-amanuensis-npu:latest

# 2. Create network
docker network create unicorn-network

# 3. Run container
docker run -d \
  --name amanuensis-npu \
  --device /dev/accel/accel0 \
  --device /dev/dri \
  -p 9000:9000 \
  -v $(pwd)/uploads:/app/uploads \
  --network unicorn-network \
  --restart unless-stopped \
  magicunicorn/unicorn-amanuensis-npu:latest
```

Access the GUI at: **http://localhost:9000/web**

---

## Build from Source

If you prefer to build locally:

```bash
# 1. Clone repository
git clone https://github.com/Unicorn-Commander/Unicorn-Amanuensis.git
cd Unicorn-Amanuensis

# 2. Run NPU installer
./install-docker-npu.sh
```

This will:
- Verify NPU device is accessible
- Build Docker image with ONNX models
- Start the container
- Run health checks

---

## Prerequisites

### 1. NPU Driver (XDNA)

Make sure the NPU driver is installed and device is accessible:

```bash
# Check NPU device
ls -la /dev/accel/accel0

# Should show: crw-rw-rw- 1 root render 261, 0
```

If device doesn't exist or has wrong permissions:

```bash
# Fix permissions
sudo chmod 666 /dev/accel/accel0

# Add user to render group
sudo usermod -aG render $USER
# Log out and log back in
```

### 2. Docker

```bash
# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
# Log out and log back in

# Verify
docker --version
```

---

## Verification

### Check Container Status

```bash
docker ps | grep amanuensis
```

Should show container as "healthy"

### Test API

```bash
# Check health
curl http://localhost:9000/health

# Test transcription (with audio file)
curl -X POST \
  -F "file=@audio.mp3" \
  http://localhost:9000/v1/audio/transcriptions
```

### View Logs

```bash
docker logs -f amanuensis-npu
```

---

## GUI Access

Open your browser to: **http://localhost:9000/web**

The GUI provides:
- 🎨 **Three beautiful themes** (Light, Dark, Magic Unicorn)
- 📤 **Drag-and-drop** file upload
- 📊 **Real-time progress** tracking
- 🎵 **Audio waveform** visualization
- 💾 **Multiple export formats** (JSON, SRT, VTT, TXT)

---

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Model settings
WHISPER_MODEL=base                    # tiny, base, small, medium, large-v3
WHISPER_MODEL_CACHE=/app/models       # Model cache directory

# API settings
API_PORT=9000                         # API server port
PYTHONUNBUFFERED=1                    # Enable live logging
```

### Use with docker-compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  amanuensis-npu:
    image: magicunicorn/unicorn-amanuensis-npu:latest
    container_name: amanuensis-npu
    ports:
      - "9000:9000"
    devices:
      - /dev/accel/accel0:/dev/accel/accel0
      - /dev/dri:/dev/dri
    volumes:
      - ./uploads:/app/uploads
      - ./models:/app/models
    environment:
      - WHISPER_MODEL=base
      - API_PORT=9000
    restart: unless-stopped
    networks:
      - unicorn-network

networks:
  unicorn-network:
    external: true
```

Then run:

```bash
docker compose up -d
```

---

## Troubleshooting

### NPU Device Not Found

```bash
# Check if NPU driver is loaded
lspci | grep -i Phoenix

# Check device permissions
ls -la /dev/accel/accel0

# Fix permissions
sudo chmod 666 /dev/accel/accel0
```

### Container Won't Start

```bash
# Check logs
docker logs amanuensis-npu

# Rebuild container
./rebuild-npu.sh

# Or manually:
docker stop amanuensis-npu
docker rm amanuensis-npu
docker rmi unicorn-amanuensis-npu:latest
./install-docker-npu.sh
```

### Model Download Issues

The container automatically downloads ONNX models from HuggingFace on first build:
- `encoder_model.onnx` (~300MB)
- `decoder_model.onnx` (~500MB)

If download fails, rebuild:

```bash
docker build -f whisperx/Dockerfile.npu -t unicorn-amanuensis-npu:latest whisperx/
```

### GUI Not Loading

1. Check container is running: `docker ps | grep amanuensis`
2. Check health endpoint: `curl http://localhost:9000/health`
3. Check logs: `docker logs amanuensis-npu`
4. Try accessing: `http://localhost:9000/web`

---

## Advanced Usage

### Use Different Models

The container supports all Whisper model sizes:

- `tiny` - 74M params, fastest, lowest accuracy
- `base` - 139M params, good balance (default)
- `small` - 483M params, better accuracy
- `medium` - 1.5GB, excellent accuracy
- `large-v3` - 3GB, best accuracy

Change model at runtime:

```bash
docker run -d \
  --name amanuensis-npu \
  --device /dev/accel/accel0 \
  --device /dev/dri \
  -p 9000:9000 \
  -e WHISPER_MODEL=small \
  magicunicorn/unicorn-amanuensis-npu:latest
```

### Remote Access

To access from other machines on your network:

```bash
# Allow firewall access
sudo ufw allow 9000/tcp

# Container already binds to 0.0.0.0:9000
# Access from other machines: http://YOUR_IP:9000/web
```

---

## Integration with UC-1 Pro

Unicorn Amanuensis integrates seamlessly with UC-1 Pro:

```yaml
# In UC-1 Pro docker-compose.yml
services:
  unicorn-amanuensis:
    image: magicunicorn/unicorn-amanuensis-npu:latest
    ports:
      - "9000:9000"
    devices:
      - /dev/accel/accel0:/dev/accel/accel0
      - /dev/dri:/dev/dri
    environment:
      - WHISPER_MODEL=base
    networks:
      - unicorn-network
```

Configure Open-WebUI to use it:

```env
AUDIO_STT_ENGINE=openai
AUDIO_STT_OPENAI_API_KEY=dummy-key
AUDIO_STT_OPENAI_API_BASE_URL=http://amanuensis-npu:9000/v1
AUDIO_STT_MODEL=whisper-1
```

---

## Performance

### AMD Phoenix NPU (XDNA1)

- **16 TOPS** AI performance
- **~2-4x realtime** transcription speed
- **Low power consumption** (~5-10W)
- **Frees GPU** for other AI workloads

### Benchmarks

| Audio Length | Processing Time | Realtime Factor |
|--------------|-----------------|-----------------|
| 1 minute     | ~15-30 seconds  | 2-4x            |
| 10 minutes   | ~2-5 minutes    | 2-5x            |
| 1 hour       | ~15-30 minutes  | 2-4x            |

*Base model, AMD Ryzen 9 8945HS with Phoenix NPU*

---

## Support

- 📖 **Documentation**: [GitHub Wiki](https://github.com/Unicorn-Commander/Unicorn-Amanuensis/wiki)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Unicorn-Commander/Unicorn-Amanuensis/issues)
- 💬 **Discord**: [Magic Unicorn Community](https://discord.gg/magicunicorn)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">
  <b>🦄 Powered by AMD Phoenix NPU 🦄</b><br>
  <i>Built with 💜 by Magic Unicorn</i><br>
  <b>Free Your GPU • Transcribe Everything • Deploy Anywhere</b>
</div>
