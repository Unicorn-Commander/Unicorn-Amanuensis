# 🦄 Unicorn Amanuensis - AMD Phoenix NPU Edition

## Installation Summary for AMD Phoenix NPU (XDNA1)

This guide is specifically for the **AMD Ryzen 9 8945HS** with **Phoenix NPU** hardware.

---

## ✅ What's Ready

All installation scripts are tested and working:

| Script | Purpose | Use Case |
|--------|---------|----------|
| `install-docker-npu.sh` | Full NPU install from source | First-time build |
| `rebuild-npu.sh` | Quick rebuild | After code changes |
| `publish-dockerhub.sh` | Publish to Docker Hub | Share with community |
| `install.sh` | Auto-detect hardware | Multi-platform install |
| `install-bare-metal.sh` | Python venv setup | Non-Docker deployment |

---

## 🚀 Quick Start (Recommended)

### Option 1: Use Pre-Built Image (Fastest)

Once published to Docker Hub:

```bash
# Pull and run
docker pull magicunicorn/unicorn-amanuensis-npu:latest
docker network create unicorn-network
docker run -d \
  --name amanuensis-npu \
  --device /dev/accel/accel0 \
  --device /dev/dri \
  -p 9000:9000 \
  --network unicorn-network \
  magicunicorn/unicorn-amanuensis-npu:latest

# Access GUI
open http://localhost:9000/web
```

### Option 2: Build from Source

```bash
# Clone repository
git clone https://github.com/Unicorn-Commander/Unicorn-Amanuensis.git
cd Unicorn-Amanuensis

# Run NPU installer
./install-docker-npu.sh

# Access GUI
open http://localhost:9000/web
```

---

## 📋 Prerequisites

### 1. Hardware Requirements
- AMD Ryzen 9 8945HS (or similar Phoenix APU)
- AMD NPU (XDNA1) with driver installed
- 16GB+ RAM recommended
- 20GB+ disk space for models

### 2. Software Requirements

```bash
# Docker
docker --version  # Should be 20.10+
docker compose version  # Should be 2.0+

# NPU driver check
ls -la /dev/accel/accel0  # Should exist with rw permissions
lspci | grep Phoenix  # Should show Phoenix devices
```

### 3. User Permissions

```bash
# Add user to required groups
sudo usermod -aG docker,render,video $USER

# Log out and log back in for changes to take effect

# Verify groups
groups | grep -E "(docker|render|video)"
```

---

## 🔧 Installation

### Step 1: Clone Repository

```bash
cd /home/ucadmin/UC-1  # Or your preferred directory
git clone https://github.com/Unicorn-Commander/Unicorn-Amanuensis.git
cd Unicorn-Amanuensis
```

### Step 2: Verify NPU Device

```bash
# Check NPU device exists
ls -la /dev/accel/accel0

# Expected output:
# crw-rw-rw- 1 root render 261, 0 Oct  7 22:32 /dev/accel/accel0

# If permissions wrong, fix with:
sudo chmod 666 /dev/accel/accel0
```

### Step 3: Run Installer

```bash
# Make installer executable (if not already)
chmod +x install-docker-npu.sh

# Run installer
./install-docker-npu.sh
```

This will:
1. ✅ Check NPU device accessibility
2. ✅ Verify Docker is installed
3. ✅ Build Docker image with ONNX models (~10 minutes)
4. ✅ Create Docker network
5. ✅ Start container
6. ✅ Run health checks
7. ✅ Display access URLs

### Step 4: Verify Installation

```bash
# Check container is running
docker ps | grep amanuensis

# Test health endpoint
curl http://localhost:9000/health

# Expected response:
# {"status":"healthy","engine":"openai_whisper","timestamp":...}

# Access GUI in browser
open http://localhost:9000/web
```

---

## 🎨 GUI Access

Open your browser to: **http://localhost:9000/web**

### Features
- 🎨 **Three beautiful themes**: Light, Dark, Magic Unicorn
- 📤 **Drag-and-drop upload**: Just drag audio files
- 🎵 **Waveform visualization**: See your audio
- 📊 **Real-time progress**: Live transcription status
- 🎙️ **Speaker diarization**: Identify speakers
- ⏱️ **Word timestamps**: Precise timing
- 💾 **Export formats**: JSON, SRT, VTT, TXT

---

## 📡 API Usage

### OpenAI-Compatible Endpoint

```bash
# Basic transcription
curl -X POST \
  -F "file=@audio.mp3" \
  http://localhost:9000/v1/audio/transcriptions

# With speaker diarization
curl -X POST \
  -F "file=@audio.mp3" \
  -F "diarization=true" \
  -F "word_timestamps=true" \
  http://localhost:9000/v1/audio/transcriptions
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:9000/v1"
)

with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json"
    )

print(transcript.text)
```

---

## 🔄 Maintenance

### View Logs

```bash
docker logs -f amanuensis-npu
```

### Restart Container

```bash
docker restart amanuensis-npu
```

### Rebuild Container

```bash
./rebuild-npu.sh
```

This script:
1. Stops existing container
2. Removes old image
3. Cleans Docker cache
4. Builds new image
5. Starts container
6. Shows status and logs

### Stop Container

```bash
cd whisperx
docker compose -f docker-compose-npu.yml down
cd ..
```

---

## 🌐 Publishing to Docker Hub

Once tested and working:

```bash
# Login to Docker Hub
docker login

# Run publish script
./publish-dockerhub.sh

# Enter version when prompted (e.g., "1.0.0" or "latest")
```

After publishing, anyone can install with:

```bash
docker pull magicunicorn/unicorn-amanuensis-npu:latest
```

---

## 🧪 Testing Checklist

After installation, verify everything works:

- [ ] Container running: `docker ps | grep amanuensis`
- [ ] Health check passes: `curl http://localhost:9000/health`
- [ ] GUI loads: Open http://localhost:9000/web
- [ ] Themes switch: Try Light, Dark, Magic Unicorn themes
- [ ] File upload works: Drag-drop an audio file
- [ ] Transcription completes: Wait for results
- [ ] API endpoint works: Test with curl
- [ ] Export functions: Download JSON, SRT, VTT, TXT
- [ ] NPU device accessible: `ls -la /dev/accel/accel0`
- [ ] Logs are clean: `docker logs amanuensis-npu`

---

## 🐛 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs amanuensis-npu

# Common issues:
# 1. Port 9000 already in use
sudo lsof -i :9000

# 2. NPU device not accessible
ls -la /dev/accel/accel0
sudo chmod 666 /dev/accel/accel0

# 3. Docker network missing
docker network create unicorn-network

# Rebuild from scratch
./rebuild-npu.sh
```

### GUI Not Loading

```bash
# Check container health
docker ps | grep amanuensis

# Test health endpoint
curl http://localhost:9000/health

# Check logs
docker logs amanuensis-npu --tail 50

# Restart container
docker restart amanuensis-npu
```

### NPU Not Detected

```bash
# Check device exists
ls -la /dev/accel/accel0

# Check Phoenix in PCI
lspci | grep Phoenix

# Check permissions
# Should be: crw-rw-rw- 1 root render

# Fix permissions
sudo chmod 666 /dev/accel/accel0

# Check user in render group
groups | grep render
```

### Slow Transcription

Currently using CPU fallback. To enable NPU acceleration:

1. Add optimized ONNX models to `whisperx/models/whisper-base-onnx/`
2. Rebuild container: `./rebuild-npu.sh`
3. Verify NPU inference loads in logs

---

## 📊 Performance

### Current Status (CPU Fallback)
- **~1x realtime** (1 minute audio = ~60 seconds processing)
- **OpenAI Whisper base model**
- **CPU mode**: Works but not optimized

### Expected with NPU Optimization
- **~2-4x realtime** (1 minute audio = ~15-30 seconds)
- **16 TOPS NPU power**
- **Low power**: ~5-10W vs ~30-50W CPU

---

## 🔗 Integration

### With UC-1 Pro

Add to your `docker-compose.yml`:

```yaml
services:
  unicorn-amanuensis:
    image: magicunicorn/unicorn-amanuensis-npu:latest
    ports:
      - "9000:9000"
    devices:
      - /dev/accel/accel0:/dev/accel/accel0
      - /dev/dri:/dev/dri
    networks:
      - unicorn-network
```

### With Open-WebUI

Add to Open-WebUI `.env`:

```env
AUDIO_STT_ENGINE=openai
AUDIO_STT_OPENAI_API_KEY=dummy-key
AUDIO_STT_OPENAI_API_BASE_URL=http://amanuensis-npu:9000/v1
AUDIO_STT_MODEL=whisper-1
```

---

## 📚 Documentation

- **Quick Start**: See `QUICK-START-NPU.md`
- **Full NPU Guide**: See `INSTALL-NPU.md`
- **Main README**: See `README.md`
- **Docker Compose**: See `whisperx/docker-compose-npu.yml`
- **Dockerfile**: See `whisperx/Dockerfile.npu`

---

## 💬 Support

- **GitHub Issues**: [Report bugs](https://github.com/Unicorn-Commander/Unicorn-Amanuensis/issues)
- **Discussions**: [Ask questions](https://github.com/Unicorn-Commander/Unicorn-Amanuensis/discussions)
- **Discord**: Join Magic Unicorn community

---

## 📝 License

MIT License - See [LICENSE](LICENSE)

---

<div align="center">
  <b>🦄 Unicorn Amanuensis 🦄</b><br>
  <i>Professional AI Transcription on AMD Phoenix NPU</i><br>
  <b>Free Your GPU • Transcribe Everything • Deploy Anywhere</b><br><br>
  <i>Built with 💜 by Magic Unicorn</i><br>
  <i>Unconventional Technology & Stuff Inc.</i>
</div>
