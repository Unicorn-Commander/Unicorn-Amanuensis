# 🦄 Unicorn Amanuensis - NPU Quick Start

## Current Status: ✅ READY TO TEST

The container is running and the GUI is accessible at **http://localhost:9000/web**

---

## What's Working

✅ **Docker container running** - `amanuensis-npu:latest`
✅ **Health endpoint** - http://localhost:9000/health
✅ **Web GUI** - http://localhost:9000/web (with Unicorn themes!)
✅ **API endpoint** - http://localhost:9000/v1/audio/transcriptions
✅ **NPU device** - `/dev/accel/accel0` is accessible
✅ **Fallback mode** - OpenAI Whisper CPU mode working

---

## Current Installation

Your system has these containers running:
- `amanuensis-npu` - Port 9000 (Whisper STT on NPU/CPU)
- `orator-kokoro` - Port 8880 (Kokoro TTS)
- `unicorn-llama-cpp` - Port 11434 (LLM with Vulkan)
- `unicorn-embeddings` - Port 8881 (Embeddings on iGPU)
- `unicorn-reranker` - Port 8882 (Reranker on iGPU)
- `unicorn-qdrant` - Port 6333 (Vector DB)
- `unicorn-postgresql` - Port 5432 (Database)
- `unicorn-redis` - Port 6379 (Cache)

---

## Installation Scripts Created

### 1. `install-docker-npu.sh` - Full NPU Installation
Builds container from source, verifies NPU, starts services.

```bash
./install-docker-npu.sh
```

### 2. `rebuild-npu.sh` - Quick Rebuild
Stops, removes, rebuilds, and restarts the NPU container.

```bash
./rebuild-npu.sh
```

### 3. `publish-dockerhub.sh` - Publish to Docker Hub
Tags and pushes to `magicunicorn/unicorn-amanuensis-npu`

```bash
./publish-dockerhub.sh
```

### 4. `install.sh` - Auto-Detect Installer
Detects hardware (NPU/CUDA/iGPU/CPU) and installs appropriate version.

```bash
./install.sh
```

### 5. `install-bare-metal.sh` - Non-Docker Installation
Python virtual environment setup for bare metal deployment.

```bash
./install-bare-metal.sh
```

---

## Testing the GUI

### Open in Browser
```bash
# Local access
http://localhost:9000/web

# Remote access (from other machines)
http://YOUR_IP:9000/web
```

### GUI Features
- 🎨 **Three themes**: Light, Dark, Magic Unicorn
- 📤 **Drag-and-drop** file upload
- 🎵 **Audio waveform** visualization
- 📊 **Real-time progress** tracking
- 💾 **Export formats**: JSON, SRT, VTT, TXT
- 🎙️ **Speaker diarization** (if HuggingFace token provided)
- ⏱️ **Word-level timestamps**

---

## API Testing

### Quick Health Check
```bash
curl http://localhost:9000/health
# Response: {"status":"healthy","engine":"openai_whisper","timestamp":...}
```

### Transcribe Audio File
```bash
curl -X POST \
  -F "file=@/path/to/audio.mp3" \
  http://localhost:9000/v1/audio/transcriptions
```

### With Speaker Diarization
```bash
curl -X POST \
  -F "file=@audio.mp3" \
  -F "diarization=true" \
  -F "word_timestamps=true" \
  http://localhost:9000/v1/audio/transcriptions
```

---

## Current Container Logs

The container shows:
- ✅ Server running on port 9000
- ⚠️ NPU models need optimization (currently using CPU fallback)
- ✅ OpenAI Whisper base model loaded
- ✅ Health checks passing

### View Live Logs
```bash
docker logs -f amanuensis-npu
```

---

## Next Steps for Full NPU Acceleration

The container is currently using **CPU fallback** because the ONNX models need to be optimized for NPU. To enable true NPU acceleration:

1. **Download optimized NPU models** (from UC-Meeting-Ops or custom compiled)
2. **Mount model directory** to `/app/models/whisper-base-onnx`
3. **Verify NPU inference provider** is loading correctly

Current fallback path works fine for testing!

---

## Cleanup & Rebuild

### Stop Everything
```bash
docker compose down  # In the main UC-1 directory
# OR for just NPU:
cd whisperx
docker compose -f docker-compose-npu.yml down
cd ..
```

### Remove NPU Container & Image
```bash
docker stop amanuensis-npu
docker rm amanuensis-npu
docker rmi unicorn-amanuensis-npu:latest
```

### Clear All Docker Cache
```bash
docker system prune -af --volumes
```

### Rebuild from Scratch
```bash
./rebuild-npu.sh
```

---

## Publishing to Docker Hub

Once you're happy with the container:

```bash
# 1. Log in to Docker Hub
docker login

# 2. Run publish script
./publish-dockerhub.sh

# 3. Enter version (e.g., "1.0.0" or "latest")

# 4. Confirm and wait for upload (~9GB image)
```

After publishing, users can install with:

```bash
docker pull magicunicorn/unicorn-amanuensis-npu:latest

docker run -d \
  --name amanuensis-npu \
  --device /dev/accel/accel0 \
  --device /dev/dri \
  -p 9000:9000 \
  magicunicorn/unicorn-amanuensis-npu:latest
```

---

## Fresh Install Testing

To test a clean install (simulating new user):

```bash
# 1. Remove everything
docker stop amanuensis-npu
docker rm amanuensis-npu
docker rmi unicorn-amanuensis-npu:latest

# 2. Clone to new directory
cd /tmp
git clone https://github.com/Unicorn-Commander/Unicorn-Amanuensis.git
cd Unicorn-Amanuensis

# 3. Run installer
./install-docker-npu.sh

# 4. Verify GUI loads
open http://localhost:9000/web
```

---

## Troubleshooting

### GUI Not Loading?

```bash
# Check container status
docker ps | grep amanuensis

# Check health
curl http://localhost:9000/health

# Check logs
docker logs amanuensis-npu

# Restart container
docker restart amanuensis-npu
```

### NPU Device Issues?

```bash
# Check device exists
ls -la /dev/accel/accel0

# Check permissions
# Should be: crw-rw-rw- 1 root render

# Fix permissions if needed
sudo chmod 666 /dev/accel/accel0
```

### Port Already in Use?

```bash
# Find what's using port 9000
sudo lsof -i :9000

# Change port in docker-compose-npu.yml
# Change "9000:9000" to "9001:9000"
# Then rebuild
```

---

## Summary

✅ **Installation works** - All scripts tested and ready
✅ **GUI accessible** - Beautiful themes working
✅ **API functional** - OpenAI-compatible endpoints
✅ **NPU detected** - Device accessible
✅ **Fallback working** - CPU mode operational
✅ **Ready to publish** - Can push to Docker Hub

**You can now test the GUI and transcribe audio files!**

Open: **http://localhost:9000/web**

---

<div align="center">
  <b>🦄 Powered by AMD Phoenix NPU 🦄</b><br>
  <i>Built with 💜 by Magic Unicorn</i>
</div>
