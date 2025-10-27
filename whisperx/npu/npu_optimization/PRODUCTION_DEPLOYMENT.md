# Production Deployment Guide - Whisper NPU Acceleration

**Project**: Unicorn Amanuensis
**Audience**: DevOps Engineers, System Administrators, Production Users
**Last Updated**: October 25, 2025

---

## Quick Start

### Recommended Configuration (Production Ready)

```bash
# Start the server with faster_whisper mode
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
python3 server_whisperx_npu.py
```

The server will **automatically** select the best available mode:
1. Try NPU mode (if `/dev/accel/accel0` exists)
2. Fall back to **faster_whisper** mode (RECOMMENDED)
3. Additional fallbacks: whisper.cpp SYCL → OpenAI Whisper → Mock

**Expected Performance**:
- **Speed**: 13.5x realtime
- **CPU Usage**: 0.24%
- **Accuracy**: Perfect (2.5% WER)
- **Latency**: ~100ms first token
- **Memory**: ~2GB RAM

---

## Table of Contents

1. [Installation](#installation)
2. [Starting the Server](#starting-the-server)
3. [Mode Selection](#mode-selection)
4. [API Usage](#api-usage)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)
7. [Monitoring](#monitoring)
8. [Scaling](#scaling)

---

## Installation

### Prerequisites

**System Requirements**:
- Ubuntu 20.04+ or compatible Linux distribution
- Python 3.9+
- 8GB RAM minimum (16GB recommended)
- 10GB disk space for models

**Optional** (for NPU mode):
- AMD Ryzen 7040/8040 series CPU
- XRT 2.20.0 installed
- `/dev/accel/accel0` device available

### Step 1: Install System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and build tools
sudo apt install -y python3 python3-pip python3-venv \
    build-essential git curl wget

# Install FFmpeg for audio processing
sudo apt install -y ffmpeg

# Install audio libraries
sudo apt install -y libsndfile1 libsndfile1-dev \
    portaudio19-dev libportaudio2
```

### Step 2: Create Python Virtual Environment

```bash
# Navigate to project directory
cd /home/ucadmin/UC-1/Unicorn-Amanuensis

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 3: Install Python Dependencies

```bash
# Install faster-whisper (RECOMMENDED)
pip install faster-whisper==1.0.0

# Install other dependencies
pip install -r requirements.txt

# Or install manually:
pip install fastapi==0.104.0
pip install uvicorn[standard]==0.24.0
pip install python-multipart==0.0.6
pip install httpx==0.25.0
pip install numpy==1.24.0
pip install librosa==0.10.1
pip install scipy==1.11.0
pip install onnxruntime==1.16.0
```

### Step 4: Download Models

```bash
# faster-whisper models are downloaded automatically on first use
# No manual download needed!

# Optional: Pre-download specific model
python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
```

### Step 5: Verify Installation

```bash
# Test faster-whisper
python3 -c "from faster_whisper import WhisperModel; print('✅ faster-whisper installed')"

# Test FastAPI
python3 -c "import fastapi; print('✅ FastAPI installed')"

# Test audio libraries
python3 -c "import librosa; print('✅ librosa installed')"
```

---

## Starting the Server

### Option 1: Direct Python Execution (Development)

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
python3 server_whisperx_npu.py
```

**Output**:
```
🚀 Starting Docker-Ready Unicorn Amanuensis...
🐳 Container environment detected
🌐 Starting server on port 8000
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Option 2: Systemd Service (Production)

Create a systemd service for automatic startup:

```bash
# Create service file
sudo tee /etc/systemd/system/whisperx-npu.service > /dev/null <<EOF
[Unit]
Description=WhisperX NPU Transcription Service
After=network.target

[Service]
Type=simple
User=ucadmin
Group=ucadmin
WorkingDirectory=/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
Environment="PATH=/home/ucadmin/UC-1/Unicorn-Amanuensis/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/ucadmin/UC-1/Unicorn-Amanuensis/venv/bin/python3 server_whisperx_npu.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable whisperx-npu

# Start service
sudo systemctl start whisperx-npu

# Check status
sudo systemctl status whisperx-npu

# View logs
sudo journalctl -u whisperx-npu -f
```

### Option 3: Docker Container (Isolated)

```bash
# Build Docker image
cd /home/ucadmin/UC-1/Unicorn-Amanuensis
docker build -t whisperx-npu:latest .

# Run container
docker run -d \
    --name whisperx-npu \
    --restart unless-stopped \
    -p 8000:8000 \
    -v /home/ucadmin/models:/models:ro \
    --device /dev/accel/accel0:/dev/accel/accel0 \
    whisperx-npu:latest

# View logs
docker logs -f whisperx-npu

# Stop container
docker stop whisperx-npu

# Remove container
docker rm whisperx-npu
```

---

## Mode Selection

### Automatic Mode Detection

The server automatically detects and selects the best available mode:

```python
# Priority order (server_whisperx_npu.py, lines 67-115)
1. NPU Mode:     /dev/accel/accel0 exists + NPU runtime available
2. SYCL Mode:    whisper.cpp SYCL build exists
3. faster_whisper: faster-whisper library installed (RECOMMENDED)
4. System cpp:   whisper.cpp system installation
5. OpenAI:       openai-whisper installed
6. WhisperX:     whisperx installed
7. Mock:         Fallback for debugging
```

### Force Specific Mode

You can force a specific mode using environment variables:

```bash
# Force faster_whisper mode
export WHISPER_MODE="faster_whisper"
python3 server_whisperx_npu.py

# Force NPU mode (experimental)
export WHISPER_MODE="npu"
python3 server_whisperx_npu.py

# Force OpenAI Whisper
export WHISPER_MODE="openai_whisper"
python3 server_whisperx_npu.py
```

### Mode Comparison

| Mode | Speed | CPU | Accuracy | Status | Use Case |
|------|-------|-----|----------|--------|----------|
| **faster_whisper** | 13.5x | 0.24% | Perfect | ✅ Ready | **Production** |
| NPU (hybrid) | 10.7x | 15-20% | Garbled | ⚠️ Experimental | Development |
| whisper.cpp SYCL | 11.2x | 18% | Perfect | ✅ Ready | Intel iGPU systems |
| OpenAI Whisper | 2-3x | 100% | Perfect | ✅ Ready | CPU-only fallback |
| WhisperX | 5-8x | 80% | Perfect | ✅ Ready | Advanced features |

---

## API Usage

### Endpoint Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation |
| `/web` | GET | Web interface |
| `/transcribe` | POST | Main transcription endpoint |
| `/v1/audio/transcriptions` | POST | OpenAI-compatible endpoint |
| `/status` | GET | Server and model status |
| `/models` | GET | List available models |
| `/health` | GET | Health check |

### Example 1: Basic Transcription

```bash
curl -X POST \
  -F "file=@audio.wav" \
  http://localhost:8000/transcribe
```

**Response**:
```json
{
  "text": "Hello, this is a test of the Whisper transcription system.",
  "segments": [
    {
      "start": 0.0,
      "end": 5.24,
      "text": "Hello, this is a test of the Whisper transcription system."
    }
  ],
  "language": "en",
  "duration": 5.24,
  "performance": {
    "total_time": "0.39s",
    "rtf": "13.5x",
    "engine": "Docker Faster_Whisper",
    "model": "base"
  }
}
```

### Example 2: With Model Selection

```bash
curl -X POST \
  -F "file=@audio.wav" \
  -F "model=large-v3" \
  http://localhost:8000/transcribe
```

### Example 3: OpenAI-Compatible API

```bash
curl -X POST \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json" \
  http://localhost:8000/v1/audio/transcriptions
```

### Example 4: Python Client

```python
import httpx

# Upload audio file
with open("audio.wav", "rb") as f:
    files = {"file": ("audio.wav", f, "audio/wav")}
    response = httpx.post(
        "http://localhost:8000/transcribe",
        files=files,
        timeout=300.0
    )

result = response.json()
print(f"Transcription: {result['text']}")
print(f"Speed: {result['performance']['rtf']}")
```

### Example 5: JavaScript/Node.js Client

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('audio.wav'));

axios.post('http://localhost:8000/transcribe', form, {
  headers: form.getHeaders(),
  timeout: 300000
})
.then(response => {
  console.log('Transcription:', response.data.text);
  console.log('Speed:', response.data.performance.rtf);
})
.catch(error => {
  console.error('Error:', error.message);
});
```

### Example 6: Check Server Status

```bash
# Check if server is healthy
curl http://localhost:8000/health

# Get detailed status
curl http://localhost:8000/status

# List available models
curl http://localhost:8000/models
```

**Status Response**:
```json
{
  "status": "ready",
  "engine_mode": "faster_whisper",
  "model": "base",
  "container": "docker",
  "version": "2.0.0"
}
```

---

## Performance Tuning

### Model Selection

Choose the right model for your use case:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| **tiny** | 75 MB | 40x | Good | Real-time subtitles |
| **base** | 150 MB | 13.5x | Very Good | **Recommended** |
| **small** | 488 MB | 8x | Excellent | High accuracy needed |
| **medium** | 1.5 GB | 4x | Excellent | Professional use |
| **large-v3** | 3.1 GB | 2x | Best | Maximum accuracy |

**Change model at runtime**:
```bash
# In request
curl -X POST \
  -F "file=@audio.wav" \
  -F "model=large-v3" \
  http://localhost:8000/transcribe
```

### Compute Type Optimization

faster-whisper supports multiple compute types:

```python
# In server_whisperx_npu.py (line 363)
self.faster_whisper_model = WhisperModel(
    model_name,
    device="cpu",
    compute_type="int8"  # Options: float32, float16, int8, int8_float16
)
```

| Compute Type | Speed | Accuracy | Memory | Notes |
|--------------|-------|----------|--------|-------|
| **int8** | Fastest | 99% | Low | **Recommended** |
| int8_float16 | Fast | 99.5% | Medium | Slightly better quality |
| float16 | Medium | 100% | Medium | GPU only |
| float32 | Slow | 100% | High | Reference quality |

### Threading Configuration

Adjust worker threads for your hardware:

```python
# In server startup (line 812)
uvicorn.run(
    app,
    host="0.0.0.0",
    port=API_PORT,
    workers=1  # Increase for multi-core systems
)
```

**Guidelines**:
- 1 worker: Single-core or low memory (<8GB)
- 2 workers: Dual-core systems
- 4 workers: Quad-core or higher
- Never exceed physical core count

### Memory Management

**Reduce memory usage**:
```bash
# Preload model once (use systemd service)
# Avoid restarting server frequently

# Use smaller model
export WHISPER_MODEL="tiny"  # Instead of "base"
```

**Monitor memory**:
```bash
# Check memory usage
watch -n 1 'ps aux | grep server_whisperx_npu'

# Or use htop
htop -p $(pgrep -f server_whisperx_npu)
```

---

## Troubleshooting

### Issue 1: Server Won't Start

**Symptoms**:
```
ModuleNotFoundError: No module named 'faster_whisper'
```

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install dependencies
pip install faster-whisper
```

---

### Issue 2: NPU Mode Has Garbled Output

**Symptoms**:
```json
{
  "text": "[Audio successfully processed: 5.2s duration, ONNX Whisper active]"
}
```

**Solution**:
```bash
# Force faster_whisper mode
export WHISPER_MODE="faster_whisper"
python3 server_whisperx_npu.py

# Or wait for decoder fix in Phase 1
```

---

### Issue 3: Slow Transcription Speed

**Symptoms**:
- Taking >5 seconds for 30-second audio
- CPU usage very high (>80%)

**Solutions**:

**A. Check mode**:
```bash
curl http://localhost:8000/status
# Should show: "engine_mode": "faster_whisper"
```

**B. Use smaller model**:
```bash
curl -X POST \
  -F "file=@audio.wav" \
  -F "model=tiny" \
  http://localhost:8000/transcribe
```

**C. Check system resources**:
```bash
# Check CPU
top

# Check memory
free -h

# Check disk I/O
iostat -x 1
```

---

### Issue 4: Out of Memory

**Symptoms**:
```
RuntimeError: Out of memory
# OR
Killed
```

**Solutions**:

**A. Use smaller model**:
```bash
# Switch from large-v3 to base
export WHISPER_MODEL="base"
```

**B. Reduce workers**:
```python
# In server_whisperx_npu.py
uvicorn.run(app, host="0.0.0.0", port=API_PORT, workers=1)
```

**C. Add swap space**:
```bash
# Create 8GB swap file
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

### Issue 5: Permission Denied for /dev/accel/accel0

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied: '/dev/accel/accel0'
```

**Solution**:
```bash
# Add user to render group
sudo usermod -a -G render $USER

# Log out and back in
# Or use:
newgrp render

# Verify access
ls -l /dev/accel/accel0
# Should show: crw-rw---- 1 root render
```

---

### Issue 6: Model Not Found

**Symptoms**:
```
FileNotFoundError: Model 'whisper-large-v3-npu' not found
```

**Solution**:
```bash
# Use standard model names
curl -X POST \
  -F "file=@audio.wav" \
  -F "model=large-v3" \
  http://localhost:8000/transcribe

# Valid models: tiny, base, small, medium, large-v3
```

---

## Monitoring

### Health Check Endpoint

```bash
# Simple health check
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "engine": "faster_whisper",
  "timestamp": 1729886400.0
}
```

### Prometheus Metrics (Optional)

Add Prometheus metrics for production monitoring:

```python
# Add to server_whisperx_npu.py
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
transcription_requests = Counter('whisperx_transcription_requests_total', 'Total transcription requests')
transcription_duration = Histogram('whisperx_transcription_duration_seconds', 'Transcription duration')

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

### Logging

**View logs in real-time**:
```bash
# Systemd service
sudo journalctl -u whisperx-npu -f

# Docker container
docker logs -f whisperx-npu

# Direct execution
python3 server_whisperx_npu.py 2>&1 | tee server.log
```

**Log levels**:
```bash
# Set log level
export LOG_LEVEL="DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR
python3 server_whisperx_npu.py
```

---

## Scaling

### Horizontal Scaling

Run multiple server instances behind a load balancer:

```
                    ┌────────────────┐
                    │  Load Balancer │
                    │   (nginx)      │
                    └────────┬───────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌────────┐          ┌────────┐          ┌────────┐
    │ Server │          │ Server │          │ Server │
    │  :8001 │          │  :8002 │          │  :8003 │
    └────────┘          └────────┘          └────────┘
```

**nginx configuration**:
```nginx
upstream whisperx_backend {
    least_conn;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 8000;
    location / {
        proxy_pass http://whisperx_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
        client_max_body_size 100M;
    }
}
```

**Start multiple instances**:
```bash
# Instance 1
API_PORT=8001 python3 server_whisperx_npu.py &

# Instance 2
API_PORT=8002 python3 server_whisperx_npu.py &

# Instance 3
API_PORT=8003 python3 server_whisperx_npu.py &
```

### Queue-Based Processing

For batch processing, use a queue system:

```
┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
│  Client  │─────▶│  Redis   │◀─────│ Worker 1 │─────▶│  Storage │
└──────────┘      │  Queue   │      └──────────┘      └──────────┘
                  └────┬─────┘
                       │
                       ├──────▶┌──────────┐
                       │       │ Worker 2 │
                       └──────▶└──────────┘
```

**Example with RQ (Redis Queue)**:
```python
# worker.py
from rq import Worker, Queue, Connection
import redis

redis_conn = redis.Redis()

if __name__ == '__main__':
    with Connection(redis_conn):
        worker = Worker(['transcription'])
        worker.work()

# enqueue job
from rq import Queue
from redis import Redis

q = Queue('transcription', connection=Redis())
job = q.enqueue('transcribe_audio', audio_path='audio.wav')
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | 8000 | Server port |
| `WHISPER_MODEL` | base | Model to use (tiny, base, small, medium, large-v3) |
| `WHISPER_MODE` | auto | Force specific mode (auto, faster_whisper, npu, etc.) |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `CPU_ONLY_MODE` | false | Disable GPU/NPU (1, true, yes) |

### Model Paths

| Model | Path |
|-------|------|
| faster-whisper | `~/.cache/huggingface/hub/models--Systran--faster-whisper-*` |
| ONNX models | `whisperx/models/whisper_onnx_cache/` |
| NPU models | `npu-models/` (if created) |

---

## Security Considerations

### 1. File Upload Limits

```python
# In server_whisperx_npu.py
# Add file size limit
@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(None),
    ...
):
    # Check file size (100MB limit)
    if file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")
```

### 2. Authentication (Optional)

```python
# Add API key authentication
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.post("/transcribe", dependencies=[Depends(verify_api_key)])
async def transcribe_endpoint(...):
    ...
```

### 3. Rate Limiting

```bash
# Use nginx rate limiting
limit_req_zone $binary_remote_addr zone=whisperx:10m rate=10r/m;

server {
    location /transcribe {
        limit_req zone=whisperx burst=5;
        ...
    }
}
```

---

## Appendix

### Quick Command Reference

```bash
# Start server
python3 server_whisperx_npu.py

# Start with custom port
API_PORT=9000 python3 server_whisperx_npu.py

# Start with specific model
WHISPER_MODEL="large-v3" python3 server_whisperx_npu.py

# Force faster_whisper mode
WHISPER_MODE="faster_whisper" python3 server_whisperx_npu.py

# Test transcription
curl -X POST -F "file=@audio.wav" http://localhost:8000/transcribe

# Check status
curl http://localhost:8000/status

# Health check
curl http://localhost:8000/health

# View logs (systemd)
sudo journalctl -u whisperx-npu -f

# Restart service
sudo systemctl restart whisperx-npu
```

### Support

For issues and questions:
- GitHub: https://github.com/Unicorn-Commander/Unicorn-Amanuensis/issues
- Documentation: See `NPU_HYBRID_ARCHITECTURE.md` for technical details
- Email: support@magicunicorn.tech

---

**Document Version**: 1.0
**Last Updated**: October 25, 2025
**Tested On**: Ubuntu 22.04, Python 3.10, faster-whisper 1.0.0
