# 🦄 Unicorn Amanuensis

<div align="center">
  <img src="assets/unicorn-amanuensis-logo.png" alt="Unicorn Amanuensis Logo" width="200">
  
  **Professional AI-Powered Transcription Service**
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
  [![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
  
  *High-accuracy speech-to-text with speaker diarization and hardware acceleration*
</div>

---

## 🌟 Overview

Unicorn Amanuensis is a professional transcription service powered by WhisperX, offering state-of-the-art speech recognition with advanced features like word-level timestamps and speaker diarization. Designed for both API integration and standalone use, it provides OpenAI-compatible endpoints for seamless integration with existing applications.

### ✨ Key Features

- **🎯 High Accuracy** - Powered by WhisperX with OpenAI Whisper models
- **👥 Speaker Diarization** - Identify who said what in multi-speaker audio
- **⏱️ Word-Level Timestamps** - Precise timing for each word
- **🌍 100+ Languages** - Automatic language detection and transcription
- **🚀 Hardware Acceleration** - Support for CPU, NVIDIA GPU, AMD NPU, and Intel iGPU
- **🔌 OpenAI Compatible** - Drop-in replacement for OpenAI's Whisper API
- **🎨 Simple Web Interface** - Basic UI for quick transcriptions
- **🐳 Docker Ready** - Easy deployment with Docker Compose

## 📋 Prerequisites

- Docker and Docker Compose
- 4GB+ RAM (8GB+ recommended for larger models)
- (Optional) NVIDIA GPU for CUDA acceleration
- (Optional) AMD Ryzen AI processor for NPU acceleration
- (Optional) Intel iGPU for OpenVINO acceleration

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Unicorn-Commander/Unicorn-Amanuensis.git
cd Unicorn-Amanuensis
```

### 2. Configure Environment

```bash
cp .env.template .env
# Edit .env with your settings
nano .env
```

### 3. Install with Hardware Detection

```bash
./install.sh
# Automatically detects and configures for your hardware
```

### 4. Access the Service

- **API Endpoint**: http://localhost:9000
- **Web Interface**: http://localhost:9001
- **API Documentation**: http://localhost:9000/docs

## 🎙️ API Usage

### OpenAI-Compatible Endpoint

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",
    base_url="http://localhost:9000/v1"
)

audio_file = open("meeting.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="json"
)
print(transcript.text)
```

### With Speaker Diarization

```bash
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@meeting.wav" \
  -F "diarize=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4"
```

### Response Format Options

- `json` - Structured JSON with metadata
- `text` - Plain text transcript
- `srt` - SubRip subtitle format
- `vtt` - WebVTT subtitle format
- `verbose_json` - Detailed JSON with word-level timestamps

## 🔧 Hardware Support

### Automatic Detection
The installer automatically detects and configures for available hardware:

| Hardware | Performance | Use Case |
|----------|------------|----------|
| **AMD NPU** | ~5x realtime | Ryzen AI laptops (7040/8040 series) |
| **Intel iGPU** | ~4x realtime | Intel Arc/Iris Xe graphics |
| **NVIDIA GPU** | ~10x realtime | Dedicated GPU systems |
| **CPU** | ~2x realtime | Universal fallback |

### Manual Configuration

```bash
# Force specific backend
./install.sh --backend=npu

# Disable diarization for better performance
./install.sh --variant=lite
```

## 📊 Model Selection

| Model | Size | Accuracy | Speed | Memory |
|-------|------|----------|-------|--------|
| `tiny` | 74M | Good | Fastest | 1GB |
| `base` | 139M | Better | Fast | 1GB |
| `small` | 483M | Great | Balanced | 2GB |
| `medium` | 1.5GB | Excellent | Moderate | 5GB |
| `large-v3` | 3GB | Best | Slow | 10GB |

## 🌐 Web Interface

Access the simple web interface at http://localhost:9001

Features:
- Drag-and-drop file upload
- Real-time transcription progress
- Speaker identification display
- Export options (TXT, SRT, VTT, JSON)
- Search within transcripts

## 🔌 Integration Examples

### Open-WebUI Integration

Add to your Open-WebUI `.env`:
```env
AUDIO_STT_ENGINE=openai
AUDIO_STT_OPENAI_API_KEY=dummy-key
AUDIO_STT_OPENAI_API_BASE_URL=http://localhost:9000/v1
AUDIO_STT_MODEL=whisper-1
```

### Python SDK

```python
import requests

# Simple transcription
with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:9000/v1/audio/transcriptions",
        files={"file": f},
        data={"model": "whisper-1"}
    )
    print(response.json()["text"])
```

### Batch Processing

```python
import glob
import asyncio
import aiohttp

async def transcribe_file(session, filepath):
    with open(filepath, 'rb') as f:
        data = aiohttp.FormData()
        data.add_field('file', f, filename=filepath)
        data.add_field('model', 'whisper-1')
        
        async with session.post('http://localhost:9000/v1/audio/transcriptions', data=data) as response:
            result = await response.json()
            return filepath, result['text']

async def batch_transcribe(files):
    async with aiohttp.ClientSession() as session:
        tasks = [transcribe_file(session, f) for f in files]
        return await asyncio.gather(*tasks)

# Process all MP3 files
files = glob.glob("recordings/*.mp3")
results = asyncio.run(batch_transcribe(files))
```

## 🛠️ Advanced Configuration

### Environment Variables

```env
# Model Configuration
WHISPER_MODEL=base              # Model size
WHISPER_LANGUAGE=auto           # Language code or 'auto'
WHISPER_DEVICE=auto             # cpu, cuda, npu, igpu, auto

# Diarization
ENABLE_DIARIZATION=true         # Enable speaker identification
HF_TOKEN=your_token_here       # Required for diarization

# Performance
BATCH_SIZE=16                   # Batch size for processing
NUM_WORKERS=4                   # Parallel processing threads
COMPUTE_TYPE=int8              # int8, float16, float32

# API Settings
API_KEY=your-api-key           # Optional API authentication
CORS_ORIGINS=*                 # CORS configuration
```

### Docker Compose Override

```yaml
# docker-compose.override.yml
services:
  whisperx:
    environment:
      - WHISPER_MODEL=large-v3
      - WHISPER_DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 📈 Performance Optimization

### For Speed
- Use smaller models (tiny, base)
- Enable batch processing
- Use hardware acceleration
- Disable diarization if not needed

### For Accuracy
- Use larger models (medium, large-v3)
- Enable VAD (Voice Activity Detection)
- Use beam search decoding
- Fine-tune on domain-specific data

## 🤝 API Compatibility

Unicorn Amanuensis is compatible with:
- OpenAI Whisper API
- Hugging Face ASR pipelines
- AssemblyAI format (with adapter)
- Google Speech-to-Text (with adapter)

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Base ASR model
- [WhisperX](https://github.com/m-bain/whisperX) - Enhanced features
- [Pyannote](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- The Unicorn Commander community

## 🔗 Related Projects

- [Unicorn Orator](https://github.com/Unicorn-Commander/Unicorn-Orator) - Text-to-Speech companion
- [UC-1 Pro](https://github.com/Unicorn-Commander/UC-1-Pro) - Complete AI infrastructure stack

---

<div align="center">
  Made with ❤️ by Unicorn Commander
  
  🦄 *Transcribe with Intelligence* 🦄
</div>