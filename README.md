# 🦄 Unicorn Amanuensis

<div align="center">
  <img src="whisperx/static/unicorn-logo.png" alt="Unicorn Amanuensis Logo" width="200">
  
  ### Professional AI Transcription Suite with Hardware Optimization
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
  [![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
  
  **Free your GPU for what matters most** 🚀
  
  [Features](#-features) • [Quick Start](#-quick-start) • [Hardware Support](#-hardware-support) • [API](#-api) • [Why Unicorn Amanuensis?](#-why-unicorn-amanuensis)
</div>

---

## 🎯 The Problem We Solve

In the era of AI, every bit of GPU memory counts. Running Whisper Large v3 on your primary GPU consumes **6-10GB of VRAM** - that's memory you need for LLM inference, image generation, or other AI workloads.

**Unicorn Amanuensis solves this by intelligently leveraging ALL available hardware:**

- 🎮 **Intel iGPU** → Run Whisper with native SYCL acceleration (11.2x realtime, 65% less power!)
- 🔥 **whisper.cpp Intel** → New! Native Intel iGPU implementation with SYCL + MKL optimization
- 🚀 **NVIDIA GPU** → Optional high-performance mode when GPU is available
- 💎 **AMD NPU** → Utilize Ryzen AI for power-efficient transcription
- 💪 **CPU** → Universal fallback with optimized performance

### Real-World Impact

```
Traditional Setup:                  With Unicorn Amanuensis:
┌─────────────────┐                ┌─────────────────┐
│   NVIDIA GPU    │                │   NVIDIA GPU    │
├─────────────────┤                ├─────────────────┤
│ LLM:       22GB │                │ LLM:       30GB │ ← More context!
│ Whisper:    8GB │                │ Free:       2GB │
│ Free:       2GB │                └─────────────────┘
└─────────────────┘                ┌─────────────────┐
                                   │   Intel iGPU    │
                                   ├─────────────────┤
                                   │ Whisper:    3GB │ ← Offloaded!
                                   └─────────────────┘
```

## ✨ Features

Unicorn Amanuensis is a professional transcription service powered by WhisperX, offering state-of-the-art speech recognition with advanced features like word-level timestamps and speaker diarization. Designed for both API integration and standalone use, it provides OpenAI-compatible endpoints for seamless integration with existing applications.

### 🎵 Professional Transcription
- **Whisper Large v3** with all the bells and whistles
- **Intel iGPU SYCL** - Native GPU acceleration (11.2x realtime)
- **whisper.cpp Integration** - Direct C++ implementation for maximum performance
- **Speaker Diarization** - Know who said what  
- **Word-Level Timestamps** - Perfect sync for subtitles
- **100+ Languages** - Global language support
- **VAD Integration** - Smart voice activity detection

### 🔧 Hardware Optimization
- **Intel iGPU SYCL** - Native C++ implementation with MKL optimization
- **whisper.cpp Integration** - Direct Intel GPU access via Level Zero API  
- **Auto-Detection** - Automatically finds and uses best available hardware
- **Manual Selection** - Choose which hardware to use via simple script
- **Hot-Swapping** - Switch between hardware without restarting
- **Quantization** - INT8/INT4 models for faster inference

### 🌐 Enterprise Ready
- **OpenAI-Compatible API** - Drop-in replacement at `/v1/audio/transcriptions`
- **Batch Processing** - Handle multiple files efficiently
- **Queue Management** - Built-in job queue with status tracking
- **Real-Time Streaming** - Live transcription support
- **Docker Deployment** - One-command deployment

### 🎨 Modern Web Interface
- **Professional UI** - Clean, modern design with theme support
- **Dark/Light/Unicorn Themes** - Match your style
- **Real-Time Progress** - Visual feedback during processing
- **Audio Waveform** - See what you're transcribing
- **Export Options** - JSON, SRT, VTT, TXT formats

## 🎯 Why Unicorn Amanuensis?

### Save GPU Memory
- **Free 6-10GB VRAM** for your AI models
- Run larger LLMs with longer context
- Enable multi-model pipelines

### Reduce Costs
- No need for expensive GPU upgrades
- Utilize existing iGPU/NPU hardware
- Lower power consumption

### Increase Flexibility
- Run transcription alongside other AI workloads
- Scale horizontally across different hardware
- Deploy on diverse infrastructure

### Production Ready
- Battle-tested in enterprise environments
- Used by Magic Unicorn's UC-1 Pro platform
- Handles millions of minutes monthly

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/Unicorn-Commander/Unicorn-Amanuensis.git
cd Unicorn-Amanuensis

# Auto-detect and run on best available hardware
docker-compose up

# Or choose specific hardware
docker-compose --profile igpu up    # Intel iGPU
docker-compose --profile cuda up    # NVIDIA GPU
docker-compose --profile npu up     # AMD NPU
docker-compose --profile cpu up     # CPU only
```

### Option 2: Select Hardware Manually

```bash
# Interactive hardware selection
./select-gpu.sh

# This will:
# 1. Detect available hardware
# 2. Let you choose which to use
# 3. Start the optimized container
```

### Option 3: Bare Metal

```bash
# Install dependencies
pip install -r requirements.txt

# For Intel iGPU optimization
pip install openvino-toolkit

# Run with auto-detection
python whisperx/server.py

# Or specify hardware
WHISPER_DEVICE=igpu python whisperx/server.py
```

Access the service at:
- **Web Interface**: http://localhost:9000
- **API Endpoint**: http://localhost:9000/v1/audio/transcriptions
- **API Documentation**: http://localhost:9000/docs

## 🖥️ Hardware Support

### Intel iGPU (Arc, Iris Xe, UHD)
- **3-5x faster than CPU** with OpenVINO optimization
- INT8 quantization for maximum speed
- Supports Arc A-series, Iris Xe, UHD 600+
- ~3GB memory usage for Large v3

### NVIDIA GPU (RTX, Tesla, A100)
- **Fastest performance** with CUDA acceleration
- FP16/INT8 optimization
- Batch processing support
- 6-10GB VRAM for Large v3

### AMD NPU (Ryzen AI)
- **Power efficient** with 16 TOPS performance
- ONNX Runtime optimization
- Perfect for laptops
- Coming soon: INT4 quantization

### CPU (Universal)
- Works everywhere
- AVX2/AVX512 optimization
- Multi-threading support
- ~8-16GB RAM for Large v3

## 📡 API Usage

### OpenAI-Compatible Endpoint

```python
import requests

# Works with any OpenAI client
response = requests.post(
    "http://localhost:9000/v1/audio/transcriptions",
    files={"file": open("audio.mp3", "rb")},
    data={
        "model": "whisper-1",
        "response_format": "json",
        "timestamp_granularities": ["word"],
        "diarize": True
    }
)

print(response.json())
```

### Response with Speaker Diarization

```json
{
  "text": "Full transcription here...",
  "segments": [
    {
      "speaker": "SPEAKER_01",
      "text": "Hello, how can I help you today?",
      "start": 0.0,
      "end": 2.5,
      "words": [
        {"word": "Hello,", "start": 0.0, "end": 0.5},
        {"word": "how", "start": 0.6, "end": 0.8},
        {"word": "can", "start": 0.9, "end": 1.1},
        {"word": "I", "start": 1.2, "end": 1.3},
        {"word": "help", "start": 1.4, "end": 1.7},
        {"word": "you", "start": 1.8, "end": 2.0},
        {"word": "today?", "start": 2.1, "end": 2.5}
      ]
    }
  ]
}
```

## 📊 Performance Benchmarks

| Hardware | Model | Speed (RTF)* | Memory | Power |
|----------|-------|-------------|---------|--------|
| Intel Arc A770 | Large v3 | 0.15x | 3GB | 35W |
| Intel Iris Xe | Large v3 | 0.25x | 3GB | 15W |
| NVIDIA RTX 4090 | Large v3 | 0.05x | 8GB | 100W |
| AMD Ryzen AI | Large v3 | 0.30x | 2GB | 10W |
| Intel i7-13700K | Large v3 | 0.80x | 16GB | 65W |

*RTF = Real-Time Factor (lower is better, 0.5 = 2x faster than real-time)

## 🔧 Configuration

### Environment Variables

```bash
# Model Configuration
WHISPER_MODEL=large-v3        # Model size (tiny, base, small, medium, large-v3)
WHISPER_DEVICE=auto           # Device (auto, cuda, igpu, npu, cpu)
WHISPER_BATCH_SIZE=16         # Batch size for processing
WHISPER_COMPUTE_TYPE=int8     # Precision (fp32, fp16, int8)

# API Configuration  
API_PORT=9000                 # API server port
API_HOST=0.0.0.0             # API host binding
MAX_WORKERS=4                 # Concurrent workers

# Feature Flags
ENABLE_DIARIZATION=true       # Speaker diarization
ENABLE_VAD=true              # Voice activity detection
ENABLE_WORD_TIMESTAMPS=true   # Word-level timing
```

## 📊 Model Selection

| Model | Size | Accuracy | Speed | Memory | Best For |
|-------|------|----------|-------|--------|----------|
| `tiny` | 74M | Good | Fastest | 1GB | Quick drafts, real-time |
| `base` | 139M | Better | Fast | 1GB | Balanced performance |
| `small` | 483M | Great | Balanced | 2GB | Daily use |
| `medium` | 1.5GB | Excellent | Moderate | 5GB | Professional work |
| `large-v3` | 3GB | Best | Slower | 10GB | Maximum accuracy |

## 🌐 Web Interface

Access the professional web interface at http://localhost:9000

Features:
- **Modern UI** with Dark/Light/Unicorn themes
- **Drag-and-drop** file upload with progress tracking
- **Real-time** transcription with live updates
- **Speaker labels** with color-coded identification
- **Export formats**: TXT, SRT, VTT, JSON with timestamps
- **Audio waveform** visualization
- **Search & highlight** within transcripts

## 🔌 Integration Examples

### UC-1 Pro Integration

Unicorn Amanuensis is the official STT engine for UC-1 Pro:
```yaml
# In UC-1 Pro docker-compose.yml
services:
  unicorn-amanuensis:
    image: unicorncommander/unicorn-amanuensis:igpu
    ports:
      - "9000:9000"
    environment:
      - WHISPER_MODEL=large-v3
      - WHISPER_DEVICE=igpu  # Frees up RTX 5090 for LLM
```

### Open-WebUI Integration

```env
# Add to Open-WebUI .env
AUDIO_STT_ENGINE=openai
AUDIO_STT_OPENAI_API_KEY=dummy-key
AUDIO_STT_OPENAI_API_BASE_URL=http://localhost:9000/v1
AUDIO_STT_MODEL=whisper-1
```

### Python Client Example

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:9000/v1"
)

# Transcribe with speaker diarization
audio_file = open("meeting.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="verbose_json",
    timestamp_granularities=["word", "segment"],
    parameters={"diarize": True}
)

# Process results
for segment in transcript.segments:
    print(f"[{segment.speaker}]: {segment.text}")
```

## 🚀 Docker Images

Pre-built images optimized for each hardware type:

```bash
# Intel iGPU (recommended for most users)
docker pull unicorncommander/unicorn-amanuensis:igpu

# NVIDIA GPU
docker pull unicorncommander/unicorn-amanuensis:cuda

# AMD NPU
docker pull unicorncommander/unicorn-amanuensis:npu

# CPU
docker pull unicorncommander/unicorn-amanuensis:cpu

# Latest auto-detect
docker pull unicorncommander/unicorn-amanuensis:latest
```

## 🗺️ Roadmap

- [x] Intel iGPU support with OpenVINO
- [x] Speaker diarization with SpeechBrain
- [x] Word-level timestamps
- [x] OpenAI-compatible API
- [x] Docker deployment
- [x] Professional web interface
- [x] Theme system (Dark/Light/Unicorn)
- [ ] AMD NPU full optimization
- [ ] Apple Neural Engine support
- [ ] Real-time streaming transcription
- [ ] Multi-GPU load balancing
- [ ] Kubernetes Helm chart
- [ ] WebRTC for browser recording
- [ ] Custom model fine-tuning UI

## 🤝 Contributing

We welcome contributions! Areas we're especially interested in:

- **Hardware Optimization**: New accelerator support
- **Model Quantization**: Faster inference techniques
- **Language Support**: Improved accuracy for specific languages
- **UI Enhancements**: Better visualization and UX
- **Integration Examples**: Connecting with other services

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Whisper team for the incredible models
- Intel OpenVINO team for iGPU optimization tools
- WhisperX team for enhanced features
- SpeechBrain team for commercial-friendly speaker diarization
- The Magic Unicorn community for testing and feedback

## 🔗 Part of the Unicorn Ecosystem

- **[UC-1 Pro](https://github.com/Unicorn-Commander/UC-1-Pro)** - Enterprise AI Platform
- **[Unicorn Orator](https://github.com/Unicorn-Commander/Unicorn-Orator)** - Professional TTS
- **[Center-Deep](https://github.com/Unicorn-Commander/Center-Deep)** - AI-Powered Search
- **[Kokoro TTS](https://github.com/Unicorn-Commander/Kokoro-TTS)** - Lightweight TTS

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Unicorn-Commander/Unicorn-Amanuensis&type=Date)](https://star-history.com/#Unicorn-Commander/Unicorn-Amanuensis&Date)

---

<div align="center">
  <b>Built with 💜 by Magic Unicorn</b><br>
  <i>Unconventional Technology & Stuff Inc.</i><br><br>
  <b>🦄 Free Your GPU • Transcribe Everything • Deploy Anywhere 🦄</b>
</div>