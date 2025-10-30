# Unicorn-Amanuensis

Multi-platform Speech-to-Text (STT) service with automatic NPU acceleration support.

## Overview

Unicorn-Amanuensis automatically detects and uses the best available compute backend:

- **XDNA2**: Strix Point NPU (AMD Ryzen AI 300 series) - *Under development*
- **XDNA1**: Phoenix/Hawk Point NPU (AMD Ryzen 7040/8040 series) - *Current implementation*
- **CPU**: Software fallback for systems without NPU

## Features

- High-accuracy transcription using WhisperX
- Word-level timestamps
- Speaker diarization (requires HF_TOKEN)
- Automatic platform detection
- NPU acceleration when available
- Batch processing for efficiency

## Architecture

```
Unicorn-Amanuensis/
├── api.py                    # Main entry point with platform detection
├── runtime/
│   └── platform_detector.py  # Auto-detects NPU/CPU
├── xdna1/                    # Phoenix/Hawk Point implementation
│   ├── server.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── docker-compose.yml
├── xdna2/                    # Strix Point implementation (WIP)
│   ├── kernels/              # Custom NPU kernels
│   ├── runtime/              # XDNA2 runtime integration
│   └── README_XDNA2.md
└── cpu/                      # CPU fallback (planned)
```

## Quick Start

### Using Docker

```bash
cd xdna1
docker-compose up
```

### Standalone

```bash
pip install -r xdna1/requirements.txt
python api.py
```

## API Endpoints

### POST /v1/audio/transcriptions

Transcribe audio file with optional speaker diarization.

**Parameters:**
- `file`: Audio file (WAV, MP3, etc.)
- `diarize`: Enable speaker diarization (default: false)
- `min_speakers`: Minimum number of speakers (optional)
- `max_speakers`: Maximum number of speakers (optional)

**Example:**

```bash
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "diarize=false"
```

**Response:**

```json
{
  "text": "Full transcription text...",
  "segments": [...],
  "language": "en",
  "words": [...]
}
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:9000/health
```

### GET /platform

Get current platform and backend information.

```bash
curl http://localhost:9000/platform
```

## Environment Variables

### Model Configuration
- `WHISPER_MODEL`: Model size (tiny, base, small, medium, large) - default: `base`
- `COMPUTE_TYPE`: Computation type (int8, float16) - default: auto-detected
- `BATCH_SIZE`: Batch size for processing - default: `16`
- `HF_TOKEN`: Hugging Face token for diarization models

### Platform Override
- `NPU_PLATFORM`: Force specific platform (xdna1, xdna2, cpu)

## Platform Support Status

| Platform | Status | Notes |
|----------|--------|-------|
| XDNA1 (Phoenix/Hawk Point) | ✅ Production | Fully tested |
| XDNA2 (Strix Point) | 🚧 Development | Kernel development in progress |
| CPU | ✅ Fallback | Uses XDNA1 backend in CPU mode |

## Development

### Adding XDNA2 Support

XDNA2 implementation is tracked in `xdna2/README_XDNA2.md`. Key components:

1. **Custom Kernels**: INT8 quantized operations for NPU
2. **Runtime Integration**: XDNA2 device management
3. **Model Optimization**: NPU-specific model transformations

### Testing Platform Detection

```python
from runtime.platform_detector import get_platform_info

info = get_platform_info()
print(info)
# Output: {'platform': 'xdna1', 'backend_path': 'xdna1', 'has_npu': True, 'npu_generation': 'XDNA1'}
```

## Integration with CC-1L

This repository is designed to be used as a Git submodule in the CC-1L project:

```bash
cd CC-1L
git submodule add https://github.com/Unicorn-Commander/Unicorn-Amanuensis.git npu-services/unicorn-amanuensis
```

## License

Proprietary - Unicorn Commander Project

## Related Projects

- **Unicorn-Orator**: TTS service (sister project)
- **unicorn-cpu-core**: Shared utilities library
- **CC-1L**: Main integration project
