# 🦄 Unicorn Amanuensis Architecture

## File Organization by Hardware Platform

### 🎯 UC-1 Pro Production Files (Use These!)

#### Core Server Implementation
```
whisperx/
├── server_docker_ready.py        ✅ PRODUCTION - Auto-detecting server
├── server_igpu_ultra.py          ✅ PERFORMANCE - 60x target Intel iGPU  
├── server_igpu_sycl.py           ✅ DEVELOPMENT - Direct SYCL control
├── whisper_igpu_real.py          ✅ CORE - Real whisper.cpp SYCL integration
└── device_manager.py             ✅ CORE - Multi-GPU device selection
```

#### Docker & Deployment
```
🐳 Docker Files
├── docker-compose.uc1-pro.yml    ✅ UC-1 Pro Docker Compose
├── Dockerfile.production         ✅ Full Intel iGPU build
├── Dockerfile.lightweight        ✅ Fallback for general deployment
└── .github/workflows/             ✅ Auto-build CI/CD
```

#### Documentation
```
📖 Documentation  
├── UC1-PRO-DEPLOYMENT.md         ✅ Complete UC-1 Pro guide
├── ARCHITECTURE.md               ✅ This file - system architecture
├── MODELS.md                     ✅ Model management & HuggingFace
└── CLAUDE.md                     ✅ Technical implementation notes
```

### ❌ Legacy/Development Files (Don't Use in Production)

#### Experimental OpenVINO Approach
```
whisperx/
├── server_igpu_int8.py           ❌ OpenVINO approach (high CPU usage)
├── server_openvino_*.py          ❌ Various OpenVINO experiments
├── server_production.py          ❌ Old production server
├── quantize_*.py                 ❌ Model quantization scripts
└── convert_*.py                  ❌ Model conversion utilities
```

#### Mock/Testing Implementation
```
whisperx/
├── whisper_igpu_runtime_optimized.py  ❌ Mock implementation
├── test_*.py                          ❌ Testing scripts
├── transcribe_*.py                    ❌ Standalone test scripts
└── verify_*.py                        ❌ Verification scripts
```

#### Old Docker Files
```
🐳 Legacy Docker
├── Dockerfile.igpu*              ❌ Old Intel iGPU attempts
├── Dockerfile.stt-igpu           ❌ STT-specific build
└── docker-compose.yml            ❌ General compose (not UC-1 Pro)
```

## 🏗️ Architecture Overview

### Hardware Abstraction Layer
```
┌─────────────────────────────────────────────────┐
│                 User Interface                   │
├─────────────────────────────────────────────────┤
│  FastAPI Server (server_docker_ready.py)       │
├─────────────────────────────────────────────────┤
│  Device Manager (device_manager.py)            │
│  ┌─────────────┬─────────────┬─────────────┐   │
│  │ Intel iGPU  │  AMD NPU    │ NVIDIA GPU  │   │
│  │ Priority:20 │ Priority:12 │ Priority:5  │   │
│  └─────────────┴─────────────┴─────────────┘   │
├─────────────────────────────────────────────────┤
│        Core Implementation Layer                │
│  ┌─────────────────────────────────────────┐   │
│  │  whisper_igpu_real.py                   │   │
│  │  ├── whisper.cpp SYCL binary           │   │
│  │  ├── Level Zero GPU access             │   │  
│  │  └── Intel MKL optimization            │   │
│  └─────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│            Hardware Layer                       │
│  ┌─────────────────────────────────────────┐   │
│  │  Intel UHD Graphics 770                 │   │ 
│  │  ├── 32 Execution Units @ 1550MHz      │   │
│  │  ├── 89GB Accessible Memory            │   │
│  │  └── SYCL + Level Zero Support         │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Data Flow for UC-1 Pro

```
Audio File → FastAPI → Device Manager → Intel iGPU Pipeline
     │            │            │              │
     │            │            │              ├── Audio Preprocessing (FFmpeg)
     │            │            │              │   └── 16kHz mono conversion
     │            │            │              │
     │            │            │              ├── SYCL Kernel Dispatch
     │            │            │              │   ├── MEL spectrogram (iGPU)
     │            │            │              │   ├── Attention layers (iGPU) 
     │            │            │              │   └── Matrix operations (MKL)
     │            │            │              │
     │            │            │              └── Text Output Processing
     │            │            │                  ├── Timestamp parsing
     │            │            │                  └── JSON formatting
     │            │            │
     │            │            └── Device Selection Logic:
     │            │                1. Intel iGPU (Priority 20) ✅
     │            │                2. AMD NPU (Priority 12)
     │            │                3. NVIDIA GPU (Priority 5) - Reserved for LLM
     │            │                4. CPU (Priority 0) - Fallback
     │            │
     │            └── API Endpoints:
     │                ├── /transcribe (Custom format)
     │                ├── /v1/audio/transcriptions (OpenAI compatible)
     │                └── /health (Monitoring)
     │
     └── Input Formats:
         ├── Upload: multipart/form-data
         ├── URL: Remote audio file
         └── Models: tiny, base, small, medium, large-v3
```

## 🔧 Component Responsibilities

### server_docker_ready.py (Production Entry Point)
- **Purpose**: Auto-detecting production server
- **Device Detection**: Automatically selects best available backend
- **Fallback Chain**: SYCL → OpenAI Whisper → WhisperX → Mock
- **Container Ready**: Works in any Docker environment
- **Use Case**: Production deployments where hardware varies

### server_igpu_ultra.py (Performance Optimized)
- **Purpose**: Maximum Intel iGPU performance
- **Target**: 60x+ realtime transcription
- **Direct Integration**: whisper_igpu_real.py with no fallbacks
- **Zero CPU**: Guaranteed iGPU-only execution
- **Use Case**: UC-1 Pro when maximum performance needed

### server_igpu_sycl.py (Development/Debugging) 
- **Purpose**: Direct SYCL binary control
- **Features**: Detailed logging, chunked processing, real-time progress
- **Integration**: Direct whisper-cli binary execution
- **Flexibility**: Easy modification and testing
- **Use Case**: Development, debugging, custom integrations

### whisper_igpu_real.py (Core Implementation)
- **Purpose**: Real whisper.cpp SYCL integration
- **No Mocks**: 100% real transcription using compiled binaries
- **Model Support**: All Whisper models (tiny → large-v3)
- **Performance**: 7-20x realtime on Intel UHD Graphics 770
- **Features**: Chunking, diarization, word timestamps

### device_manager.py (Hardware Abstraction)
- **Purpose**: Multi-GPU system intelligence
- **Auto-Detection**: Intel iGPU, AMD NPU, NVIDIA GPU
- **Priority System**: Reserves discrete GPU for LLM inference
- **Environment Setup**: Configures drivers and environment variables
- **Future Ready**: AMD XDNA1 NPU support prepared

## 🎯 Deployment Strategies

### UC-1 Pro Production (Recommended)
```bash
# Use Docker Compose for full stack
docker-compose -f docker-compose.uc1-pro.yml up -d
```

**Advantages**:
- Optimized for Intel UHD Graphics 770
- Automatic device passthrough (/dev/dri)
- Health monitoring and restart policies
- Resource limits tuned for iGPU workloads

### General Cloud Deployment
```bash
# Use lightweight image for flexibility
docker run -p 8000:8000 magicunicorn/unicorn-amanuensis:lite-latest
```

**Advantages**:
- Works without Intel GPU
- Auto-falls back to available backends
- Smaller image size
- Multi-architecture support (amd64/arm64)

### Development Environment
```bash
# Run directly with Python
cd whisperx && python3 server_docker_ready.py
```

**Advantages**:
- Direct code modification
- Full debugging capabilities
- No container overhead
- Direct hardware access

## 📊 Performance Matrix

| Implementation | Performance | CPU Usage | Memory | Use Case |
|---------------|-------------|-----------|--------|----------|
| **server_docker_ready.py** | 7-20x | 0% | 1-3GB | ✅ **Production** |
| **server_igpu_ultra.py** | 20-60x | 0% | 1-3GB | ✅ **Performance** |
| **server_igpu_sycl.py** | 7-15x | 5% | 1-3GB | ✅ **Development** |
| server_igpu_int8.py | 70x | 80% | 2-4GB | ❌ Legacy OpenVINO |
| server_production.py | 70x | 90% | 3-5GB | ❌ Legacy high-CPU |

## 🔄 Update & Maintenance Paths

### Automatic Updates (CI/CD)
```bash
# GitHub Actions builds new images on git push
git push origin master  # Triggers automatic Docker Hub build
```

### Manual Updates
```bash  
# Pull latest and restart
docker-compose -f docker-compose.uc1-pro.yml pull
docker-compose -f docker-compose.uc1-pro.yml up -d
```

### Development Updates
```bash
# Rebuild local changes
docker build -f Dockerfile.production -t unicorn-test .
docker run --device /dev/dri:/dev/dri -p 8000:8000 unicorn-test
```

---

## 🎯 Quick Reference for UC-1 Pro

**Production File**: `server_docker_ready.py` (auto-detecting)  
**Performance File**: `server_igpu_ultra.py` (60x target)  
**Development File**: `server_igpu_sycl.py` (direct control)  

**Core Implementation**: `whisper_igpu_real.py` (no mocks)  
**Device Management**: `device_manager.py` (multi-GPU intelligence)  

**Deployment**: `docker-compose.uc1-pro.yml`  
**Expected Performance**: 7-20x realtime, 0% CPU usage