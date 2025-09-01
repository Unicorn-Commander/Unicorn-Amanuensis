# 🦄 UC-1 Pro Deployment Guide

## Hardware Platform: UC-1 Pro Specifications

This guide is specifically for the **UC-1 Pro** hardware platform with:
- **CPU**: Intel Core processor with UHD Graphics 770
- **iGPU**: Intel UHD Graphics 770 (32 EUs @ 1550MHz, 89GB accessible memory)  
- **Software**: Ubuntu 22.04 + Intel oneAPI toolkit
- **Performance Target**: 7-20x realtime transcription with zero CPU usage

## 📁 File Structure for UC-1 Pro

### Production Files (Use These!)
```
📦 UC-1 Pro Deployment Files
├── 🐳 Docker & Compose
│   ├── docker-compose.uc1-pro.yml        # UC-1 Pro Docker Compose
│   ├── Dockerfile.production              # Full Intel iGPU build
│   └── Dockerfile.lightweight            # Fallback version
│
├── 🚀 Server Implementation
│   ├── whisperx/server_docker_ready.py   # ✅ Production server (auto-detecting)
│   ├── whisperx/server_igpu_ultra.py     # ✅ Intel iGPU optimized
│   ├── whisperx/server_igpu_sycl.py      # ✅ Direct SYCL implementation
│   └── whisperx/device_manager.py        # ✅ Multi-GPU device selection
│
├── 🔧 Core Implementation  
│   ├── whisperx/whisper_igpu_real.py     # ✅ Real whisper.cpp SYCL integration
│   └── /tmp/whisper.cpp/                 # ✅ SYCL-compiled whisper.cpp binary
│
└── 📖 Documentation
    ├── UC1-PRO-DEPLOYMENT.md             # This file
    ├── MODELS.md                          # Model management
    └── CLAUDE.md                          # Technical implementation notes
```

### Legacy/Development Files (Don't Use)
```
❌ Development/Testing Only
├── whisperx/server_igpu_int8.py          # OpenVINO approach (high CPU)
├── whisperx/server_openvino_*.py         # Various OpenVINO experiments  
├── whisperx/whisper_igpu_runtime_optimized.py  # Mock implementation
└── whisperx/test_*.py                    # Testing scripts
```

## 🐳 Docker Compose for UC-1 Pro

### Quick Start
```bash
# Clone repository
git clone https://github.com/Unicorn-Commander/Unicorn-Amanuensis.git
cd Unicorn-Amanuensis

# Start with Docker Compose
docker-compose -f docker-compose.uc1-pro.yml up -d

# Access web interface
open https://amanuensis.yoda.unicorncommander.ai
```

### docker-compose.uc1-pro.yml
```yaml
version: '3.8'

services:
  unicorn-amanuensis:
    image: magicunicorn/unicorn-amanuensis:latest
    container_name: unicorn-amanuensis-uc1-pro
    restart: unless-stopped
    
    # Intel iGPU device access
    devices:
      - /dev/dri:/dev/dri
    
    # Intel GPU environment
    environment:
      - API_PORT=8000
      - WHISPER_MODEL=base
      - ONEAPI_DEVICE_SELECTOR=level_zero:gpu
      - SYCL_DEVICE_FILTER=gpu
      - OMP_NUM_THREADS=1
      - MKL_NUM_THREADS=1
    
    # Port mapping
    ports:
      - "8000:8000"
    
    # Volume mounts
    volumes:
      - ./models:/app/models:cached
      - ./logs:/app/logs:rw
      - /tmp:/tmp:rw
    
    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    
    # Resource limits for UC-1 Pro
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 1G
    
    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

  # Optional: Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: unicorn-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - unicorn-amanuensis
    profiles:
      - proxy

networks:
  default:
    name: unicorn-network
```

## 🔧 Server Selection Guide

### For UC-1 Pro Production (Recommended)
Use **`server_docker_ready.py`** - it auto-detects and uses the best available backend:

```python
# Automatically selects in order:
1. whisper.cpp SYCL (Intel iGPU) ← Target for UC-1 Pro  
2. System whisper.cpp
3. OpenAI Whisper  
4. WhisperX
5. Mock mode (fallback)
```

**Performance on UC-1 Pro**: 7-20x realtime, 0% CPU usage

### For Direct SYCL Control
Use **`server_igpu_ultra.py`** for maximum performance:
- Direct whisper.cpp SYCL integration
- 60x+ realtime target performance  
- Zero CPU usage guaranteed
- Intel UHD Graphics 770 optimized

### For Development/Testing
Use **`server_igpu_sycl.py`** for debugging:
- Direct binary execution
- Detailed logging
- Development features

## 🎯 Hardware-Specific Optimizations

### Intel UHD Graphics 770 (UC-1 Pro)
```bash
# Environment variables
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export SYCL_DEVICE_FILTER=gpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Device detection
ls -la /dev/dri/         # Should show renderD128
sycl-ls                  # Should list Intel GPU
```

### Performance Expectations
| Model | Expected Performance | Memory Usage | Use Case |
|-------|---------------------|--------------|----------|
| tiny | 50-100x realtime | ~500MB | Speed demos |
| base | **7-20x realtime** | ~1GB | **Production** |
| small | 10-15x realtime | ~1.5GB | Better accuracy |
| medium | 5-10x realtime | ~2GB | High quality |
| large-v3 | 3-7x realtime | ~3GB | Best accuracy |

## 🚀 Deployment Commands

### Manual Build & Run
```bash
# Build production image
docker build -f Dockerfile.production -t unicorn-amanuensis:uc1-pro .

# Run with Intel iGPU access
docker run -d \
  --name unicorn-amanuensis \
  --device /dev/dri:/dev/dri \
  -p 8000:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  -e SYCL_DEVICE_FILTER=gpu \
  unicorn-amanuensis:uc1-pro
```

### Using Docker Hub Image
```bash
# Pull and run latest
docker run -d \
  --name unicorn-amanuensis \
  --device /dev/dri:/dev/dri \
  -p 8000:8000 \
  magicunicorn/unicorn-amanuensis:latest
```

### Docker Compose (Recommended)
```bash
# Start services
docker-compose -f docker-compose.uc1-pro.yml up -d

# View logs
docker-compose -f docker-compose.uc1-pro.yml logs -f

# Stop services
docker-compose -f docker-compose.uc1-pro.yml down
```

## 📊 Monitoring & Troubleshooting

### Health Checks
```bash
# Server health
curl http://localhost:8000/health

# Device detection
curl http://localhost:8000/status

# Performance test
curl -X POST http://localhost:8000/transcribe \
  -F "file=@test_audio.mp3" \
  -F "model=base"
```

### Intel GPU Monitoring
```bash
# Inside container or host
intel_gpu_top           # Real-time GPU usage
vainfo                 # Video acceleration info
sycl-ls                # SYCL device listing
```

### Common Issues

**Problem**: 500 Error during transcription
**Solution**: Check device permissions and Intel GPU access
```bash
# Fix device permissions
sudo chmod 666 /dev/dri/renderD128
# Or run container with --privileged flag
```

**Problem**: Low performance (<5x realtime)
**Solution**: Verify Intel iGPU is being used
```bash
# Check logs for "sycl mode" 
docker logs unicorn-amanuensis | grep "sycl mode"
```

**Problem**: Model not found
**Solution**: Models download automatically, but check network access
```bash
# Manual model download
docker exec -it unicorn-amanuensis \
  bash -c "cd /tmp/whisper.cpp && ./models/download-ggml-model.sh base"
```

## 🔄 Updates & Maintenance

### Update Container
```bash
# Pull latest image
docker pull magicunicorn/unicorn-amanuensis:latest

# Restart with new image
docker-compose -f docker-compose.uc1-pro.yml pull
docker-compose -f docker-compose.uc1-pro.yml up -d
```

### Backup Configuration
```bash
# Backup volumes
docker run --rm -v unicorn-models:/source -v $(pwd):/backup \
  alpine tar czf /backup/models-backup.tar.gz -C /source .
```

## 📈 Performance Tuning

### Model Selection for UC-1 Pro
- **Development**: `tiny` (100x+ realtime, basic quality)
- **Production**: `base` (7-20x realtime, good quality) ← **Recommended**
- **High Quality**: `small` (10-15x realtime, better accuracy)  
- **Best Quality**: `large-v3` (3-7x realtime, best accuracy)

### Resource Allocation
```yaml
# In docker-compose.uc1-pro.yml
deploy:
  resources:
    limits:
      memory: 4G        # Adjust based on model size
    reservations:
      memory: 1G        # Minimum guaranteed
```

---

## ⚡ Quick Reference

**Web Interface**: http://localhost:8000/web  
**API Documentation**: http://localhost:8000/  
**Health Check**: http://localhost:8000/health  
**OpenAI Compatible**: http://localhost:8000/v1/audio/transcriptions

**Key Files for UC-1 Pro**:
- Production: `server_docker_ready.py` (auto-detect)
- Performance: `server_igpu_ultra.py` (60x target)  
- Development: `server_igpu_sycl.py` (direct control)

**Expected Performance**: 7-20x realtime with 0% CPU usage on Intel UHD Graphics 770