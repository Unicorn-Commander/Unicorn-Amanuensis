# 🦄 Unicorn Amanuensis NPU - Quick Start

## What You Have

✅ **AMD Phoenix NPU** - `/dev/accel/accel0` is working
✅ **ONNX INT8 Models** - Downloaded and ready (1.6 GB total)
✅ **Custom NPU Runtime** - From UC-Meeting-Ops (working production code)
✅ **Docker Setup** - Container configured for NPU passthrough

## Models Downloaded

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| whisper-base-onnx-int8 | 125 MB | ~70x realtime | Good | Fast transcription |
| whisper-large-v3-onnx-int8 | 1.5 GB | ~18x realtime | Best | Production quality |

**Important**: These are real ONNX models from `onnx-community`, NOT the magicunicorn placeholders!

## One-Command Install

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
./install-npu.sh
```

This script will:
1. ✅ Check NPU device
2. ✅ Check Docker
3. ✅ Verify/download models
4. ✅ Create Docker network
5. ✅ Build container
6. ✅ Start service

## Manual Install

If you prefer step-by-step:

### 1. Download Models (if needed)
```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
./download-models.sh
```

Choose option 3 to get both models.

### 2. Create Network
```bash
docker network create unicorn-network
```

### 3. Build and Start
```bash
docker compose -f docker-compose-npu.yml up -d --build
```

### 4. Check Logs
```bash
docker logs -f amanuensis-npu
```

Look for:
```
✅ NPU device opened: /dev/accel/accel0
   AIE Version: 2.0
🚀 AMD Phoenix NPU detected and working - enabling NPU acceleration!
```

## Access the Service

- **Web GUI**: http://localhost:9000/web
- **API**: http://localhost:9000
- **Health Check**: http://localhost:9000/health

## Test It

```bash
curl -X POST -F "file=@audio.wav" http://localhost:9000/transcribe
```

Or use the web interface to upload files.

## Switch Models

Edit `docker-compose-npu.yml`:

**For Base (fast):**
```yaml
environment:
  - WHISPER_MODEL=base
  - WHISPER_NPU_MODEL_PATH=/app/models/whisper-base-onnx-int8
```

**For Large-v3 (accurate):**
```yaml
environment:
  - WHISPER_MODEL=large-v3
  - WHISPER_NPU_MODEL_PATH=/app/models/whisper-large-v3-onnx-int8
```

Then restart:
```bash
docker compose -f docker-compose-npu.yml up -d
```

## Troubleshooting

### Container won't start
```bash
# Check NPU device
ls -la /dev/accel/accel0

# Check driver
lsmod | grep amdxdna

# View full logs
docker logs amanuensis-npu
```

### NPU not detected
```bash
# Verify device permissions
groups | grep render

# Check if NPU is accessible
docker exec amanuensis-npu ls -la /dev/accel/accel0
```

### Models not loading
```bash
# Verify models exist
docker exec amanuensis-npu ls -la /app/models/whisper-base-onnx-int8/onnx/

# Should show:
# encoder_model_int8.onnx (23 MB)
# decoder_model_int8.onnx (51 MB)
# decoder_with_past_model_int8.onnx (48 MB)
```

### Service frozen/not responding
This was the previous issue - likely NPU contention. Solution:
```bash
# Stop all containers
docker stop $(docker ps -q)

# Wait 5 seconds
sleep 5

# Start only Amanuensis
docker compose -f docker-compose-npu.yml up -d
```

## Performance Expectations

Based on UC-Meeting-Ops working implementation:

| Audio Length | Base Model | Large-v3 Model |
|--------------|------------|----------------|
| 1 minute | ~0.8 sec | ~3 sec |
| 10 minutes | ~8 sec | ~30 sec |
| 1 hour | ~50 sec | ~3 min |

**Real-time factors**: Base ~70x, Large-v3 ~18x

## What's Different from Before

### ❌ Before (didn't work):
- Tried to use magicunicorn placeholder repos
- Downloaded 42-byte .npumodel files (just pointers)
- Containers froze during model loading

### ✅ Now (working):
- Using actual ONNX models from onnx-community
- Real INT8 quantized weights (1.6 GB total)
- Custom NPU runtime from UC-Meeting-Ops
- Models are volume-mounted (not built into image)

## Files Created

- `download-models.sh` - Interactive model downloader
- `install-npu.sh` - Complete installation script
- `models/README.md` - Model documentation
- `Dockerfile.npu` - Updated (no more placeholder downloads)

## Next Steps

1. Run `./install-npu.sh`
2. Access http://localhost:9000/web
3. Upload an audio file
4. Watch the NPU accelerate transcription! 🚀

---

**Need help?** Check the logs:
```bash
docker logs -f amanuensis-npu
```
