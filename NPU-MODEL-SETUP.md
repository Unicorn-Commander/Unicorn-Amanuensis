# 🦄 AMD NPU Model Setup

## Current Status

✅ **NPU Detected**: AMD Phoenix XDNA1 (AIE 1.1) is working
✅ **Container Built**: With model download capability
✅ **Scripts Ready**: Download and rebuild scripts created
⚠️ **Models Needed**: Must download to host before container starts

## Issue Identified

The Docker container downloads models during build, but the volume mount in docker-compose.yml overwrites `/app/models` with the empty host directory.

**Solution**: Download models to host first, then mount them into the container.

---

## Quick Fix (Run These Commands)

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis

# Step 1: Fix permissions (requires sudo password)
echo "MagicUnicorn!8-)" | sudo -S chown -R $USER:$USER whisperx/models/

# Step 2: Download AMD NPU models
./download-npu-models.sh

# Step 3: Rebuild container (models will now be available)
./rebuild-npu.sh
```

---

## What Models Will Be Downloaded

### 1. **whisper-base-amd-npu-int8**
- Model: Whisper Base
- Optimization: INT8 quantized for AMD NPU
- Size: ~300MB
- Use: Fast transcription
- Location: `whisperx/models/whisper-base-amd-npu-int8/`

### 2. **whisperx-large-v3-npu**
- Model: Whisper Large-v3
- Features: **Diarization** + **Word-level timestamps**
- Optimization: AMD NPU optimized
- Size: ~3GB
- Use: Production-quality transcription
- Location: `whisperx/models/whisperx-large-v3-npu/`

---

## Manual Download (If Script Fails)

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx

# Fix permissions first
echo "MagicUnicorn!8-)" | sudo -S chown -R $USER:$USER models/

# Create directories
mkdir -p models/whisper-base-amd-npu-int8
mkdir -p models/whisperx-large-v3-npu

# Download base model
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('magicunicorn/whisper-base-amd-npu-int8',
                 local_dir='models/whisper-base-amd-npu-int8')
"

# Download large-v3 model
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('magicunicorn/whisperx-large-v3-npu',
                 local_dir='models/whisperx-large-v3-npu')
"
```

---

## Verify Models Downloaded

```bash
cd whisperx/models

# Check base model
ls -lh whisper-base-amd-npu-int8/
# Should see: config.json, encoder*.onnx, decoder*.onnx, etc.

# Check large-v3 model
ls -lh whisperx-large-v3-npu/
# Should see: config.json, model files, diarization config, etc.
```

---

## Update docker-compose-npu.yml

The current docker-compose file mounts `./models:/app/models`, which is correct.

Just make sure environment variables point to the right paths:

```yaml
environment:
  - WHISPER_MODEL=base  # or large-v3
  - WHISPER_NPU_MODEL_PATH=/app/models/whisper-base-amd-npu-int8
  - WHISPER_LARGE_V3_PATH=/app/models/whisperx-large-v3-npu
```

---

## Testing After Model Download

```bash
# Restart container
./rebuild-npu.sh

# Check logs for model loading
docker logs amanuensis-npu | grep -E "(model|NPU|Loading)"

# Should see:
# ✅ AMD Phoenix NPU detected and working
# 📁 Using AMD NPU INT8 models from: /app/models/whisper-base-amd-npu-int8
# ✅ Encoder loaded: ...
# ✅ Decoder loaded: ...
# ✅ Models loaded to NPU buffers
```

---

## API Usage with Both Models

### Use Base Model (Faster)
```bash
curl -X POST \
  -F "file=@audio.mp3" \
  -F "model=base" \
  http://localhost:9000/v1/audio/transcriptions
```

### Use Large-v3 (Diarization + Word Timestamps)
```bash
curl -X POST \
  -F "file=@audio.mp3" \
  -F "model=large-v3" \
  -F "diarization=true" \
  -F "word_timestamps=true" \
  http://localhost:9000/v1/audio/transcriptions
```

---

## Model Sizes

| Model | Size | Download Time | Features |
|-------|------|---------------|----------|
| whisper-base-amd-npu-int8 | ~300MB | 1-2 min | Fast transcription |
| whisperx-large-v3-npu | ~3GB | 10-15 min | Diarization, word timestamps |

---

## Troubleshooting

### Models Not Loading

```bash
# Check if models exist
docker exec amanuensis-npu ls -la /app/models/

# Should show:
# whisper-base-amd-npu-int8/
# whisperx-large-v3-npu/
```

### Permission Denied

```bash
# Fix ownership
echo "MagicUnicorn!8-)" | sudo -S chown -R $USER:$USER whisperx/models/
```

### Models Downloaded But Empty

```bash
# Check host directory
ls -lh whisperx/models/whisper-base-amd-npu-int8/

# If empty, re-download
./download-npu-models.sh
```

---

## Next Steps After Download

1. ✅ Download models: `./download-npu-models.sh`
2. ✅ Rebuild container: `./rebuild-npu.sh`
3. ✅ Test GUI: http://localhost:9000/web
4. ✅ Test API with base model
5. ✅ Test API with large-v3 model (diarization)
6. ✅ Publish to Docker Hub: `./publish-dockerhub.sh`

---

## Expected Performance

### AMD Phoenix NPU (16 TOPS)

**Base Model (INT8)**
- Speed: ~2-4x realtime
- Power: ~5-10W
- Use: Quick transcription

**Large-v3 Model (NPU optimized)**
- Speed: ~1-2x realtime
- Power: ~8-15W
- Features: Speaker diarization, word timestamps
- Use: Production quality

---

<div align="center">
  <b>🦄 Powered by AMD Phoenix NPU 🦄</b><br>
  <i>Built with 💜 by Magic Unicorn</i>
</div>
