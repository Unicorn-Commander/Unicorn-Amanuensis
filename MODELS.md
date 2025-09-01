# Unicorn Amanuensis Model Management

## Model Storage Strategy

To keep our git repository lean and fast, we store models separately:

### 1. Whisper Models (Standard)
- Downloaded automatically from OpenAI via whisper.cpp
- Stored locally in `/tmp/whisper.cpp/models/`
- Formats: GGML binary format for whisper.cpp

### 2. Quantized Models (INT8/INT4)
- **Hugging Face Repository**: https://huggingface.co/Unicorn-Commander/whisper-igpu-optimized
- Optimized for Intel iGPU with INT8 quantization
- 2-4x faster inference with minimal accuracy loss

### 3. OpenVINO Models
- Stored in `~/openvino-models/` (not in git)
- Can be published to HuggingFace if needed
- Formats: OpenVINO IR format

## Available Models

| Model | Size | Performance | Use Case |
|-------|------|-------------|----------|
| tiny | 39 MB | 100x+ realtime | Fast drafts |
| base | 74 MB | 50x realtime | General use |
| small | 244 MB | 30x realtime | Better accuracy |
| medium | 769 MB | 20x realtime | High accuracy |
| large-v3 | 1550 MB | 10x realtime | Best accuracy |

## Model Downloads

### Automatic Download
Models are downloaded automatically when first requested:
```python
from whisper_igpu_real import WhisperIGPUReal
whisper = WhisperIGPUReal('base')  # Downloads if not present
```

### Manual Download
```bash
cd /tmp/whisper.cpp
./models/download-ggml-model.sh base
./models/download-ggml-model.sh large-v3
```

## Quantization

### Create INT8 Model
```python
python3 quantize_to_int8.py \
  --model-dir ~/openvino-models/whisper-base-openvino \
  --output-dir ~/openvino-models/whisper-base-int8
```

### Performance Gains
- INT8: 2-4x faster, <1% accuracy loss
- INT4: 4-8x faster, 1-3% accuracy loss (experimental)

## Publishing to Hugging Face

### Setup
```bash
pip install huggingface-hub
huggingface-cli login
```

### Upload Model
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="~/openvino-models/whisper-base-int8",
    repo_id="Unicorn-Commander/whisper-igpu-optimized",
    repo_type="model",
    path_in_repo="whisper-base-int8"
)
```

## Device-Specific Optimizations

### Intel iGPU (Primary Target)
- Use SYCL-compiled whisper.cpp
- INT8 quantization recommended
- Achieve 10-20x realtime

### AMD NPU (Future)
- ONNX Runtime with Vitis AI
- Model conversion required
- Target: 20-40x realtime

### NVIDIA GPU (Reserved for LLM)
- CUDA-enabled whisper.cpp
- Keep for LLM inference
- Not primary target for Unicorn Amanuensis

## Best Practices

1. **Never commit models to git** - Use .gitignore
2. **Use HuggingFace for distribution** - Better CDN and versioning
3. **Cache models locally** - Avoid re-downloading
4. **Quantize for production** - Better performance/accuracy trade-off
5. **Test accuracy after quantization** - Ensure quality maintained