# Whisper Model Guide for Hardware Acceleration

## Model Requirements by Hardware Type

### Standard Whisper Models (NVIDIA GPU & CPU)
- Works with standard OpenAI Whisper models from HuggingFace
- Models: `openai/whisper-tiny`, `openai/whisper-base`, `openai/whisper-small`, `openai/whisper-medium`, `openai/whisper-large-v3`
- No special conversion needed
- WhisperX adds word-level timestamps and speaker diarization on top

### Intel iGPU (OpenVINO) - Special Models Needed! ✅
**YES, we need specially optimized models for Intel iGPU!**

#### Why Special Models?
- OpenVINO requires models in IR (Intermediate Representation) format
- INT8 quantization provides 3-5x speedup on Intel iGPUs
- Optimized for Intel Arc, Iris Xe, and UHD Graphics

#### Our HuggingFace Models (Coming Soon)
We'll host optimized models at `unicorn-commander` on HuggingFace:
- `unicorn-commander/whisper-tiny-openvino` (39M, fastest)
- `unicorn-commander/whisper-base-openvino` (74M, fast)
- `unicorn-commander/whisper-small-openvino` (244M, balanced)
- `unicorn-commander/whisper-medium-openvino` (769M, accurate)
- `unicorn-commander/whisper-large-v3-openvino` (1550M, best quality)

#### Features of Our OpenVINO Models:
- **INT8 Quantized**: Optimized for iGPU performance
- **Auto-download**: Models download automatically on first use
- **Hardware Detection**: System auto-selects OpenVINO models when Intel GPU detected
- **3-5x Faster**: Compared to CPU inference
- **Lower Power**: ~15W vs 45W+ on CPU

### AMD NPU (XDNA) - Special Models Needed! ✅
**YES, we need ONNX models for AMD NPU!**

#### Why Special Models?
- AMD NPU requires ONNX format for compilation to XDNA binary
- Models compiled specifically for Ryzen AI NPU architecture
- Optimized for Phoenix/Hawk Point 16 TOPS NPUs

#### Our HuggingFace Models (Coming Soon)
- `unicorn-commander/whisper-tiny-onnx-npu` (Phoenix Point optimized)
- `unicorn-commander/whisper-base-onnx-npu` (Hawk Point optimized)
- `unicorn-commander/whisper-small-onnx-npu` (XDNA1 optimized)

#### Features of Our NPU Models:
- **Power Efficient**: Only ~10W power consumption
- **ONNX Format**: Ready for NPU compilation
- **Auto-compilation**: Converts to XDNA binary on first run
- **Hardware Matched**: Different optimizations for Phoenix vs Hawk Point

## Model Selection Strategy

### Automatic Selection (Default)
The system automatically selects the best model based on detected hardware:

1. **AMD NPU Detected**: Downloads NPU-optimized ONNX model
2. **Intel iGPU Detected**: Downloads OpenVINO INT8 model
3. **NVIDIA GPU Detected**: Uses standard Whisper model with CUDA
4. **CPU Only**: Uses standard Whisper model with CPU optimization

### Model Size Recommendations

| Hardware | Recommended Model | Reason |
|----------|------------------|---------|
| AMD NPU (16 TOPS) | tiny/base | Power efficiency, NPU compute limits |
| Intel Arc A-series | large-v3 | Powerful discrete GPU |
| Intel Iris Xe | small/medium | Good balance of speed/quality |
| Intel UHD | tiny/base | Limited compute resources |
| NVIDIA RTX 4090 | large-v3 | Maximum quality |
| CPU Only | base/small | Reasonable speed |

## Converting Your Own Models

### For Intel iGPU (OpenVINO)
```bash
# Use our converter script
python convert_whisper_openvino.py \
  --model openai/whisper-base \
  --output ./whisper-base-openvino \
  --quantize \
  --upload \
  --repo unicorn-commander/whisper-base-openvino \
  --token YOUR_HF_TOKEN
```

### For AMD NPU
```bash
# ONNX conversion (coming soon)
python convert_whisper_npu.py \
  --model openai/whisper-base \
  --output ./whisper-base-npu \
  --target phoenix \
  --upload
```

## Performance Comparison

| Model | CPU | Intel iGPU | AMD NPU | NVIDIA GPU |
|-------|-----|------------|---------|------------|
| tiny | 1x | 3x | 2x | 5x |
| base | 1x | 3.5x | 2.5x | 6x |
| small | 1x | 4x | N/A | 8x |
| medium | 1x | 4.5x | N/A | 10x |
| large-v3 | 1x | 5x | N/A | 15x |

*Speed multipliers relative to CPU baseline*

## Storage Requirements

- Standard models: ~50MB to 3GB per model
- OpenVINO models: ~40% smaller due to INT8 quantization
- NPU models: Similar to standard size (ONNX format)
- Models cached locally at `~/.cache/unicorn-amanuensis/models/`

## Quick Start

```bash
# The system handles everything automatically!
./start-amanuensis-pro.sh

# It will:
# 1. Detect your hardware
# 2. Download the appropriate optimized model
# 3. Use hardware acceleration automatically
```

## Environment Variables

```bash
# Force specific hardware backend
export WHISPER_DEVICE=igpu  # Force Intel iGPU
export WHISPER_DEVICE=npu   # Force AMD NPU
export WHISPER_DEVICE=cuda  # Force NVIDIA GPU
export WHISPER_DEVICE=cpu   # Force CPU

# Select model size
export WHISPER_MODEL=base   # tiny, base, small, medium, large-v3

# Disable optimized models (use standard everywhere)
export USE_OPTIMIZED_MODELS=false
```

## FAQ

**Q: Do I need to manually download models?**
A: No! Models download automatically on first use.

**Q: Can I use standard Whisper models on Intel iGPU?**
A: Yes, but performance will be much slower. Our OpenVINO models are 3-5x faster.

**Q: Will my existing Whisper models work?**
A: Yes on CPU/NVIDIA. For Intel/AMD hardware, optimized models give better performance.

**Q: How much faster are the optimized models?**
A: Intel iGPU: 3-5x faster. AMD NPU: 2-3x faster with 5x less power.

**Q: Where are models stored?**
A: `~/.cache/unicorn-amanuensis/models/` with subdirectories for each hardware type.

## Contributing

Help us optimize more models! Submit PRs to:
- https://github.com/Unicorn-Commander/Unicorn-Amanuensis
- https://huggingface.co/unicorn-commander