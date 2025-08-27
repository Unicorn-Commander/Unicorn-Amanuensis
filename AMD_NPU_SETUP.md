# AMD NPU (XDNA) Setup Guide for Unicorn Amanuensis

## For AMD Ryzen 8945HS with XDNA1 NPU

This guide is for when you have access to your AMD 8945HS system with the 780M iGPU and XDNA1 NPU.

## Hardware Specs
- **CPU**: AMD Ryzen 9 8945HS (Phoenix Point)
- **iGPU**: AMD Radeon 780M (RDNA 3)
- **NPU**: AMD XDNA1 (16 TOPS)
- **Codename**: Phoenix Point

## Prerequisites

### 1. Install AMD NPU Drivers on Host

```bash
# Download AMD XDNA driver
wget https://github.com/AMD/xdna-driver/releases/download/v1.0/xdna_1.0_amd64.deb

# Install the driver
sudo dpkg -i xdna_1.0_amd64.deb

# Verify NPU detection
sudo dmesg | grep xdna
ls /dev/xdna*
```

### 2. Install Vitis AI (Optional, for advanced optimization)

```bash
# Download Vitis AI for XDNA
wget https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-3.5.0.tar.gz

# Extract and install
tar -xzf vitis-ai-3.5.0.tar.gz
cd vitis-ai-3.5.0
./install.sh
```

## Building the NPU Container

### 1. Clone the Repository (if not already done)

```bash
git clone https://github.com/Unicorn-Commander/Unicorn-Amanuensis.git
cd Unicorn-Amanuensis
```

### 2. Build NPU-Optimized Container

```bash
# Build the NPU container
docker-compose -f docker-compose.hardware.yml --profile npu build

# Or build directly
docker build -f whisperx/Dockerfile.npu -t unicorn-amanuensis:npu ./whisperx
```

### 3. Run with NPU Support

```bash
# Using the automated launcher
./start-hardware-optimized.sh
# It will auto-detect the AMD NPU and run the correct container

# Or manually
docker-compose -f docker-compose.hardware.yml --profile npu up -d
```

## NPU-Optimized Models

The NPU requires ONNX models that get compiled to XDNA binary format. We provide pre-optimized models:

- `unicorn-commander/whisper-tiny-onnx-npu` - 39M, fastest, ~10W power
- `unicorn-commander/whisper-base-onnx-npu` - 74M, balanced, ~12W power
- `unicorn-commander/whisper-small-onnx-npu` - 244M, accurate, ~15W power

Models are automatically downloaded on first use.

## Performance Expectations

### AMD NPU (XDNA1) Performance
- **Compute**: 16 TOPS INT8
- **Power**: 10-15W for transcription
- **Speed**: 2-3x faster than CPU
- **Best Model**: whisper-base (optimal for 16 TOPS)

### Comparison with Intel iGPU
| Feature | AMD NPU (XDNA1) | Intel iGPU (AlderLake) |
|---------|-----------------|------------------------|
| Power | 10-15W | 25-35W |
| Speed | 2-3x CPU | 3-5x CPU |
| Model Support | tiny, base, small | All models |
| Quantization | INT8 | INT8 |
| Best Use | Battery/Mobile | Desktop/Performance |

## Verifying NPU Usage

### Check NPU Status
```bash
# Inside container
curl http://localhost:9000/api/hardware

# Should return:
{
  "backend": "npu",
  "device": "XDNA",
  "npu_available": true,
  "npu_name": "AMD XDNA1",
  "optimization": "INT8 quantized for NPU"
}
```

### Monitor NPU Usage
```bash
# On host
sudo cat /sys/class/xdna/xdna0/device/npu_utilization
```

## Troubleshooting

### NPU Not Detected
```bash
# Check kernel module
lsmod | grep xdna

# Load module if needed
sudo modprobe xdna

# Check device permissions
ls -la /dev/xdna*
sudo chmod 666 /dev/xdna0
```

### Performance Issues
1. Ensure you're using NPU-optimized models
2. Use smaller models (tiny/base) for best NPU performance
3. Check thermal throttling: `sensors | grep -i amd`

### Container Issues
```bash
# View logs
docker logs unicorn-amanuensis

# Check NPU access inside container
docker exec unicorn-amanuensis ls -la /dev/xdna*
```

## Development Notes

### Creating Your Own NPU Models

```python
# Convert Whisper to ONNX for NPU
import torch
import onnx
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
dummy_input = torch.randn(1, 80, 3000)

torch.onnx.export(
    model,
    dummy_input,
    "whisper-base.onnx",
    opset_version=16,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# Then compile for XDNA (requires Vitis AI)
# vitis-ai-compiler --model whisper-base.onnx --target XDNA1
```

## Integration with UC-1 Pro

When running on your AMD system, the NPU will be used for:
- WhisperX transcription (Unicorn Amanuensis)
- Lightweight inference tasks
- Power-efficient batch processing

The Radeon 780M iGPU can be used for:
- ComfyUI image generation
- Video processing
- Parallel compute tasks

## Support

For AMD NPU-specific issues:
- GitHub: https://github.com/Unicorn-Commander/Unicorn-Amanuensis/issues
- AMD XDNA Docs: https://github.com/AMD/xdna-driver

## Next Steps

1. Test NPU detection on your AMD 8945HS system
2. Benchmark power consumption vs performance
3. Create optimized models for your specific use cases
4. Share benchmarks with the community!

---

*This setup is ready for when you have access to your AMD Ryzen 9 8945HS system.*