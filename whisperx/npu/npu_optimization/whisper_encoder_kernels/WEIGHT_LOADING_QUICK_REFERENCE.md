# Whisper Weight Loading - Quick Reference Guide

## TL;DR - Key Findings

### 1. Model Location
```
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/onnx/
```

### 2. Available Models
- **encoder_model.onnx** (78.6 MB) - Full FP32 (RECOMMENDED)
- encoder_model_fp16.onnx (39.4 MB) - Half precision
- encoder_model_int8.onnx (22.1 MB) - Quantized
- decoder_model*.onnx - 6 variants (50-200 MB each)

### 3. Encoder Architecture
- 6 identical transformer layers
- Each layer: 512 hidden dimension, 8 attention heads, 2048 FFN intermediate
- Q/K/V/O projections: [512 x 512] weight matrices
- FFN weights: [2048 x 512] and [512 x 2048]
- LayerNorm: gamma/beta [512]

### 4. Weight Tensor Names Pattern
```
/encoder/layers.{i}/self_attn/q_proj/weight       [512, 512]
/encoder/layers.{i}/self_attn/k_proj/weight       [512, 512]
/encoder/layers.{i}/self_attn/v_proj/weight       [512, 512]
/encoder/layers.{i}/self_attn/out_proj/weight     [512, 512]
/encoder/layers.{i}/fc1/weight                    [2048, 512]
/encoder/layers.{i}/fc2/weight                    [512, 2048]
/encoder/layers.{i}/self_attn_layer_norm/weight   [512]
/encoder/layers.{i}/final_layer_norm/weight       [512]
(All have /bias variants as well)
```

### 5. Total Weight Size
- **Per layer:** ~1.05M parameters
- **All 6 layers:** ~6.3M parameters
- **In FP32:** 25.2 MB
- **In BF16:** 12.6 MB (2x compression!)
- **In INT8:** 6.3 MB (4x compression, slight accuracy loss)

---

## Quick Implementation Guide

### Step 1: Load Weights from ONNX
```python
import onnx
import numpy as np

model = onnx.load('encoder_model.onnx')
weights = {}

for init in model.graph.initializer:
    data = np.frombuffer(init.raw_data, dtype=np.float32)
    shape = tuple(init.dims)
    weights[init.name] = data.reshape(shape)
```

### Step 2: Convert to BF16
```python
import torch

bf16_weights = {}
for name, weight in weights.items():
    tensor = torch.from_numpy(weight)
    bf16_weights[name] = tensor.to(torch.bfloat16).numpy()
```

### Step 3: Organize by Layer
```python
layers = {}
for layer_idx in range(6):
    prefix = f'/encoder/layers.{layer_idx}'
    layer_weights = {
        'attention': {},
        'ffn': {},
        'layernorm': {}
    }
    
    for name, weight in bf16_weights.items():
        if prefix in name:
            if 'self_attn' in name:
                layer_weights['attention'][name] = weight
            elif 'fc' in name:
                layer_weights['ffn'][name] = weight
            elif 'layer_norm' in name:
                layer_weights['layernorm'][name] = weight
    
    layers[layer_idx] = layer_weights
```

### Step 4: Transfer to NPU
```python
import xrt

device = xrt.device(0)  # /dev/accel/accel0

for layer_idx, layer_weights in layers.items():
    for section, weights_dict in layer_weights.items():
        for name, weight in weights_dict.items():
            # Create XRT buffer
            buf = xrt.bo(device, weight.nbytes)
            
            # Copy data to device
            buf.write(weight.tobytes(), 0, weight.nbytes)
            buf.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
```

---

## Weight Tensor Details

### Attention Weights (Per Layer)
| Tensor | Shape | Params | BF16 Size |
|--------|-------|--------|-----------|
| q_proj/weight | 512×512 | 262,144 | 512 KB |
| k_proj/weight | 512×512 | 262,144 | 512 KB |
| v_proj/weight | 512×512 | 262,144 | 512 KB |
| out_proj/weight | 512×512 | 262,144 | 512 KB |
| **Attention subtotal** | - | **1.05M** | **2.0 MB** |

### FFN Weights (Per Layer)
| Tensor | Shape | Params | BF16 Size |
|--------|-------|--------|-----------|
| fc1/weight | 2048×512 | 1,048,576 | 2.0 MB |
| fc2/weight | 512×2048 | 1,048,576 | 2.0 MB |
| **FFN subtotal** | - | **2.1M** | **4.0 MB** |

### LayerNorm (Per Layer)
| Tensor | Shape | Params | BF16 Size |
|--------|-------|--------|-----------|
| self_attn_layer_norm/weight | 512 | 512 | 1 KB |
| self_attn_layer_norm/bias | 512 | 512 | 1 KB |
| final_layer_norm/weight | 512 | 512 | 1 KB |
| final_layer_norm/bias | 512 | 512 | 1 KB |
| **LayerNorm subtotal** | - | **2K** | **4 KB** |

### Biases (Per Layer)
- Attention projections: 4×512 = 2,048 params (4 KB BF16)
- FFN: 2×512 + 2048 = 3,072 params (6 KB BF16)
- **Total biases: ~10 KB per layer**

### Per-Layer Total
- **Parameters:** ~1.05M
- **FP32 Size:** 4.2 MB
- **BF16 Size:** 2.1 MB
- **INT8 Size:** 1.05 MB

### 6 Layers Total
- **Parameters:** 6.3M
- **FP32 Size:** 25.2 MB
- **BF16 Size:** 12.6 MB
- **INT8 Size:** 6.3 MB

---

## Recommended Weight Format for NPU

### Option 1: BF16 (RECOMMENDED)
**Pros:**
- 2x smaller than FP32
- Same range as FP32 (no precision loss in exponent)
- Excellent for transformer models
- Native support in AMD XDNA1 NPU
- PyTorch: `torch.bfloat16`

**Cons:**
- Reduced mantissa precision (7 vs 23 bits)
- Slightly slower than INT8

### Option 2: INT8
**Pros:**
- 4x smaller than FP32
- Faster computation on NPU
- Used in production models (Kokoro TTS)

**Cons:**
- Requires dequantization before use
- Per-channel quantization needed for accuracy
- Slight accuracy loss possible

### Option 3: FP16
**Pros:**
- Balanced precision/size

**Cons:**
- Unstable in some transformer operations
- Known issues with layer normalization
- Not recommended for Whisper

---

## Example: Complete Loading Pipeline

```python
#!/usr/bin/env python3
"""Complete Whisper weight loading for NPU kernels"""

import onnx
import numpy as np
import torch
import xrt
from pathlib import Path

class WhisperNPUWeightLoader:
    def __init__(self, onnx_path, device_id=0):
        self.onnx_path = onnx_path
        self.device_id = device_id
        self.device = xrt.device(device_id)
        
    def load_and_convert(self):
        """Load ONNX weights and convert to BF16"""
        
        # 1. Load ONNX model
        print(f"Loading: {self.onnx_path}")
        model = onnx.load(str(self.onnx_path))
        
        # 2. Extract and convert weights
        bf16_weights = {}
        for init in model.graph.initializer:
            data = np.frombuffer(init.raw_data, dtype=np.float32)
            weight = data.reshape(tuple(init.dims))
            
            # Convert to BF16
            tensor = torch.from_numpy(weight)
            bf16_weights[init.name] = tensor.to(torch.bfloat16).numpy()
        
        print(f"Loaded {len(bf16_weights)} weight tensors")
        return bf16_weights
    
    def organize_by_layer(self, weights):
        """Organize weights by encoder layer"""
        layers = {}
        
        for layer_idx in range(6):
            prefix = f'/encoder/layers.{layer_idx}'
            layer_weights = {
                'attention': {},
                'ffn': {},
                'layernorm': {}
            }
            
            for name, weight in weights.items():
                if prefix not in name:
                    continue
                
                if 'self_attn' in name:
                    layer_weights['attention'][name] = weight
                elif 'fc1' in name or 'fc2' in name:
                    layer_weights['ffn'][name] = weight
                elif 'layer_norm' in name:
                    layer_weights['layernorm'][name] = weight
            
            layers[layer_idx] = layer_weights
        
        return layers
    
    def transfer_to_device(self, layers):
        """Transfer weights to NPU device memory"""
        device_buffers = {}
        
        for layer_idx, layer_weights in layers.items():
            device_buffers[layer_idx] = {}
            
            for section, weights_dict in layer_weights.items():
                device_buffers[layer_idx][section] = {}
                
                for name, weight in weights_dict.items():
                    # Create XRT buffer
                    size = weight.nbytes
                    buf = xrt.bo(self.device, size)
                    
                    # Write data
                    buf.write(weight.tobytes(), 0, size)
                    buf.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                    
                    device_buffers[layer_idx][section][name] = buf
                    
                    print(f"  ✓ Layer {layer_idx}/{section}/{name.split('/')[-2]}: {weight.shape}")
        
        return device_buffers
    
    def run(self):
        """Execute complete pipeline"""
        print("\n" + "="*70)
        print("WHISPER WEIGHT LOADING FOR NPU")
        print("="*70 + "\n")
        
        # Load and convert
        weights = self.load_and_convert()
        
        # Organize by layer
        layers = self.organize_by_layer(weights)
        
        # Transfer to device
        print("\nTransferring to NPU device...\n")
        device_buffers = self.transfer_to_device(layers)
        
        print("\n✅ Weight loading complete!")
        print(f"   Total weights transferred: {len(weights)}")
        print(f"   Total layers: {len(layers)}")
        print(f"   Expected NPU SRAM: ~12.6 MB (BF16)")
        
        return device_buffers

# Usage
if __name__ == "__main__":
    loader = WhisperNPUWeightLoader(
        '/home/ucadmin/UC-1/Unicorn-Amanuensis/'
        'whisperx/models/whisper_onnx_cache/'
        'models--onnx-community--whisper-base/'
        'onnx/encoder_model.onnx',
        device_id=0
    )
    
    buffers = loader.run()
```

---

## Performance Expectations

### With BF16 Weights + Custom MLIR Kernels
- **Mel Spectrogram:** 20-30x faster (custom kernel)
- **Attention:** 30-50x faster (custom kernel + BF16)
- **FFN:** 30-50x faster (custom kernel + BF16)
- **Overall:** 50-100x faster than CPU baseline
- **Target:** 200-220x realtime (with full optimization)

### Memory Requirements
- **Weights only (BF16):** 12.6 MB
- **Activation buffers (per request):** ~10 MB
- **Total per request:** ~25 MB
- **NPU SRAM:** 256 MB available (plenty!)

---

## Debugging Tips

### Check Model Structure
```python
import onnx
model = onnx.load('encoder_model.onnx')
print(f"Inputs: {[inp.name for inp in model.graph.input]}")
print(f"Outputs: {[out.name for out in model.graph.output]}")
print(f"Total initializers (weights): {len(model.graph.initializer)}")
```

### Verify Weight Names
```python
# Find all weight tensor names
for init in model.graph.initializer:
    print(f"{init.name}: {tuple(init.dims)}")
```

### Check Data Types
```python
import onnx
proto_dtype = init.data_type
if proto_dtype == onnx.TensorProto.FLOAT:
    print("FP32")
elif proto_dtype == onnx.TensorProto.FLOAT16:
    print("FP16")
elif proto_dtype == onnx.TensorProto.INT8:
    print("INT8")
```

### Verify BF16 Conversion
```python
# Compare before/after
print(f"Original max: {weight.max()}")
print(f"BF16 max: {bf16_weight.max()}")
print(f"Error: {abs(weight.max() - bf16_weight.max())}")
```

---

## File Locations

| Item | Location |
|------|----------|
| ONNX Models | `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/onnx/` |
| Config | `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/config.json` |
| Existing weight code | `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/` |
| Kernel directory | `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/` |
| Research document | `WHISPER_WEIGHTS_LOADING_RESEARCH.md` (this directory) |

---

## Key References

- **ONNX Spec:** https://github.com/onnx/onnx/blob/main/docs/IR.md
- **PyTorch BF16:** https://pytorch.org/docs/stable/generated/torch.bfloat16.html
- **XRT API:** https://xilinx.github.io/XRT/master/html/
- **Whisper Model:** https://github.com/openai/whisper

---

**Date:** 2025-11-19
**Status:** Quick reference for weight loading
**See Also:** WHISPER_WEIGHTS_LOADING_RESEARCH.md for detailed guide
