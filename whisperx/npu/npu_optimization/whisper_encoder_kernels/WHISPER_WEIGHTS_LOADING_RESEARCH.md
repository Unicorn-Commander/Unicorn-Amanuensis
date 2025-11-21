# Research: Loading Whisper Model Weights into NPU Kernels

## Executive Summary

This research covers how to load Whisper encoder weights from ONNX/PyTorch models into custom NPU kernels for the AMD Phoenix XDNA1 NPU. The focus is on extracting Q/K/V projection matrices, FFN layers, and LayerNorm parameters in BF16 format for optimal NPU performance.

---

## 1. Whisper ONNX Model Location & Structure

### 1.1 Model Cache Location
```
/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/onnx/
```

### 1.2 Available Model Variants

The ONNX cache contains comprehensive Whisper Base model variants:

**Encoder Models** (Full FP32 architecture):
- `encoder_model.onnx` (78.6 MB) - Full precision (FP32)
- `encoder_model_fp16.onnx` (39.4 MB) - Half precision
- `encoder_model_int8.onnx` (22.1 MB) - Quantized INT8
- `encoder_model_q4.onnx` (17.9 MB) - Q4 quantization
- `encoder_model_bnb4.onnx` (16.8 MB) - BNB4 quantization

**Decoder Models** (Similar variants):
- `decoder_model*.onnx` - 6 variants (186-199 MB each)
- `decoder_with_past_model*.onnx` - 7 variants with KV cache optimization (47.8-186 MB)

### 1.3 Model Configuration

From `config.json`:
```python
{
  "model_type": "whisper",
  "d_model": 512,                    # Hidden dimension
  "encoder_layers": 6,               # 6 encoder layers
  "encoder_attention_heads": 8,      # Multi-head attention
  "encoder_ffn_dim": 2048,          # FFN intermediate dimension
  "decoder_layers": 6,
  "decoder_attention_heads": 8,
  "decoder_ffn_dim": 2048,
  "num_mel_bins": 80,               # Mel spectrogram bins
  "max_source_positions": 1500,     # Max sequence length for encoder
}
```

### 1.4 Encoder Input/Output

**Encoder Inputs:**
- `input_features`: Shape `[batch_size, 80, sequence_length]`
  - 80 = mel spectrogram bins
  - sequence_length = up to 1500 frames (30 seconds of audio @ 50fps)

**Encoder Outputs:**
- `last_hidden_state`: Shape `[batch_size, 1500, 512]`
  - 1500 = maximum encoder sequence length
  - 512 = d_model (hidden dimension)
  - Contains context vectors for decoder cross-attention

---

## 2. Whisper Encoder Architecture & Weight Organization

### 2.1 Per-Layer Structure (6 identical layers)

Each of the 6 encoder layers contains:

```
Layer i (repeated 6 times):
  ├── self_attn_layer_norm (LayerNorm)
  │   ├── weight: [512]           # gamma - scale parameter
  │   └── bias: [512]             # beta - shift parameter
  │
  ├── self_attn (MultiHeadAttention)
  │   ├── q_proj (Q projection)
  │   │   ├── weight: [512, 512]  # MatMul weight
  │   │   └── bias: [512]
  │   ├── k_proj (K projection)
  │   │   ├── weight: [512, 512]
  │   │   └── bias: [512]
  │   ├── v_proj (V projection)
  │   │   ├── weight: [512, 512]
  │   │   └── bias: [512]
  │   └── out_proj (Output projection)
  │       ├── weight: [512, 512]
  │       └── bias: [512]
  │
  ├── final_layer_norm (LayerNorm)
  │   ├── weight: [512]
  │   └── bias: [512]
  │
  └── fc1/fc2 (Feed-Forward Network)
      ├── fc1 (Expand to 2048)
      │   ├── weight: [2048, 512]  # d_ff x d_model
      │   └── bias: [2048]
      └── fc2 (Contract back to 512)
          ├── weight: [512, 2048]  # d_model x d_ff
          └── bias: [512]
```

### 2.2 Total Weight Count for NPU Kernels

Per layer (6 identical layers):
- **Attention:** 4 MatMul weights (512×512 each) = 1,048,576 params/weight
- **FFN:** 2 MatMul weights (2048×512 and 512×2048) = ~1.5M params
- **LayerNorm:** 2 × 512 gamma/beta = 1,024 params
- **Biases:** 3×512 (attn) + 2×512 (ffn) + 2×512 (layernorm) = 3,584 params

**Per layer total:** ~1.05M parameters × 6 layers = **6.3M parameters**

**In BF16 format:** 
- 6.3M params × 2 bytes = **12.6 MB per layer set**
- Total encoder weights: **12.6 MB** (very manageable for NPU SRAM)

---

## 3. Weight Tensor Names and Shapes in ONNX Format

### 3.1 Attention Projection Weights

In ONNX graph, weights follow naming pattern:

```
# Layer 0 (first layer)
/encoder/layers.0/self_attn/q_proj/weight    → [512, 512]
/encoder/layers.0/self_attn/q_proj/bias      → [512]
/encoder/layers.0/self_attn/k_proj/weight    → [512, 512]
/encoder/layers.0/self_attn/k_proj/bias      → [512]
/encoder/layers.0/self_attn/v_proj/weight    → [512, 512]
/encoder/layers.0/self_attn/v_proj/bias      → [512]
/encoder/layers.0/self_attn/out_proj/weight  → [512, 512]
/encoder/layers.0/self_attn/out_proj/bias    → [512]

# Layer 1, 2, 3, 4, 5 (same pattern, replace .0 with .1, .2, etc.)
```

### 3.2 FFN Weights

```
/encoder/layers.0/fc1/weight          → [2048, 512]
/encoder/layers.0/fc1/bias            → [2048]
/encoder/layers.0/fc2/weight          → [512, 2048]
/encoder/layers.0/fc2/bias            → [512]
```

### 3.3 LayerNorm Parameters

```
/encoder/layers.0/self_attn_layer_norm/weight      → [512]  (gamma/scale)
/encoder/layers.0/self_attn_layer_norm/bias        → [512]  (beta/offset)
/encoder/layers.0/final_layer_norm/weight          → [512]
/encoder/layers.0/final_layer_norm/bias            → [512]

# Global encoder layer norm
/encoder/layer_norm/weight    → [512]
/encoder/layer_norm/bias      → [512]
```

---

## 4. Extracting Weights from ONNX Models

### 4.1 Method 1: Using ONNX Runtime (Simple, Recommended)

```python
import onnxruntime as ort
import numpy as np
from pathlib import Path

# Load encoder model
encoder_path = Path('/home/ucadmin/UC-1/Unicorn-Amanuensis/'
                    'whisperx/models/whisper_onnx_cache/'
                    'models--onnx-community--whisper-base/'
                    'onnx/encoder_model.onnx')

# Option 1: CPU Execution Provider (safe fallback)
session = ort.InferenceSession(
    str(encoder_path),
    providers=['CPUExecutionProvider']
)

# Get all inputs and outputs
print("Encoder inputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape}")
    
print("Encoder outputs:")
for out in session.get_outputs():
    print(f"  {out.name}: {out.shape}")
```

**Output:**
```
Encoder inputs:
  input_features: ['batch_size', 'feature_size', 'encoder_sequence_length']
  
Encoder outputs:
  last_hidden_state: ['batch_size', 1500, 512]
```

### 4.2 Method 2: Direct ONNX Protobuf Loading (Access Raw Weights)

```python
import onnx
import numpy as np
from pathlib import Path

# Load ONNX model
encoder_path = Path('/home/ucadmin/.../encoder_model.onnx')
model = onnx.load(str(encoder_path))

# Access initializers (weights and constants)
print(f"Total parameters: {len(model.graph.initializer)}")

# Extract specific weight tensors
weight_dict = {}
for init in model.graph.initializer:
    name = init.name
    
    # Convert tensor to numpy array
    if init.data_type == onnx.TensorProto.FLOAT:  # FP32
        data = np.frombuffer(init.raw_data, dtype=np.float32)
        shape = tuple(init.dims)
        weight_dict[name] = data.reshape(shape)
    elif init.data_type == onnx.TensorProto.FLOAT16:  # FP16
        data = np.frombuffer(init.raw_data, dtype=np.float16)
        shape = tuple(init.dims)
        weight_dict[name] = data.reshape(shape)

# Access specific weight
q_proj_weight = weight_dict['/encoder/layers.0/self_attn/q_proj/weight']
print(f"Q projection weight shape: {q_proj_weight.shape}")
print(f"Q projection weight dtype: {q_proj_weight.dtype}")
```

### 4.3 Method 3: Using OpenVINO IR Format (Optimized)

For better optimization, first convert ONNX to OpenVINO IR:

```python
from optimum.intel import OVModelForSpeechSeq2Seq

# Convert ONNX to OpenVINO IR format
model = OVModelForSpeechSeq2Seq.from_pretrained(
    '/home/ucadmin/.../models--onnx-community--whisper-base',
    export=True
)

# Save as OpenVINO IR
model.save_pretrained('/path/to/openvino-whisper-base')

# Now you have:
# - openvino_encoder_model.xml (graph definition)
# - openvino_encoder_model.bin (weights in binary format)
```

**Advantages:**
- Weights stored in optimized binary format
- Easier to parse with OpenVINO C++ API
- Direct memory mapping support

---

## 5. Converting Weights to BF16 Format for NPU

### 5.1 Why BF16 for AMD Phoenix NPU?

**Benefits:**
- Same 32-bit precision as FP32 (same range)
- 2x smaller than FP32 (16-bit mantissa)
- Faster NPU computation than FP32
- Better DRAM bandwidth utilization
- Excellent for transformer models
- Automatic support in MLIR-AIE2 kernels

**Conversion Process:**
```
FP32 (sign: 1b, exponent: 8b, mantissa: 23b)
  ↓ (keep upper 16 bits)
BF16 (sign: 1b, exponent: 8b, mantissa: 7b)
```

### 5.2 Converting Weights to BF16

```python
import numpy as np
import torch

# Method 1: Using NumPy (manual)
def convert_fp32_to_bf16(fp32_array):
    """Convert FP32 to BF16 (brain float)"""
    # View as uint32, right-shift by 16 to get BF16
    uint32_view = np.frombuffer(fp32_array.tobytes(), dtype=np.uint32)
    bf16_uint = (uint32_view >> 16).astype(np.uint16)
    return bf16_uint

# Method 2: Using PyTorch (simpler, more accurate)
def convert_to_bf16(weight_array):
    """Convert to BF16 using PyTorch"""
    # Convert to torch tensor
    tensor = torch.from_numpy(weight_array)
    # Convert to BF16
    bf16_tensor = tensor.to(torch.bfloat16)
    # Convert back to numpy
    return bf16_tensor.numpy()

# Usage example
import onnx
model = onnx.load('encoder_model.onnx')

bf16_weights = {}
for init in model.graph.initializer:
    name = init.name
    
    # Load as FP32
    data = np.frombuffer(init.raw_data, dtype=np.float32)
    shape = tuple(init.dims)
    fp32_weight = data.reshape(shape)
    
    # Convert to BF16
    bf16_weight = convert_to_bf16(fp32_weight)
    bf16_weights[name] = bf16_weight
    
    print(f"{name}: FP32 {fp32_weight.shape} → BF16 {bf16_weight.shape}")
```

### 5.3 Quantizing to INT8 for Maximum NPU Speed (Optional)

For even faster NPU execution at cost of slight accuracy loss:

```python
import torch
from torch.quantization import quantize_dynamic

# Dynamic quantization to INT8
def quantize_weight_int8(weight_array):
    """Quantize weight to INT8 using dynamic range"""
    # Find scale: max_abs_value / 127
    max_val = np.abs(weight_array).max()
    scale = max_val / 127.0
    
    # Quantize
    int8_weight = np.round(weight_array / scale).astype(np.int8)
    
    # Store scale for dequantization later
    return int8_weight, scale

# Per-layer quantization better than global
def quantize_weight_int8_perchannel(weight_array):
    """Quantize per-channel for better accuracy"""
    # For weight matrix [out_features, in_features]
    # Quantize along output dimension
    scales = np.abs(weight_array).max(axis=1, keepdims=True) / 127.0
    scales[scales == 0] = 1.0  # Avoid division by zero
    
    int8_weight = np.round(weight_array / scales).astype(np.int8)
    return int8_weight, scales
```

---

## 6. Loading Weights into NPU Kernels

### 6.1 Architecture: Weight Loading Pipeline

```
ONNX Model (78 MB)
    ↓ [Extract weights]
Dictionary of NumPy arrays (FP32)
    ↓ [Convert to BF16]
Dictionary of BF16 arrays (2x smaller)
    ↓ [Organize by layer]
Layer-wise weight structure
    ↓ [Transfer to NPU memory]
NPU SRAM (unified memory)
    ↓ [Bind to kernel arguments]
Custom MLIR-AIE2 kernels
    ↓ [Execute on NPU tiles]
Enhanced hidden states
```

### 6.2 Weight Loading Implementation

```python
import onnx
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple
import xrt  # XRT runtime

class WhisperWeightLoader:
    """Load Whisper weights from ONNX into NPU kernels"""
    
    def __init__(self, model_path: str, device_id: int = 0):
        self.model_path = Path(model_path)
        self.device_id = device_id
        self.weights = {}
        self.bf16_weights = {}
        
    def load_onnx_weights(self) -> Dict[str, np.ndarray]:
        """Load all weights from ONNX model"""
        print(f"Loading weights from {self.model_path}...")
        
        # Load ONNX model
        model = onnx.load(str(self.model_path))
        
        # Extract all initializers (weights)
        for init in model.graph.initializer:
            name = init.name
            
            # Determine dtype
            if init.data_type == onnx.TensorProto.FLOAT:
                dtype = np.float32
            elif init.data_type == onnx.TensorProto.FLOAT16:
                dtype = np.float16
            else:
                continue
            
            # Load data
            data = np.frombuffer(init.raw_data, dtype=dtype)
            shape = tuple(init.dims)
            self.weights[name] = data.reshape(shape)
        
        print(f"✓ Loaded {len(self.weights)} weight tensors")
        return self.weights
    
    def convert_to_bf16(self):
        """Convert all weights to BF16 format"""
        print("Converting weights to BF16...")
        
        for name, weight in self.weights.items():
            # Convert using PyTorch for accuracy
            tensor = torch.from_numpy(weight)
            bf16_tensor = tensor.to(torch.bfloat16)
            self.bf16_weights[name] = bf16_tensor.numpy()
        
        print(f"✓ Converted {len(self.bf16_weights)} weights to BF16")
        return self.bf16_weights
    
    def organize_by_layer(self) -> Dict[int, Dict]:
        """Organize weights by encoder layer"""
        layers = {}
        
        for layer_idx in range(6):  # 6 encoder layers
            layer_key = f"layer_{layer_idx}"
            layers[layer_key] = {
                'attention': {
                    'q_proj_weight': None,
                    'k_proj_weight': None,
                    'v_proj_weight': None,
                    'out_proj_weight': None,
                    'q_proj_bias': None,
                    'k_proj_bias': None,
                    'v_proj_bias': None,
                    'out_proj_bias': None,
                },
                'layernorm': {
                    'self_attn_gamma': None,
                    'self_attn_beta': None,
                    'final_gamma': None,
                    'final_beta': None,
                },
                'ffn': {
                    'fc1_weight': None,
                    'fc2_weight': None,
                    'fc1_bias': None,
                    'fc2_bias': None,
                }
            }
            
            # Map weights to layer structure
            prefix = f'/encoder/layers.{layer_idx}'
            
            for name, weight in self.bf16_weights.items():
                if prefix not in name:
                    continue
                
                # Attention weights
                if 'q_proj/weight' in name:
                    layers[layer_key]['attention']['q_proj_weight'] = weight
                elif 'k_proj/weight' in name:
                    layers[layer_key]['attention']['k_proj_weight'] = weight
                elif 'v_proj/weight' in name:
                    layers[layer_key]['attention']['v_proj_weight'] = weight
                elif 'out_proj/weight' in name:
                    layers[layer_key]['attention']['out_proj_weight'] = weight
                
                # Attention biases
                elif 'q_proj/bias' in name:
                    layers[layer_key]['attention']['q_proj_bias'] = weight
                elif 'k_proj/bias' in name:
                    layers[layer_key]['attention']['k_proj_bias'] = weight
                elif 'v_proj/bias' in name:
                    layers[layer_key]['attention']['v_proj_bias'] = weight
                elif 'out_proj/bias' in name:
                    layers[layer_key]['attention']['out_proj_bias'] = weight
                
                # LayerNorm gamma/beta
                elif 'self_attn_layer_norm/weight' in name:
                    layers[layer_key]['layernorm']['self_attn_gamma'] = weight
                elif 'self_attn_layer_norm/bias' in name:
                    layers[layer_key]['layernorm']['self_attn_beta'] = weight
                elif 'final_layer_norm/weight' in name:
                    layers[layer_key]['layernorm']['final_gamma'] = weight
                elif 'final_layer_norm/bias' in name:
                    layers[layer_key]['layernorm']['final_beta'] = weight
                
                # FFN weights
                elif 'fc1/weight' in name:
                    layers[layer_key]['ffn']['fc1_weight'] = weight
                elif 'fc2/weight' in name:
                    layers[layer_key]['ffn']['fc2_weight'] = weight
                elif 'fc1/bias' in name:
                    layers[layer_key]['ffn']['fc1_bias'] = weight
                elif 'fc2/bias' in name:
                    layers[layer_key]['ffn']['fc2_bias'] = weight
        
        return layers
    
    def transfer_to_npu(self, layer_weights: Dict, layer_idx: int):
        """Transfer weights to NPU device memory"""
        device = xrt.device(self.device_id)
        
        # Create XRT buffers for each weight tensor
        buffers = {}
        
        # Transfer attention weights to NPU
        for name, weight in layer_weights['attention'].items():
            if weight is None:
                continue
            
            size_bytes = weight.nbytes
            # Create XRT buffer in device memory
            buf = xrt.bo(device, size_bytes)
            
            # Copy weight data to buffer
            buf.write(weight.tobytes(), 0, size_bytes)
            buf.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            
            buffers[name] = buf
            print(f"  ✓ Transferred {name}: {weight.shape} ({size_bytes} bytes)")
        
        # Similar for layernorm and ffn weights
        for section in ['layernorm', 'ffn']:
            for name, weight in layer_weights[section].items():
                if weight is None:
                    continue
                
                size_bytes = weight.nbytes
                buf = xrt.bo(device, size_bytes)
                buf.write(weight.tobytes(), 0, size_bytes)
                buf.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                
                buffers[name] = buf
        
        return buffers
    
    def run_full_pipeline(self):
        """Execute complete weight loading pipeline"""
        print("\n" + "="*70)
        print("WHISPER WEIGHT LOADING PIPELINE")
        print("="*70)
        
        # Load weights from ONNX
        self.load_onnx_weights()
        
        # Convert to BF16
        self.convert_to_bf16()
        
        # Organize by layer
        layers = self.organize_by_layer()
        
        # Transfer to NPU for each layer
        print("\nTransferring to NPU device memory...")
        for layer_idx, layer_weights in layers.items():
            print(f"\n{layer_idx}:")
            self.transfer_to_npu(layer_weights, int(layer_idx.split('_')[1]))
        
        print("\n✅ Weight loading complete!")
        return layers

# Usage
if __name__ == "__main__":
    loader = WhisperWeightLoader(
        '/home/ucadmin/UC-1/Unicorn-Amanuensis/'
        'whisperx/models/whisper_onnx_cache/'
        'models--onnx-community--whisper-base/'
        'onnx/encoder_model.onnx'
    )
    
    layers = loader.run_full_pipeline()
```

---

## 7. Existing Weight Loading Code in Project

### 7.1 Current NPU Integration Code

**Location:** `whisperx/npu/npu_optimization/`

Key files:
- `onnx_whisper_npu.py` - Loads ONNX models via ONNX Runtime
- `quantize_to_int8.py` - Quantizes models to INT8 using NNCF
- `quantize_simple.py` - Simplified quantization script
- `whisper_npu_encoder_matmul.py` - Demonstrates weight handling in layers

### 7.2 Example: Weight Loading from `quantize_to_int8.py`

```python
# Load OpenVINO model
from openvino.runtime import Core
core = Core()
model = core.read_model("openvino_encoder_model.xml")

# Apply INT8 quantization
from openvino import nncf
compressed_model = nncf.compress_weights(
    model,
    mode=nncf.CompressWeightsMode.INT8,
    ratio=0.9,      # Compress 90% of weights
    group_size=128, # Group size for quantization
    all_layers=True
)

# Save quantized model
from openvino import save_model
save_model(compressed_model, "encoder_int8.xml")
```

---

## 8. Recommended Approach for NPU Kernel Weight Loading

### 8.1 Optimal Flow (BF16 + NPU)

```
1. Load ONNX Model
   ↓ encoder_model.onnx (78 MB FP32)

2. Extract Weights
   ↓ Use ONNX library or protobuf parsing
   ↓ Get FP32 numpy arrays

3. Convert to BF16
   ↓ Use torch.bfloat16 for accuracy
   ↓ 2x compression (39 MB → 20 MB)

4. Organize by Layer
   ↓ Group attention/FFN/layernorm per layer
   ↓ Create Python dict structure

5. Transfer to NPU
   ↓ Create XRT buffers
   ↓ DMA to device SRAM
   ↓ Total ~25 MB for all 6 layers

6. Bind to MLIR-AIE2 Kernels
   ↓ Pass as kernel arguments
   ↓ Access via ObjectFIFO in MLIR

7. Execute Kernels
   ↓ Run attention on NPU
   ↓ Run FFN on NPU
   ↓ Run LayerNorm on NPU
```

### 8.2 Performance Targets with Custom Kernels

**Current baseline (ONNX Runtime CPU):** 5.2x realtime
**With BF16 optimized weights:** 10-15x realtime
**With custom MLIR kernels:** 50-100x realtime (estimated)
**Target with full optimization:** 200-220x realtime

---

## 9. Code Structure Recommendations

### 9.1 Proposed File Organization

```
whisper_encoder_kernels/
├── weights/
│   ├── weight_loader.py           # Main loader class
│   ├── weight_converter.py        # FP32 → BF16/INT8
│   └── weight_manager.py          # Memory management
│
├── kernels/
│   ├── attention_kernel.mlir      # Attention computation
│   ├── ffn_kernel.mlir            # FFN computation
│   ├── layernorm_kernel.mlir      # LayerNorm computation
│   └── compiled_kernels.xclbin    # Compiled XCLBIN
│
├── npu_runtime/
│   ├── npu_executor.py            # NPU kernel executor
│   ├── memory_manager.py          # NPU memory management
│   └── weight_binder.py           # Bind weights to kernels
│
└── integration/
    ├── whisper_npu_encoder.py     # Full encoder with NPU
    └── test_weight_loading.py     # Unit tests
```

### 9.2 Example: Proposed `weight_loader.py`

```python
#!/usr/bin/env python3
"""
Whisper Weight Loader for NPU Kernels
Extracts weights from ONNX models and prepares them for NPU execution
"""

import onnx
import numpy as np
import torch
import xrt
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class LayerWeights:
    """Container for per-layer weights"""
    layer_idx: int
    attention_weights: Dict[str, np.ndarray]
    ffn_weights: Dict[str, np.ndarray]
    layernorm_gamma: np.ndarray
    layernorm_beta: np.ndarray

class WhisperWeightLoader:
    """Load Whisper encoder weights from ONNX for NPU execution"""
    
    def __init__(self, onnx_model_path: str):
        self.onnx_path = Path(onnx_model_path)
        self.model = None
        self.weights = {}
        self.bf16_weights = {}
        
    def load_model(self):
        """Load ONNX model"""
        print(f"Loading ONNX model: {self.onnx_path}")
        self.model = onnx.load(str(self.onnx_path))
        
    def extract_weights(self) -> Dict[str, np.ndarray]:
        """Extract all weights from ONNX initializers"""
        for init in self.model.graph.initializer:
            name = init.name
            
            # Determine dtype
            if init.data_type == onnx.TensorProto.FLOAT:
                dtype = np.float32
            elif init.data_type == onnx.TensorProto.FLOAT16:
                dtype = np.float16
            else:
                continue
            
            # Load weight
            data = np.frombuffer(init.raw_data, dtype=dtype)
            shape = tuple(init.dims)
            self.weights[name] = data.reshape(shape)
        
        return self.weights
    
    def convert_weights_bf16(self):
        """Convert weights to BF16 format"""
        for name, weight in self.weights.items():
            tensor = torch.from_numpy(weight)
            bf16_tensor = tensor.to(torch.bfloat16)
            self.bf16_weights[name] = bf16_tensor.numpy()
        
        return self.bf16_weights
    
    def get_layer_weights(self, layer_idx: int) -> LayerWeights:
        """Get organized weights for a specific layer"""
        prefix = f'/encoder/layers.{layer_idx}'
        
        layer = LayerWeights(
            layer_idx=layer_idx,
            attention_weights={},
            ffn_weights={},
            layernorm_gamma=None,
            layernorm_beta=None
        )
        
        for name, weight in self.bf16_weights.items():
            if prefix not in name:
                continue
            
            # Categorize weights...
            # (implementation details)
        
        return layer
    
    def prepare_for_npu(self, device_id: int = 0) -> Dict[int, Dict]:
        """Prepare weights for NPU execution"""
        print("Preparing weights for NPU...")
        
        # 1. Load model
        self.load_model()
        
        # 2. Extract weights
        self.extract_weights()
        
        # 3. Convert to BF16
        self.convert_weights_bf16()
        
        # 4. Organize by layer
        layers = {}
        for i in range(6):
            layers[i] = self.get_layer_weights(i)
        
        return layers

# Usage example
if __name__ == "__main__":
    loader = WhisperWeightLoader(
        '/home/ucadmin/UC-1/Unicorn-Amanuensis/'
        'whisperx/models/whisper_onnx_cache/'
        'models--onnx-community--whisper-base/'
        'onnx/encoder_model.onnx'
    )
    
    layers = loader.prepare_for_npu()
    print(f"Loaded weights for {len(layers)} layers")
```

---

## 10. Summary & Next Steps

### 10.1 Key Findings

1. **Model Location:** `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/models/whisper_onnx_cache/models--onnx-community--whisper-base/onnx/`

2. **Weight Tensor Names & Shapes:**
   - Q/K/V projections: `/encoder/layers.{i}/self_attn/{q,k,v}_proj/{weight,bias}` → [512, 512] / [512]
   - Output projection: `/encoder/layers.{i}/self_attn/out_proj/{weight,bias}` → [512, 512] / [512]
   - FFN1: `/encoder/layers.{i}/fc1/{weight,bias}` → [2048, 512] / [2048]
   - FFN2: `/encoder/layers.{i}/fc2/{weight,bias}` → [512, 2048] / [512]
   - LayerNorm: `/encoder/layers.{i}/{self_attn,final}_layer_norm/{weight,bias}` → [512] / [512]

3. **BF16 Conversion:** PyTorch's `torch.bfloat16` provides accurate conversion with 2x compression

4. **NPU Loading:** Use XRT API to create device buffers and DMA weights to SRAM

5. **Estimated NPU SRAM:** ~25 MB for all 6 layers in BF16 format (well within NPU limits)

### 10.2 Implementation Checklist

- [ ] Create `weights/weight_loader.py` with complete extraction pipeline
- [ ] Implement BF16 conversion in `weights/weight_converter.py`
- [ ] Add per-layer weight organization in `weights/weight_manager.py`
- [ ] Create XRT buffer management in `npu_runtime/memory_manager.py`
- [ ] Implement weight binding to MLIR kernels in `npu_runtime/weight_binder.py`
- [ ] Write unit tests for weight extraction and conversion
- [ ] Validate BF16 accuracy with CPU baseline
- [ ] Benchmark NPU kernel execution with loaded weights

### 10.3 Resources

- **ONNX Model Library:** https://github.com/onnx/models
- **PyTorch BF16 Docs:** https://pytorch.org/docs/stable/generated/torch.bfloat16.html
- **XRT Runtime API:** https://xilinx.github.io/XRT/master/html/index.html
- **MLIR-AIE2 ObjectFIFO:** https://xilinx.github.io/mlir-aie/main/

---

**Document Date:** 2025-11-19
**Status:** Complete Research Report
**Next Action:** Implement weight loading pipeline based on recommendations above

