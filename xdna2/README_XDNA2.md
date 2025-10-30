# XDNA2 Implementation Guide

## Overview

This directory contains XDNA2-specific optimizations for Strix Point NPU (AMD Ryzen AI 300 series).

## Architecture

```
xdna2/
├── kernels/               # Custom NPU kernels
│   ├── matmul_int8.py    # INT8 matrix multiplication
│   ├── softmax.py        # Softmax operation
│   ├── attention.py      # Attention mechanism
│   └── quantize.py       # Model quantization utilities
├── runtime/              # Runtime integration
│   ├── xdna2_runtime.py  # XDNA2 device management
│   ├── model_loader.py   # NPU-optimized model loading
│   └── buffer_mgr.py     # Buffer management for NPU
└── server.py            # XDNA2-optimized server (WIP)
```

## Development Status

| Component | Status | Notes |
|-----------|--------|-------|
| INT8 Kernels | 🚧 Planned | Shared with CC-1L kernels |
| Runtime Integration | 🚧 Planned | XDNA2 device initialization |
| Model Quantization | 🚧 Planned | Whisper → INT8 conversion |
| Server Implementation | 🚧 Planned | NPU-accelerated inference |

## Integration with CC-1L

XDNA2 kernels will be symlinked from CC-1L's kernel library:

```bash
# From CC-1L
ln -s /home/ccadmin/CC-1L/kernels/matmul_int8.py kernels/matmul_int8.py
ln -s /home/ccadmin/CC-1L/kernels/softmax.py kernels/softmax.py
```

This ensures kernel development happens centrally in CC-1L and is shared across services.

## Implementation Plan

### Phase 1: Runtime Setup
1. Initialize XDNA2 device
2. Implement buffer management
3. Create device context management

### Phase 2: Model Optimization
1. Quantize Whisper model to INT8
2. Create NPU-compatible model format
3. Implement model loading pipeline

### Phase 3: Kernel Integration
1. Integrate INT8 matmul from CC-1L
2. Implement attention mechanism
3. Add softmax operation

### Phase 4: Server Implementation
1. Port server.py to use NPU kernels
2. Implement NPU inference pipeline
3. Add fallback to XDNA1/CPU

## Performance Targets

| Metric | XDNA1 | XDNA2 Target | Improvement |
|--------|-------|--------------|-------------|
| Inference Time | ~100ms | ~50ms | 2x |
| Power Usage | ~15W | ~8W | ~50% |
| Accuracy | 95% | 95% | Same |

## Testing

```bash
# Set platform override
export NPU_PLATFORM=xdna2

# Run tests
python -m pytest tests/test_xdna2.py
```

## References

- [CC-1L Kernel Library](../../../CC-1L/kernels/)
- [XDNA2 Documentation](https://www.xilinx.com/products/design-tools/vitis/xdna.html)
- [Whisper Model](https://github.com/openai/whisper)
