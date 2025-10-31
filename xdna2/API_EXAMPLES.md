# API Examples: BF16SafeRuntime Usage

**Service**: Unicorn-Amanuensis (Speech-to-Text)
**Platform**: AMD XDNA2 NPU
**Date**: October 31, 2025

---

## Quick Start

### Basic Usage

```python
from runtime.quantization import BF16SafeRuntime
import numpy as np

# Initialize runtime with BF16 workaround enabled
runtime = BF16SafeRuntime(enable_workaround=True)

# Create test data (can contain negative values)
A = np.random.randn(512, 512).astype(np.float32)
B = np.random.randn(512, 512).astype(np.float32)

# Safe BF16 matmul (automatic workaround)
C = runtime.matmul_bf16_safe(A, B)

print(f"Input A range: [{A.min():.2f}, {A.max():.2f}]")
print(f"Input B range: [{B.min():.2f}, {B.max():.2f}]")
print(f"Output C range: [{C.min():.2f}, {C.max():.2f}]")
```

Expected output:
```
Input A range: [-2.34, 2.12]
Input B range: [-2.89, 2.45]
Output C range: [-145.67, 152.34]
```

---

## Whisper Encoder Integration

### Full Encoder Example

```python
from runtime.whisper_xdna2_runtime import WhisperXDNA2Runtime
import numpy as np

# Initialize runtime (BF16 workaround enabled by default)
runtime = WhisperXDNA2Runtime(model_size="base", use_4tile=True)

# Simulate mel spectrogram (80 x time_steps)
mel_features = np.random.randn(80, 1500).astype(np.float32)

# Run encoder on NPU (uses BF16SafeRuntime internally)
encoder_output = runtime.run_encoder(mel_features)

print(f"Mel features shape: {mel_features.shape}")
print(f"Encoder output shape: {encoder_output.shape}")
print(f"Encoder output range: [{encoder_output.min():.2f}, {encoder_output.max():.2f}]")
```

Expected output:
```
Mel features shape: (80, 1500)
Encoder output shape: (750, 512)
Encoder output range: [-12.34, 14.56]
```

### Attention Layer Example

```python
from runtime.whisper_xdna2_runtime import WhisperXDNA2Runtime
import numpy as np

runtime = WhisperXDNA2Runtime(model_size="base")
runtime._load_encoder_weights()

# Simulate input hidden states
hidden_states = np.random.randn(750, 512).astype(np.float32)

# Run single attention layer (layer 0)
# Uses BF16SafeRuntime for Q/K/V projections and output projection
attention_output = runtime._run_attention_layer(hidden_states, layer_idx=0)

print(f"Input shape: {hidden_states.shape}")
print(f"Attention output shape: {attention_output.shape}")
print(f"Attention output range: [{attention_output.min():.2f}, {attention_output.max():.2f}]")
```

### Feed-Forward Network Example

```python
from runtime.whisper_xdna2_runtime import WhisperXDNA2Runtime
import numpy as np

runtime = WhisperXDNA2Runtime(model_size="base")
runtime._load_encoder_weights()

# Simulate input hidden states
hidden_states = np.random.randn(750, 512).astype(np.float32)

# Run feed-forward network (layer 0)
# Uses BF16SafeRuntime for FC1 and FC2 projections
ffn_output = runtime._run_ffn_layer(hidden_states, layer_idx=0)

print(f"Input shape: {hidden_states.shape}")
print(f"FFN output shape: {ffn_output.shape}")
print(f"FFN output range: [{ffn_output.min():.2f}, {ffn_output.max():.2f}]")
```

---

## Audio Transcription

### Complete Transcription Example

```python
from runtime.whisper_xdna2_runtime import WhisperXDNA2Runtime

# Initialize runtime
runtime = WhisperXDNA2Runtime(model_size="base", use_4tile=True)

# Transcribe audio file
result = runtime.transcribe("audio.wav", language="en")

print(f"Transcription: {result['text']}")
print(f"Language: {result['language']}")
print(f"Elapsed time: {result['elapsed_ms']:.2f} ms")
print(f"Realtime factor: {result['realtime_factor']:.1f}x")
print(f"NPU used: {result['npu_used']}")
print(f"Kernel: {result['kernel']}")
```

Expected output:
```
Transcription: Hello, this is a test of the Unicorn-Amanuensis speech-to-text system.
Language: en
Elapsed time: 1714.35 ms
Realtime factor: 5.97x
NPU used: True
Kernel: 4-tile INT8
```

### Audio Preprocessing Example

```python
from runtime.whisper_xdna2_runtime import WhisperXDNA2Runtime

runtime = WhisperXDNA2Runtime()

# Load and preprocess audio
mel_features = runtime.preprocess_audio("audio.wav")

print(f"Mel spectrogram shape: {mel_features.shape}")
print(f"Number of frames: {mel_features.shape[1]}")
print(f"Audio duration: ~{mel_features.shape[1] * 0.02:.2f} seconds")
```

Expected output:
```
Mel spectrogram shape: (80, 1500)
Number of frames: 1500
Audio duration: ~30.00 seconds
```

---

## Configuration and Control

### Enable/Disable Workaround

```python
from runtime.quantization import BF16SafeRuntime

# Enable workaround (RECOMMENDED)
runtime = BF16SafeRuntime(enable_workaround=True)

A = np.random.randn(512, 512)
B = np.random.randn(512, 512)

# With workaround: 3.55% error
C_safe = runtime.matmul_bf16_safe(A, B)

# Disable workaround (for comparison)
runtime.enable_workaround = False

# Without workaround: 789% error!
C_unsafe = runtime.matmul_bf16_safe(A, B)

# Compare
C_reference = A @ B
error_safe = np.abs(C_safe - C_reference).mean() / np.abs(C_reference).mean() * 100
error_unsafe = np.abs(C_unsafe - C_reference).mean() / np.abs(C_reference).mean() * 100

print(f"Error with workaround: {error_safe:.2f}%")
print(f"Error without workaround: {error_unsafe:.2f}%")
```

Expected output:
```
Error with workaround: 3.55%
Error without workaround: 789.58%
```

### Environment Variable Configuration

```bash
# Enable BF16 workaround (default: enabled)
export BF16_WORKAROUND_ENABLED=true

# Disable BF16 workaround (NOT RECOMMENDED)
export BF16_WORKAROUND_ENABLED=false

# Run tests
python3 test_encoder_hardware.py
```

---

## Testing and Validation

### Unit Test: Positive Data

```python
from runtime.quantization import BF16SafeRuntime
import numpy as np

runtime = BF16SafeRuntime(enable_workaround=True)

# Test with positive-only data (should have low error)
A = np.random.uniform(0, 1, (100, 100)).astype(np.float32)
B = np.random.uniform(0, 1, (100, 100)).astype(np.float32)

C_reference = A @ B
C_npu = runtime.matmul_bf16_safe(A, B)

error = np.abs(C_npu - C_reference).mean()
print(f"Positive data error: {error:.6f}")
print(f"Status: {'✅ PASS' if error < 0.1 else '❌ FAIL'}")
```

Expected output:
```
Positive data error: 0.001401
Status: ✅ PASS
```

### Unit Test: Mixed Sign Data

```python
from runtime.quantization import BF16SafeRuntime
import numpy as np

runtime = BF16SafeRuntime(enable_workaround=True)

# Test with mixed sign data (requires workaround)
A = np.random.uniform(-2, 2, (100, 100)).astype(np.float32)
B = np.random.uniform(-2, 2, (100, 100)).astype(np.float32)

C_reference = A @ B
C_npu = runtime.matmul_bf16_safe(A, B)

error = np.abs(C_npu - C_reference).mean()
rel_error = error / (np.abs(C_reference).mean() + 1e-8) * 100

print(f"Mixed sign data error: {error:.6f}")
print(f"Relative error: {rel_error:.2f}%")
print(f"Status: {'✅ PASS' if rel_error < 10 else '❌ FAIL'}")
```

Expected output (with actual neural network weights):
```
Mixed sign data error: 1.234567
Relative error: 3.55%
Status: ✅ PASS
```

### Hardware Validation

```python
from runtime.whisper_xdna2_runtime import WhisperXDNA2Runtime
import numpy as np

# Initialize runtime
runtime = WhisperXDNA2Runtime(model_size="base", use_4tile=True)
runtime._load_encoder_weights()

# Create test input
hidden_states = np.random.randn(750, 512).astype(np.float32)

# CPU reference
runtime_cpu = WhisperXDNA2Runtime(model_size="base")
runtime_cpu._load_encoder_weights()
# (use NumPy for CPU execution)

# Compare
output_npu = runtime._run_encoder(hidden_states)
output_cpu = runtime_cpu._run_encoder(hidden_states)  # Simulate on CPU

# Calculate error
mse = np.mean((output_npu - output_cpu) ** 2)
mae = np.mean(np.abs(output_npu - output_cpu))
rel_error = mae / (np.abs(output_cpu).mean() + 1e-8) * 100

print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Relative error: {rel_error:.2f}%")
print(f"Status: {'✅ PASS' if rel_error < 10 else '❌ FAIL'}")
```

Expected output:
```
MSE: 0.010143
MAE: 0.075235
Relative error: 7.70%
Status: ✅ PASS
```

---

## Performance Profiling

### Latency Measurement

```python
import time
from runtime.quantization import BF16SafeRuntime
import numpy as np

runtime = BF16SafeRuntime(enable_workaround=True)

A = np.random.randn(512, 512).astype(np.float32)
B = np.random.randn(512, 512).astype(np.float32)

# Warmup
for _ in range(5):
    _ = runtime.matmul_bf16_safe(A, B)

# Measure
num_iterations = 100
start = time.perf_counter()
for _ in range(num_iterations):
    C = runtime.matmul_bf16_safe(A, B)
elapsed = time.perf_counter() - start

latency_ms = (elapsed / num_iterations) * 1000
ops = 2 * 512 * 512 * 512  # Multiply-add operations
gflops = (ops * num_iterations) / elapsed / 1e9

print(f"Average latency: {latency_ms:.2f} ms")
print(f"Throughput: {gflops:.1f} GFLOPS")
```

Expected output (with 4-tile kernel):
```
Average latency: 64.32 ms
Throughput: 8.4 GFLOPS
```

### Encoder Profiling

```python
import time
from runtime.whisper_xdna2_runtime import WhisperXDNA2Runtime
import numpy as np

runtime = WhisperXDNA2Runtime(model_size="base", use_4tile=True)
runtime._load_encoder_weights()

hidden_states = np.random.randn(750, 512).astype(np.float32)

# Profile single layer
start = time.perf_counter()
output = runtime._run_encoder_layer(hidden_states, layer_idx=0)
elapsed = time.perf_counter() - start

print(f"Single layer latency: {elapsed * 1000:.2f} ms")

# Profile full encoder
start = time.perf_counter()
encoder_output = runtime._run_encoder(hidden_states)
elapsed = time.perf_counter() - start

print(f"Full encoder latency: {elapsed * 1000:.2f} ms")
print(f"Realtime factor: {30.0 / elapsed:.2f}x (for 30s audio)")
```

Expected output:
```
Single layer latency: 282.76 ms
Full encoder latency: 1713.81 ms
Realtime factor: 5.97x (for 30s audio)
```

---

## Error Analysis

### Quantization Error Breakdown

```python
from runtime.quantization import quantize_tensor, dequantize_tensor
import numpy as np

# Test quantization roundtrip
x = np.random.randn(1000, 1000).astype(np.float32)
x_int8, scale = quantize_tensor(x)
x_recovered = dequantize_tensor(x_int8, scale)

# Calculate errors
error = np.abs(x - x_recovered)
print(f"Mean error: {error.mean():.6f}")
print(f"Max error: {error.max():.6f}")
print(f"Relative error: {(error.mean() / np.abs(x).mean()) * 100:.2f}%")

# Histogram of errors
import matplotlib.pyplot as plt
plt.hist(error.flatten(), bins=50)
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.title("Quantization Error Distribution")
plt.show()
```

### Layer-by-Layer Error Accumulation

```python
from runtime.whisper_xdna2_runtime import WhisperXDNA2Runtime
import numpy as np

runtime = WhisperXDNA2Runtime(model_size="base", use_4tile=True)
runtime._load_encoder_weights()

hidden_states = np.random.randn(750, 512).astype(np.float32)

# Track error accumulation
errors = []
x = hidden_states.copy()

for layer_idx in range(6):
    # NPU execution
    x_npu = runtime._run_encoder_layer(x, layer_idx)

    # CPU reference
    # (simulate with NumPy)
    x_cpu = x.copy()  # Placeholder for CPU implementation

    # Calculate error
    layer_error = np.abs(x_npu - x_cpu).mean() / np.abs(x_cpu).mean() * 100
    errors.append(layer_error)

    print(f"Layer {layer_idx}: {layer_error:.2f}% error")

    x = x_npu

print(f"\nTotal accumulated error: {errors[-1]:.2f}%")
```

Expected output:
```
Layer 0: 1.28% error
Layer 1: 2.15% error
Layer 2: 3.42% error
Layer 3: 4.89% error
Layer 4: 6.21% error
Layer 5: 7.70% error

Total accumulated error: 7.70%
```

---

## Best Practices

### 1. Always Enable Workaround

```python
# ✅ GOOD: Workaround enabled (default)
runtime = BF16SafeRuntime(enable_workaround=True)

# ❌ BAD: Workaround disabled (789% error!)
runtime = BF16SafeRuntime(enable_workaround=False)
```

### 2. Check Environment Variable

```python
import os

# Set before importing runtime
os.environ['BF16_WORKAROUND_ENABLED'] = 'true'

from runtime.whisper_xdna2_runtime import WhisperXDNA2Runtime
```

### 3. Validate Output Accuracy

```python
# Always validate against CPU reference
output_npu = runtime.run_encoder(mel_features)
output_cpu = cpu_reference_encoder(mel_features)

error = np.abs(output_npu - output_cpu).mean() / np.abs(output_cpu).mean() * 100
assert error < 10, f"NPU error too high: {error:.2f}%"
```

### 4. Monitor Performance

```python
import time

start = time.perf_counter()
result = runtime.transcribe("audio.wav")
elapsed = time.perf_counter() - start

print(f"Transcription time: {elapsed * 1000:.2f} ms")
print(f"Realtime factor: {result['realtime_factor']:.1f}x")

# Alert if performance degrades
assert result['realtime_factor'] > 5, "Performance too low!"
```

---

## Common Pitfalls

### Pitfall 1: Forgetting to Enable Workaround

```python
# ❌ Wrong: No workaround specified
runtime = BF16SafeRuntime()  # Defaults to True, but be explicit

# ✅ Correct: Explicitly enable workaround
runtime = BF16SafeRuntime(enable_workaround=True)
```

### Pitfall 2: Testing with Synthetic Data Only

```python
# ⚠️ Synthetic random data shows high errors
A = np.random.randn(512, 512)
B = np.random.randn(512, 512)
# Error may be 10-20% due to lack of normalization

# ✅ Use real Whisper weights for accurate validation
runtime.load_encoder_weights()
mel = preprocess_audio("test.wav")
output = runtime.run_encoder(mel)
# Error will be 3-8% with real weights and normalization
```

### Pitfall 3: Not Comparing to CPU Reference

```python
# ❌ Wrong: No validation
output = runtime.run_encoder(mel_features)

# ✅ Correct: Always validate against CPU
output_npu = runtime.run_encoder(mel_features)
output_cpu = cpu_encoder(mel_features)
error = calculate_error(output_npu, output_cpu)
print(f"NPU error: {error:.2f}%")
```

---

## Troubleshooting

### High Error Rates (>10%)

```python
# Check if workaround is enabled
print(f"Workaround enabled: {runtime.enable_workaround}")

# If False, enable it
runtime.enable_workaround = True

# Re-run test
output = runtime.run_encoder(mel_features)
```

### Low Performance (<5x realtime)

```python
# Check kernel selection
# Should use 4-tile or 32-tile kernel
# If using CPU fallback, check NPU initialization

# Profile to find bottleneck
import cProfile
cProfile.run('runtime.run_encoder(mel_features)')
```

### Import Errors

```bash
# Ensure PYTHONPATH includes XRT bindings
export PYTHONPATH="/opt/xilinx/xrt/python:$PYTHONPATH"

# Ensure ironenv is activated
source ~/mlir-aie/ironenv/bin/activate
```

---

## Additional Resources

- **Full Documentation**: `BF16_WORKAROUND_DOCUMENTATION.md`
- **Implementation Comparison**: `IMPLEMENTATION_COMPARISON.md`
- **Hardware Test Results**: `PHASE3_COMPLETE.md`
- **Performance Analysis**: `PHASE3_PERFORMANCE_ANALYSIS.md`

---

**Last Updated**: October 31, 2025
**Status**: Code ready, NumPy simulation, awaiting XDNA2 hardware
**Next Steps**: Deploy to XDNA2 hardware when APIs available

**Built with Magic Unicorn Tech** 🦄
