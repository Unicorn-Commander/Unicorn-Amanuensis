#!/usr/bin/env python3
"""
PyTorch Baseline Comparison Test for C++ Encoder Numerical Accuracy

This test validates the C++ encoder implementation against the official
PyTorch Whisper implementation by comparing outputs using:
- Cosine similarity (expect >0.99)
- Mean absolute error (expect <1.0)
- Max absolute error
- Element-wise relative error percentage

Both implementations use the SAME random input (seed=42) for fair comparison.
"""

import numpy as np
import torch
import ctypes
from ctypes import c_void_p, c_float, c_int, c_size_t, POINTER
import sys
from pathlib import Path

print("="*80)
print("  PYTORCH VS C++ ENCODER ACCURACY VALIDATION TEST")
print("="*80)

# ============================================================================
# 1. LOAD PYTORCH WHISPER ENCODER
# ============================================================================

print(f"\n{'='*80}")
print("  LOADING PYTORCH WHISPER BASE ENCODER")
print(f"{'='*80}")

try:
    from transformers import WhisperModel
    print("✅ Transformers library loaded")
except ImportError:
    print("❌ Transformers not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import WhisperModel
    print("✅ Transformers installed and loaded")

# Load Whisper Base model
model_name = "openai/whisper-base"
print(f"\n📥 Loading model: {model_name}")

try:
    model = WhisperModel.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Extract encoder
encoder = model.encoder
print(f"✅ Encoder extracted (6 layers)")

# Get configuration
config = model.config
n_layers = config.encoder_layers
n_heads = config.encoder_attention_heads
n_state = config.d_model
ffn_dim = config.encoder_ffn_dim
seq_len = 512  # Fixed for Whisper Base

print(f"\nEncoder Configuration:")
print(f"  Layers:        {n_layers}")
print(f"  Attention Heads: {n_heads}")
print(f"  Hidden Size:   {n_state}")
print(f"  FFN Dim:       {ffn_dim}")
print(f"  Sequence Len:  {seq_len}")

# ============================================================================
# 2. PREPARE TEST INPUT (SAME SEED FOR BOTH)
# ============================================================================

print(f"\n{'='*80}")
print("  PREPARING TEST INPUT (SEED=42)")
print(f"{'='*80}")

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create random input (matching C++ test)
input_np = np.random.randn(seq_len, n_state).astype(np.float32)
input_torch = torch.from_numpy(input_np).unsqueeze(0)  # Add batch dimension

print(f"✅ Input shape: {input_np.shape}")
print(f"  Mean: {input_np.mean():.4f}")
print(f"  Std:  {input_np.std():.4f}")
print(f"  Min:  {input_np.min():.4f}")
print(f"  Max:  {input_np.max():.4f}")

# ============================================================================
# 3. RUN PYTORCH ENCODER
# ============================================================================

print(f"\n{'='*80}")
print("  RUNNING PYTORCH ENCODER (6 LAYERS)")
print(f"{'='*80}")

# Note: We're testing the encoder layers directly, bypassing the conv layers
# The C++ implementation tests the transformer layers only (post-convolution)
# So we feed the random input directly to the encoder layers

with torch.no_grad():
    # Start with input and run through each encoder layer manually
    hidden_states = input_torch

    # Apply initial layer norm (if exists in encoder)
    if hasattr(encoder, 'layer_norm') and encoder.layer_norm is not None:
        hidden_states = encoder.layer_norm(hidden_states)

    # Run through all 6 encoder layers
    for layer_idx, layer in enumerate(encoder.layers):
        layer_output = layer(
            hidden_states,
            attention_mask=None,
            layer_head_mask=None,
            output_attentions=False
        )
        hidden_states = layer_output[0]  # Extract hidden states
        print(f"✅ Layer {layer_idx} complete")

    output_torch = hidden_states.squeeze(0).numpy()

print(f"\n✅ PyTorch encoder complete")
print(f"  Output shape: {output_torch.shape}")
print(f"  Mean: {output_torch.mean():.4f}")
print(f"  Std:  {output_torch.std():.4f}")
print(f"  Min:  {output_torch.min():.4f}")
print(f"  Max:  {output_torch.max():.4f}")

# ============================================================================
# 4. LOAD C++ ENCODER LIBRARY
# ============================================================================

print(f"\n{'='*80}")
print("  LOADING C++ ENCODER LIBRARY")
print(f"{'='*80}")

lib_path = "/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build/libwhisper_encoder_cpp.so"

if not Path(lib_path).exists():
    print(f"❌ C++ library not found: {lib_path}")
    print("   Please build the C++ encoder first:")
    print("   cd cpp/build && cmake .. && make")
    sys.exit(1)

lib = ctypes.CDLL(lib_path)
print(f"✅ Loaded: {lib_path}")

# Define C API
lib.encoder_layer_create.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t]
lib.encoder_layer_create.restype = c_void_p

lib.encoder_layer_destroy.argtypes = [c_void_p]

lib.encoder_layer_load_weights.argtypes = [
    c_void_p,
    POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    c_size_t, c_size_t
]
lib.encoder_layer_load_weights.restype = c_int

lib.encoder_layer_forward.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float), c_size_t, c_size_t]
lib.encoder_layer_forward.restype = c_int

print(f"✅ C API bindings complete")

# ============================================================================
# 5. LOAD REAL WHISPER WEIGHTS INTO C++ ENCODER
# ============================================================================

print(f"\n{'='*80}")
print("  LOADING REAL WHISPER WEIGHTS INTO C++ ENCODER")
print(f"{'='*80}")

weights_dir = Path("./weights/whisper_base_fp32")
if not weights_dir.exists():
    print(f"❌ Weights directory not found: {weights_dir}")
    print("   Please run extract_whisper_weights.py first")
    sys.exit(1)

def load_weight(name):
    """Load a weight tensor from NumPy file."""
    path = weights_dir / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Weight not found: {path}")
    return np.load(path).astype(np.float32)

# Create 6 encoder layers
layers = []

for layer_idx in range(6):
    print(f"\nLayer {layer_idx}:")

    # Create layer
    handle = lib.encoder_layer_create(layer_idx, n_heads, n_state, ffn_dim)
    if not handle:
        print(f"❌ Failed to create layer {layer_idx}")
        sys.exit(1)

    try:
        # Load real weights for this layer
        # Attention weights (transpose for C++)
        q_w = load_weight(f"layers_{layer_idx}_self_attn_q_proj_weight").T
        k_w = load_weight(f"layers_{layer_idx}_self_attn_k_proj_weight").T
        v_w = load_weight(f"layers_{layer_idx}_self_attn_v_proj_weight").T
        out_w = load_weight(f"layers_{layer_idx}_self_attn_out_proj_bias")  # Check if needs transpose

        # Check if out_proj has weight (not just bias)
        try:
            out_w = load_weight(f"layers_{layer_idx}_self_attn_out_proj_weight").T
        except:
            print(f"  ⚠️  out_proj_weight not found, using zeros")
            out_w = np.zeros((n_state, n_state), dtype=np.float32)

        # Attention biases
        q_b = load_weight(f"layers_{layer_idx}_self_attn_q_proj_bias")
        k_b = np.zeros(n_state, dtype=np.float32)  # K projection has no bias in Whisper
        v_b = load_weight(f"layers_{layer_idx}_self_attn_v_proj_bias")

        try:
            out_b = load_weight(f"layers_{layer_idx}_self_attn_out_proj_bias")
        except:
            out_b = np.zeros(n_state, dtype=np.float32)

        # FFN weights
        fc1_w = load_weight(f"layers_{layer_idx}_fc1_weight").T
        fc2_w = load_weight(f"layers_{layer_idx}_fc2_weight").T
        fc1_b = load_weight(f"layers_{layer_idx}_fc1_bias")
        fc2_b = load_weight(f"layers_{layer_idx}_fc2_bias")

        # Layer norm weights
        attn_ln_w = load_weight(f"layers_{layer_idx}_self_attn_layer_norm_weight")
        attn_ln_b = load_weight(f"layers_{layer_idx}_self_attn_layer_norm_bias")
        ffn_ln_w = load_weight(f"layers_{layer_idx}_final_layer_norm_weight")
        ffn_ln_b = load_weight(f"layers_{layer_idx}_final_layer_norm_bias")

        # Load into C++
        result = lib.encoder_layer_load_weights(
            handle,
            q_w.ctypes.data_as(POINTER(c_float)),
            k_w.ctypes.data_as(POINTER(c_float)),
            v_w.ctypes.data_as(POINTER(c_float)),
            out_w.ctypes.data_as(POINTER(c_float)),
            q_b.ctypes.data_as(POINTER(c_float)),
            k_b.ctypes.data_as(POINTER(c_float)),
            v_b.ctypes.data_as(POINTER(c_float)),
            out_b.ctypes.data_as(POINTER(c_float)),
            fc1_w.ctypes.data_as(POINTER(c_float)),
            fc2_w.ctypes.data_as(POINTER(c_float)),
            fc1_b.ctypes.data_as(POINTER(c_float)),
            fc2_b.ctypes.data_as(POINTER(c_float)),
            attn_ln_w.ctypes.data_as(POINTER(c_float)),
            attn_ln_b.ctypes.data_as(POINTER(c_float)),
            ffn_ln_w.ctypes.data_as(POINTER(c_float)),
            ffn_ln_b.ctypes.data_as(POINTER(c_float)),
            n_state,
            ffn_dim
        )

        if result != 0:
            print(f"❌ Failed to load weights for layer {layer_idx}")
            sys.exit(1)

        layers.append(handle)
        print(f"  ✅ Layer {layer_idx} loaded with real Whisper weights")

    except Exception as e:
        print(f"❌ Failed to load weights for layer {layer_idx}: {e}")
        sys.exit(1)

print(f"\n✅ All 6 layers loaded with REAL WHISPER BASE WEIGHTS!")

# ============================================================================
# 6. RUN C++ ENCODER
# ============================================================================

print(f"\n{'='*80}")
print("  RUNNING C++ ENCODER (6 LAYERS)")
print(f"{'='*80}")

# Prepare buffers
current = input_np.copy()
buffer_a = np.zeros((seq_len, n_state), dtype=np.float32)

# Run all 6 layers sequentially
for layer_idx, handle in enumerate(layers):
    result = lib.encoder_layer_forward(
        handle,
        current.ctypes.data_as(POINTER(c_float)),
        buffer_a.ctypes.data_as(POINTER(c_float)),
        seq_len,
        n_state
    )

    if result != 0:
        print(f"❌ Forward pass failed at layer {layer_idx}")
        sys.exit(1)

    current = buffer_a.copy()
    print(f"✅ Layer {layer_idx} complete")

output_cpp = current

print(f"\n✅ C++ encoder complete")
print(f"  Output shape: {output_cpp.shape}")
print(f"  Mean: {output_cpp.mean():.4f}")
print(f"  Std:  {output_cpp.std():.4f}")
print(f"  Min:  {output_cpp.min():.4f}")
print(f"  Max:  {output_cpp.max():.4f}")

# ============================================================================
# 7. COMPARE OUTPUTS
# ============================================================================

print(f"\n{'='*80}")
print("  NUMERICAL ACCURACY COMPARISON")
print(f"{'='*80}")

# Cosine similarity
def cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot_product = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    return dot_product / (norm_a * norm_b)

cos_sim = cosine_similarity(output_torch, output_cpp)

# Absolute errors
abs_diff = np.abs(output_torch - output_cpp)
mae = abs_diff.mean()
max_abs_error = abs_diff.max()

# Relative errors (avoid division by zero)
epsilon = 1e-8
rel_error = abs_diff / (np.abs(output_torch) + epsilon)
mean_rel_error = rel_error.mean()
max_rel_error = rel_error.max()

# Element-wise accuracy
threshold = 0.01  # 1% relative error threshold
accurate_elements = np.sum(rel_error < threshold)
total_elements = rel_error.size
accuracy_pct = (accurate_elements / total_elements) * 100

print(f"\n📊 Similarity Metrics:")
print(f"  Cosine Similarity:       {cos_sim:.6f}")
print(f"  Target:                  >0.99")
print(f"  Status:                  {'✅ PASS' if cos_sim > 0.99 else '❌ FAIL'}")

print(f"\n📊 Absolute Error Metrics:")
print(f"  Mean Absolute Error:     {mae:.6f}")
print(f"  Max Absolute Error:      {max_abs_error:.6f}")
print(f"  Target MAE:              <1.0")
print(f"  Status:                  {'✅ PASS' if mae < 1.0 else '❌ FAIL'}")

print(f"\n📊 Relative Error Metrics:")
print(f"  Mean Relative Error:     {mean_rel_error*100:.4f}%")
print(f"  Max Relative Error:      {max_rel_error*100:.4f}%")

print(f"\n📊 Element-wise Accuracy:")
print(f"  Elements < 1% error:     {accurate_elements:,} / {total_elements:,}")
print(f"  Accuracy:                {accuracy_pct:.2f}%")
print(f"  Target:                  >99%")
print(f"  Status:                  {'✅ PASS' if accuracy_pct > 99 else '❌ FAIL'}")

# ============================================================================
# 8. DETAILED ANALYSIS
# ============================================================================

print(f"\n{'='*80}")
print("  DETAILED ANALYSIS")
print(f"{'='*80}")

# Find regions with largest discrepancies
worst_indices = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
print(f"\n🔍 Largest discrepancy at position {worst_indices}:")
print(f"  PyTorch:  {output_torch[worst_indices]:.6f}")
print(f"  C++:      {output_cpp[worst_indices]:.6f}")
print(f"  Diff:     {abs_diff[worst_indices]:.6f}")
print(f"  Rel Err:  {rel_error[worst_indices]*100:.4f}%")

# Statistics on error distribution
print(f"\n📈 Error Distribution:")
percentiles = [50, 90, 95, 99, 99.9]
for p in percentiles:
    val = np.percentile(abs_diff, p)
    print(f"  {p:5.1f}th percentile:      {val:.6f}")

# Check for numerical instabilities
has_nan_torch = np.isnan(output_torch).any()
has_nan_cpp = np.isnan(output_cpp).any()
has_inf_torch = np.isinf(output_torch).any()
has_inf_cpp = np.isinf(output_cpp).any()

print(f"\n⚠️  Numerical Stability:")
print(f"  PyTorch NaN:             {'❌ FOUND' if has_nan_torch else '✅ None'}")
print(f"  C++ NaN:                 {'❌ FOUND' if has_nan_cpp else '✅ None'}")
print(f"  PyTorch Inf:             {'❌ FOUND' if has_inf_torch else '✅ None'}")
print(f"  C++ Inf:                 {'❌ FOUND' if has_inf_cpp else '✅ None'}")

# ============================================================================
# 9. FINAL VERDICT
# ============================================================================

print(f"\n{'='*80}")
print("  FINAL VERDICT")
print(f"{'='*80}")

all_tests_pass = (
    cos_sim > 0.99 and
    mae < 1.0 and
    accuracy_pct > 99.0 and
    not has_nan_cpp and
    not has_inf_cpp
)

if all_tests_pass:
    print(f"\n✅ ACCURACY VALIDATION: PASS")
    print(f"\n🎉 C++ encoder matches PyTorch implementation!")
    print(f"   - Cosine similarity: {cos_sim:.6f} (>0.99 ✅)")
    print(f"   - Mean abs error:    {mae:.6f} (<1.0 ✅)")
    print(f"   - Element accuracy:  {accuracy_pct:.2f}% (>99% ✅)")
    print(f"\n✅ The C++ encoder is numerically accurate and ready for production!")
else:
    print(f"\n❌ ACCURACY VALIDATION: FAIL")
    print(f"\n⚠️  C++ encoder has accuracy issues:")

    if cos_sim <= 0.99:
        print(f"   ❌ Cosine similarity too low: {cos_sim:.6f} (need >0.99)")

    if mae >= 1.0:
        print(f"   ❌ Mean absolute error too high: {mae:.6f} (need <1.0)")

    if accuracy_pct <= 99.0:
        print(f"   ❌ Element accuracy too low: {accuracy_pct:.2f}% (need >99%)")

    if has_nan_cpp or has_inf_cpp:
        print(f"   ❌ Numerical instabilities detected (NaN/Inf)")

    print(f"\n🔧 Recommendations:")
    print(f"   1. Check weight transposition (PyTorch uses different memory layout)")
    print(f"   2. Verify layer norm epsilon matches (PyTorch default: 1e-5)")
    print(f"   3. Check attention mask implementation")
    print(f"   4. Verify softmax numerical stability")
    print(f"   5. Check residual connection ordering")

# ============================================================================
# 10. SAVE COMPARISON DATA
# ============================================================================

print(f"\n{'='*80}")
print("  SAVING COMPARISON DATA")
print(f"{'='*80}")

output_dir = Path("./accuracy_comparison")
output_dir.mkdir(exist_ok=True)

np.save(output_dir / "input.npy", input_np)
np.save(output_dir / "output_pytorch.npy", output_torch)
np.save(output_dir / "output_cpp.npy", output_cpp)
np.save(output_dir / "abs_diff.npy", abs_diff)
np.save(output_dir / "rel_error.npy", rel_error)

# Save metrics to text file
with open(output_dir / "metrics.txt", "w") as f:
    f.write("PyTorch vs C++ Encoder Accuracy Comparison\n")
    f.write("="*80 + "\n\n")
    f.write(f"Cosine Similarity:       {cos_sim:.6f}\n")
    f.write(f"Mean Absolute Error:     {mae:.6f}\n")
    f.write(f"Max Absolute Error:      {max_abs_error:.6f}\n")
    f.write(f"Mean Relative Error:     {mean_rel_error*100:.4f}%\n")
    f.write(f"Max Relative Error:      {max_rel_error*100:.4f}%\n")
    f.write(f"Element Accuracy:        {accuracy_pct:.2f}%\n")
    f.write(f"\nTest Result:             {'PASS' if all_tests_pass else 'FAIL'}\n")

print(f"✅ Comparison data saved to: {output_dir}")

# ============================================================================
# CLEANUP
# ============================================================================

for handle in layers:
    lib.encoder_layer_destroy(handle)

print(f"\n{'='*80}")
print("  TEST COMPLETE")
print(f"{'='*80}")
print(f"\nResult: {'✅ PASS' if all_tests_pass else '❌ FAIL'}")
print(f"Cosine Similarity: {cos_sim:.6f}")
print(f"Mean Absolute Error: {mae:.6f}")
print(f"Element Accuracy: {accuracy_pct:.2f}%")
print(f"\n{'='*80}")
