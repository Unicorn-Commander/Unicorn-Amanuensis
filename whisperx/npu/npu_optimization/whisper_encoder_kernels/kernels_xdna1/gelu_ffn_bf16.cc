//===- gelu_ffn_bf16.cc ------------------------------------------*- C++ -*-===//
//
// GELU Activation for FFN - processes 2048 elements (FFN intermediate dimension)
// Uses Gaussian Error Linear Unit: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//
// Input: 2048 bfloat16 elements (4096 bytes - fits in 32KB tile with headroom)
// Output: 2048 bfloat16 elements
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FFN_INTERMEDIATE_DIM 2048  // Whisper FFN intermediate dimension

extern "C" {

void gelu_ffn_bf16(bfloat16* __restrict input,
                   bfloat16* __restrict output) {
  // GELU constants
  const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/π)
  const float coeff = 0.044715f;
  const float half = 0.5f;

  // Use AIE2 vector units (32-wide for bf16)
  constexpr int vec_factor = 32;
  constexpr int num_vectors = FFN_INTERMEDIATE_DIM / vec_factor;  // 2048/32 = 64

  // Process in vectorized chunks
  for (int i = 0; i < num_vectors; i++) {
    aie::vector<bfloat16, vec_factor> v = aie::load_v<vec_factor>(input + i * vec_factor);
    aie::vector<bfloat16, vec_factor> result;

    // GELU formula for each element
    for (int j = 0; j < vec_factor; j++) {
      float x = (float)v[j];
      float x_cubed = x * x * x;
      float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
      float tanh_val = tanhf(inner);
      float gelu_val = x * half * (1.0f + tanh_val);
      result[j] = (bfloat16)gelu_val;
    }

    aie::store_v(output + i * vec_factor, result);
  }
}

}  // extern "C"
