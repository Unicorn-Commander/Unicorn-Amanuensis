//===- softmax_streaming_bf16.cc ---------------------------------*- C++ -*-===//
//
// Streaming Softmax for Attention - processes 1500 elements (full sequence)
// Two-pass algorithm: 1) find max + exp, 2) normalize
//
// Input: 1500 bfloat16 elements (3000 bytes - fits in 32KB tile)
// Output: 1500 bfloat16 elements
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SEQ_LENGTH 1500  // Whisper sequence length

extern "C" {

void softmax_streaming_bf16(bfloat16* __restrict input,
                            bfloat16* __restrict output) {
  // Use AIE2 vector units (32-wide for bf16)
  constexpr int vec_factor = 32;
  constexpr int num_full_vectors = SEQ_LENGTH / vec_factor;  // 1500/32 = 46
  constexpr int remainder = SEQ_LENGTH % vec_factor;  // 1500 % 32 = 28

  // Pass 1: Find maximum value
  float max_val = -INFINITY;

  for (int i = 0; i < num_full_vectors; i++) {
    aie::vector<bfloat16, vec_factor> v = aie::load_v<vec_factor>(input + i * vec_factor);
    for (int j = 0; j < vec_factor; j++) {
      float val = (float)v[j];
      if (val > max_val) max_val = val;
    }
  }

  // Handle remainder
  for (int i = 0; i < remainder; i++) {
    float val = (float)input[num_full_vectors * vec_factor + i];
    if (val > max_val) max_val = val;
  }

  // Pass 2: Compute exp(x - max) and sum
  float sum = 0.0f;

  for (int i = 0; i < num_full_vectors; i++) {
    aie::vector<bfloat16, vec_factor> v = aie::load_v<vec_factor>(input + i * vec_factor);
    aie::vector<bfloat16, vec_factor> result;

    for (int j = 0; j < vec_factor; j++) {
      float val = (float)v[j] - max_val;
      float exp_val = expf(val);
      result[j] = (bfloat16)exp_val;
      sum += exp_val;
    }

    aie::store_v(output + i * vec_factor, result);
  }

  // Handle remainder
  for (int i = 0; i < remainder; i++) {
    int idx = num_full_vectors * vec_factor + i;
    float val = (float)input[idx] - max_val;
    float exp_val = expf(val);
    output[idx] = (bfloat16)exp_val;
    sum += exp_val;
  }

  // Pass 3: Normalize by sum
  float inv_sum = 1.0f / sum;

  for (int i = 0; i < num_full_vectors; i++) {
    aie::vector<bfloat16, vec_factor> v = aie::load_v<vec_factor>(output + i * vec_factor);
    aie::vector<bfloat16, vec_factor> result;

    for (int j = 0; j < vec_factor; j++) {
      result[j] = (bfloat16)((float)v[j] * inv_sum);
    }

    aie::store_v(output + i * vec_factor, result);
  }

  // Handle remainder
  for (int i = 0; i < remainder; i++) {
    int idx = num_full_vectors * vec_factor + i;
    output[idx] = (bfloat16)((float)output[idx] * inv_sum);
  }
}

}  // extern "C"
