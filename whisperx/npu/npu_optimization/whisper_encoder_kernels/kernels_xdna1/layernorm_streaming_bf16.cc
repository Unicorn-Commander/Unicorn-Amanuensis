//===- layernorm_streaming_bf16.cc --------------------------------*- C++ -*-===//
//
// Streaming LayerNorm for Whisper Encoder - processes 512 elements at a time
// Designed for full sequence streaming (1500 chunks of 512 elements)
//
// Input: 512 bfloat16 elements (1024 bytes)
// Output: 512 bfloat16 elements (1024 bytes)
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define EMBEDDING_DIM 512  // Whisper hidden dimension

// Vectorized LayerNorm for streaming 512 elements
extern "C" {

void layernorm_streaming_bf16(bfloat16* __restrict input,
                              bfloat16* __restrict output) {
  // Constants
  const bfloat16 eps = 1e-5;
  const int32_t vec_size = EMBEDDING_DIM;

  // Use AIE2 vector units (32-wide for bf16)
  constexpr int vec_factor = 32;
  constexpr int num_vectors = EMBEDDING_DIM / vec_factor;  // 512/32 = 16

  // Pass 1: Compute mean
  aie::vector<bfloat16, vec_factor> sum_vec = aie::zeros<bfloat16, vec_factor>();

  for (int i = 0; i < num_vectors; i++) {
    aie::vector<bfloat16, vec_factor> v = aie::load_v<vec_factor>(input + i * vec_factor);
    sum_vec = aie::add(sum_vec, v);
  }

  // Horizontal sum
  float sum = 0.0f;
  for (int i = 0; i < vec_factor; i++) {
    sum += (float)sum_vec[i];
  }
  float mean = sum / vec_size;
  bfloat16 mean_bf = (bfloat16)mean;

  // Pass 2: Compute variance
  aie::vector<bfloat16, vec_factor> var_sum_vec = aie::zeros<bfloat16, vec_factor>();
  aie::vector<bfloat16, vec_factor> mean_broadcast = aie::broadcast<bfloat16, vec_factor>(mean_bf);

  for (int i = 0; i < num_vectors; i++) {
    aie::vector<bfloat16, vec_factor> v = aie::load_v<vec_factor>(input + i * vec_factor);
    aie::vector<bfloat16, vec_factor> diff = aie::sub(v, mean_broadcast);
    aie::vector<bfloat16, vec_factor> sq = aie::mul(diff, diff);
    var_sum_vec = aie::add(var_sum_vec, sq);
  }

  // Horizontal sum for variance
  float var_sum = 0.0f;
  for (int i = 0; i < vec_factor; i++) {
    var_sum += (float)var_sum_vec[i];
  }
  float variance = var_sum / vec_size;
  float inv_std = 1.0f / sqrtf(variance + (float)eps);
  bfloat16 inv_std_bf = (bfloat16)inv_std;

  // Pass 3: Normalize
  aie::vector<bfloat16, vec_factor> inv_std_broadcast = aie::broadcast<bfloat16, vec_factor>(inv_std_bf);

  for (int i = 0; i < num_vectors; i++) {
    aie::vector<bfloat16, vec_factor> v = aie::load_v<vec_factor>(input + i * vec_factor);
    aie::vector<bfloat16, vec_factor> diff = aie::sub(v, mean_broadcast);
    aie::vector<bfloat16, vec_factor> normalized = aie::mul(diff, inv_std_broadcast);
    aie::store_v(output + i * vec_factor, normalized);
  }
}

}  // extern "C"
