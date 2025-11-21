//===- layernorm_512_simple.cc -----------------------------------*- C++ -*-===//
//
// Simple LayerNorm for 512 elements (Whisper embedding dimension)
// Scalar implementation for initial testing
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <cmath>

#define EMBEDDING_DIM 512

extern "C" {

void layernorm_512_simple(bfloat16* __restrict input,
                          bfloat16* __restrict output) {
  const bfloat16 eps = 1e-5;

  // Pass 1: Compute mean
  float sum = 0.0f;
  for (int i = 0; i < EMBEDDING_DIM; i++) {
    sum += (float)input[i];
  }
  float mean = sum / EMBEDDING_DIM;

  // Pass 2: Compute variance
  float var_sum = 0.0f;
  for (int i = 0; i < EMBEDDING_DIM; i++) {
    float diff = (float)input[i] - mean;
    var_sum += diff * diff;
  }
  float variance = var_sum / EMBEDDING_DIM;
  float inv_std = 1.0f / std::sqrt(variance + (float)eps);

  // Pass 3: Normalize
  for (int i = 0; i < EMBEDDING_DIM; i++) {
    float normalized = ((float)input[i] - mean) * inv_std;
    output[i] = (bfloat16)normalized;
  }
}

}  // extern "C"
