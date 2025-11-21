//===- layernorm_512_nosqrt.cc -----------------------------------*- C++ -*-===//
//
// Simple LayerNorm for 512 elements (Whisper embedding dimension)
// Scalar implementation for initial testing - NO SQRT DEPENDENCY
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#define EMBEDDING_DIM 512

// Fast inverse square root approximation (no sqrtf needed)
inline float fast_inv_sqrt(float x) {
  // Newton-Raphson approximation for 1/sqrt(x)
  float xhalf = 0.5f * x;
  int i = *(int*)&x;
  i = 0x5f3759df - (i >> 1);  // Magic number from Quake III
  float y = *(float*)&i;
  y = y * (1.5f - xhalf * y * y);  // One iteration
  y = y * (1.5f - xhalf * y * y);  // Two iterations for better accuracy
  return y;
}

extern "C" {

void layernorm_512_nosqrt(bfloat16* __restrict input,
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
  float inv_std = fast_inv_sqrt(variance + (float)eps);

  // Pass 3: Normalize
  for (int i = 0; i < EMBEDDING_DIM; i++) {
    float normalized = ((float)input[i] - mean) * inv_std;
    output[i] = (bfloat16)normalized;
  }
}

}  // extern "C"
