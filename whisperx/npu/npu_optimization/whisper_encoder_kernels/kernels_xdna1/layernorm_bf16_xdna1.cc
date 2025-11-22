//===- layernorm_bf16_xdna1.cc ------------------------------------*- C++ -*-===//
//
// XDNA1 (Phoenix NPU) - BF16 Layer Normalization Kernel
// Scalar implementation for Whisper encoder/decoder
//
// LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

// Fast inverse square root approximation (Quake III algorithm)
static inline float fast_rsqrt(float x) {
  union {
    float f;
    int32_t i;
  } conv = {x};
  conv.i = 0x5f3759df - (conv.i >> 1);
  float y = conv.f;
  // One Newton-Raphson iteration
  y = y * (1.5f - (x * 0.5f * y * y));
  return y;
}

extern "C" {

// Layer normalization for 1024 elements (standard Whisper hidden dimension)
void layernorm_bf16(bfloat16 *restrict input, 
                    bfloat16 *restrict output,
                    bfloat16 *restrict gamma,
                    bfloat16 *restrict beta) {
  event0();
  
  const int32_t vector_size = 1024;
  const float eps = 1e-5f;
  
  // Pass 1: Compute mean
  float sum = 0.0f;
  for (int32_t i = 0; i < vector_size; i++) {
    sum += (float)input[i];
  }
  float mean = sum / (float)vector_size;
  
  // Pass 2: Compute variance
  float var_sum = 0.0f;
  for (int32_t i = 0; i < vector_size; i++) {
    float diff = (float)input[i] - mean;
    var_sum += diff * diff;
  }
  float variance = var_sum / (float)vector_size;
  
  // Compute inverse standard deviation
  // Using fast inverse square root approximation
  float rsqrt_var = fast_rsqrt(variance + eps);
  
  // Pass 3: Normalize and apply affine transform
  for (int32_t i = 0; i < vector_size; i++) {
    float x = (float)input[i];
    float normalized = (x - mean) * rsqrt_var;
    float g = (float)gamma[i];
    float b = (float)beta[i];
    float result = normalized * g + b;
    output[i] = (bfloat16)result;
  }
  
  event1();
}

// Simplified LayerNorm without learnable parameters (just normalize)
void layernorm_simple_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  event0();
  
  const int32_t vector_size = 1024;
  const float eps = 1e-5f;
  
  // Pass 1: Compute mean
  float sum = 0.0f;
  for (int32_t i = 0; i < vector_size; i++) {
    sum += (float)input[i];
  }
  float mean = sum / (float)vector_size;
  
  // Pass 2: Compute variance
  float var_sum = 0.0f;
  for (int32_t i = 0; i < vector_size; i++) {
    float diff = (float)input[i] - mean;
    var_sum += diff * diff;
  }
  float variance = var_sum / (float)vector_size;

  // Compute inverse standard deviation
  float rsqrt_var = fast_rsqrt(variance + eps);

  // Pass 3: Normalize
  for (int32_t i = 0; i < vector_size; i++) {
    float x = (float)input[i];
    float normalized = (x - mean) * rsqrt_var;
    output[i] = (bfloat16)normalized;
  }

  event1();
}

} // extern "C"
