//===- gelu_simple_xdna1.cc -------------------------------------*- C++ -*-===//
//
// XDNA1 (Phoenix NPU) - Simple BF16 GELU Kernel
// Scalar implementation without lookup tables
//
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Using fast tanh approximation instead of lookup tables for simplicity
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

// Fast tanh approximation using rational function
// tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2)
// Good for |x| < 4.5
static inline float fast_tanh(float x) {
  float x2 = x * x;
  return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

extern "C" {

void gelu_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  event0();
  
  const int32_t vector_size = 1024;
  const float sqrt_2_over_pi = 0.79788456f;  // sqrt(2/pi)
  const float beta = 0.044715f;
  
  for (int32_t i = 0; i < vector_size; i++) {
    float x = (float)input[i];
    
    // Compute x^3
    float x2 = x * x;
    float x3 = x * x2;
    
    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    float inner = sqrt_2_over_pi * (x + beta * x3);
    
    // tanh_out = tanh(inner)
    float tanh_out = fast_tanh(inner);
    
    // result = 0.5 * x * (1 + tanh_out)
    float result = 0.5f * x * (1.0f + tanh_out);
    
    output[i] = (bfloat16)result;
  }
  
  event1();
}

} // extern "C"
