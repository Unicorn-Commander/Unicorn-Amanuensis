//===- softmax_bf16_xdna1_batched.cc ----------------------------*- C++ -*-===//
//
// XDNA1 (Phoenix NPU) - Batched BF16 Softmax Kernel
// Processes 4 softmax operations per invocation to amortize overhead
//
// OPTIMIZED FOR SIZE: Single function with loop-based processing
// to fit within AIE tile program memory constraints (~16-32 KB)
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

extern "C" {

// Batched softmax - 4 frames per invocation
// Memory: 4 * 1024 * 2 bytes = 8 KB input + 8 KB output = 16 KB
// Expected: 4 * 0.175ms compute + 1.4ms overhead = 2.1ms total
// Per-frame: 0.52ms (3x speedup from 1.565ms)
void softmax_batched_bf16_4(bfloat16 *restrict input,
                            bfloat16 *restrict output) {
  event0();
  
  const int32_t frame_size = 1024;
  const int32_t batch_size = 4;
  
  // Process each frame in sequence
  for (int32_t b = 0; b < batch_size; b++) {
    bfloat16 *in_ptr = input + b * frame_size;
    bfloat16 *out_ptr = output + b * frame_size;
    
    // Find maximum for numerical stability
    float max_val = (float)in_ptr[0];
    for (int32_t i = 1; i < frame_size; i++) {
      float val = (float)in_ptr[i];
      if (val > max_val) {
        max_val = val;
      }
    }

    // First pass: compute exp(x) and sum
    float sum = 0.0f;
    for (int32_t i = 0; i < frame_size; i++) {
      float x = (float)in_ptr[i] - max_val;

      // Exp approximation using log2(e) = 1.442695
      int32_t ix = (int32_t)(x * 1.442695f);
      float fx = x * 1.442695f - ix;

      // Compute 2^ix using bit manipulation
      ix = (ix + 127) << 23;
      float pow2_ix;
      memcpy(&pow2_ix, &ix, sizeof(float));

      // Approximation for 2^fx: 1 + 0.693147 * fx + 0.240160 * fx^2
      float pow2_fx = 1.0f + 0.693147f * fx + 0.240160f * fx * fx;

      float result = pow2_ix * pow2_fx;
      out_ptr[i] = (bfloat16)result;
      sum += result;
    }

    // Second pass: normalize
    float inv_sum = 1.0f / (sum + 1e-7f);
    for (int32_t i = 0; i < frame_size; i++) {
      float val = (float)out_ptr[i] * inv_sum;
      out_ptr[i] = (bfloat16)val;
    }
  }
  
  event1();
}

} // extern "C"
