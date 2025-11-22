//===- matmul_bf16_xdna1.cc ------------------------------------*- C++ -*-===//
//
// XDNA1 (Phoenix NPU) - BF16 Matrix Multiplication Kernel
// Tile-based matrix multiply for Whisper attention and FFN layers
//
// C = A * B
// Where: A is M×K, B is K×N, C is M×N
//
// Uses BF16 inputs/outputs with FP32 accumulation for numerical stability.
// Supports Whisper dimensions: 512, 768, 1024, 1280, 1536
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

// Tile size for matrix multiplication
// 64x64 tiles fit well in AIE local memory
constexpr int TILE_SIZE = 64;

// Scalar BF16 matrix multiplication with FP32 accumulation
// A: M×K matrix (row-major)
// B: K×N matrix (row-major)
// C: M×N matrix (row-major, output)
void matmul_scalar_bf16(bfloat16 *restrict A, bfloat16 *restrict B,
                        bfloat16 *restrict C, int32_t M, int32_t N, int32_t K) {
  event0();

  // Initialize output to zero
  for (int32_t i = 0; i < M * N; i++) {
    C[i] = (bfloat16)0.0f;
  }

  // Perform matrix multiplication with FP32 accumulation
  // C[i,j] = sum_k(A[i,k] * B[k,j])
  for (int32_t i = 0; i < M; i++) {
    for (int32_t j = 0; j < N; j++) {
      float acc = 0.0f;
      for (int32_t k = 0; k < K; k++) {
        float a_val = (float)A[i * K + k];
        float b_val = (float)B[k * N + j];
        acc += a_val * b_val;
      }
      C[i * N + j] = (bfloat16)acc;
    }
  }

  event1();
}

// Tiled matrix multiplication for better cache efficiency
// Processes TILE_SIZE x TILE_SIZE blocks at a time
void matmul_tiled_bf16(bfloat16 *restrict A, bfloat16 *restrict B,
                       bfloat16 *restrict C, int32_t M, int32_t N, int32_t K) {
  event0();

  // Initialize output to zero
  for (int32_t i = 0; i < M * N; i++) {
    C[i] = (bfloat16)0.0f;
  }

  // Tile-based multiplication
  // Iterate over tiles of C
  for (int32_t i0 = 0; i0 < M; i0 += TILE_SIZE) {
    for (int32_t j0 = 0; j0 < N; j0 += TILE_SIZE) {
      // For each tile of C, accumulate contributions from K dimension
      for (int32_t k0 = 0; k0 < K; k0 += TILE_SIZE) {
        // Compute contribution from this A-tile × B-tile
        int32_t i_max = (i0 + TILE_SIZE < M) ? (i0 + TILE_SIZE) : M;
        int32_t j_max = (j0 + TILE_SIZE < N) ? (j0 + TILE_SIZE) : N;
        int32_t k_max = (k0 + TILE_SIZE < K) ? (k0 + TILE_SIZE) : K;

        for (int32_t i = i0; i < i_max; i++) {
          for (int32_t j = j0; j < j_max; j++) {
            float acc = (float)C[i * N + j];
            for (int32_t k = k0; k < k_max; k++) {
              float a_val = (float)A[i * K + k];
              float b_val = (float)B[k * N + j];
              acc += a_val * b_val;
            }
            C[i * N + j] = (bfloat16)acc;
          }
        }
      }
    }
  }

  event1();
}

extern "C" {

// Fixed-size 64x64 matrix multiplication for MLIR wrapper
// A: 64×64 BF16 matrix (8192 bytes)
// B: 64×64 BF16 matrix (8192 bytes)
// C: 64×64 BF16 matrix (8192 bytes, output)
void matmul_bf16_64x64(bfloat16 *restrict A, bfloat16 *restrict B,
                       bfloat16 *restrict C) {
  const int32_t size = 64;
  matmul_scalar_bf16(A, B, C, size, size, size);
}

// Fixed-size 32x32 matrix multiplication (smaller tile for testing)
// A: 32×32 BF16 matrix (2048 bytes)
// B: 32×32 BF16 matrix (2048 bytes)
// C: 32×32 BF16 matrix (2048 bytes, output)
void matmul_bf16_32x32(bfloat16 *restrict A, bfloat16 *restrict B,
                       bfloat16 *restrict C) {
  const int32_t size = 32;
  matmul_scalar_bf16(A, B, C, size, size, size);
}

// General matrix multiplication (for future dynamic sizes)
// M, N, K passed as parameters
void matmul_bf16(bfloat16 *restrict A, bfloat16 *restrict B,
                 bfloat16 *restrict C, int32_t M, int32_t N, int32_t K) {
  if (M <= 64 && N <= 64 && K <= 64) {
    // Small matrices: use scalar for simplicity
    matmul_scalar_bf16(A, B, C, M, N, K);
  } else {
    // Larger matrices: use tiled approach
    matmul_tiled_bf16(A, B, C, M, N, K);
  }
}

} // extern "C"
