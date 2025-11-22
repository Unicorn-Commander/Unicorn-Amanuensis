//===- matmul_bf16_vectorized_xdna1.cc ---------------------------*- C++ -*-===//
//
// XDNA1 (Phoenix NPU) - Vectorized BF16 Matrix Multiplication Kernel
// Tile-based matrix multiply for Whisper attention and FFN layers
//
// C = A * B
// Where: A is M*K, B is K*N, C is M*N
//
// Uses AIE2 vector intrinsics for massive speedup over scalar implementation.
// Target: 30-200x speedup over scalar (49ms for 64x64)
//
// Vectorization strategy:
// - Use BF16 vector operations directly (AIE2 native support)
// - Use accfloat accumulators for numerical stability
// - Broadcast A values and multiply with B row vectors
// - Tiled approach for cache efficiency
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
constexpr int TILE_SIZE = 64;

// Vector width for BF16 operations
// AIE2 can process 16 BF16 elements efficiently in parallel
constexpr int VEC_SIZE = 16;

// Vectorized BF16 matrix multiplication with accumulator
// Uses broadcast multiply-accumulate pattern
// A: M*K matrix (row-major)
// B: K*N matrix (row-major)
// C: M*N matrix (row-major, output)
void matmul_vectorized_bf16(bfloat16 *restrict A, bfloat16 *restrict B,
                            bfloat16 *restrict C, int32_t M, int32_t N, int32_t K) {
  event0();

  // Process each output row
  for (int32_t i = 0; i < M; i++) {
    // Process output columns in groups of VEC_SIZE
    for (int32_t j = 0; j < N; j += VEC_SIZE) {
      // Initialize accumulator for VEC_SIZE output elements
      aie::accum<accfloat, VEC_SIZE> acc;
      acc = aie::zeros<accfloat, VEC_SIZE>();

      // Accumulate over K dimension
      for (int32_t k = 0; k < K; k++) {
        // Get A[i, k] value
        bfloat16 a_val = A[i * K + k];

        // Broadcast to vector
        aie::vector<bfloat16, VEC_SIZE> a_vec = aie::broadcast<bfloat16, VEC_SIZE>(a_val);

        // Load VEC_SIZE elements from B row k, columns j to j+VEC_SIZE-1
        // B[k, j:j+VEC_SIZE]
        aie::vector<bfloat16, VEC_SIZE> b_vec = aie::load_v<VEC_SIZE>(&B[k * N + j]);

        // Multiply and accumulate: acc += a_vec * b_vec
        acc = aie::mac(acc, a_vec, b_vec);
      }

      // Convert accumulated result back to BF16 and store
      aie::vector<bfloat16, VEC_SIZE> result = acc.to_vector<bfloat16>();
      aie::store_v(&C[i * N + j], result);
    }
  }

  event1();
}

// Highly optimized vectorized matmul with loop unrolling
// Processes 4 output rows simultaneously for better instruction-level parallelism
void matmul_vectorized_unrolled_bf16(bfloat16 *restrict A, bfloat16 *restrict B,
                                      bfloat16 *restrict C, int32_t M, int32_t N, int32_t K) {
  event0();

  constexpr int UNROLL = 4;  // Process 4 rows at a time

  // Process rows in groups of UNROLL
  for (int32_t i = 0; i < M; i += UNROLL) {
    // Process output columns in groups of VEC_SIZE
    for (int32_t j = 0; j < N; j += VEC_SIZE) {
      // Initialize accumulators for UNROLL rows * VEC_SIZE columns
      aie::accum<accfloat, VEC_SIZE> acc0, acc1, acc2, acc3;
      acc0 = aie::zeros<accfloat, VEC_SIZE>();
      acc1 = aie::zeros<accfloat, VEC_SIZE>();
      acc2 = aie::zeros<accfloat, VEC_SIZE>();
      acc3 = aie::zeros<accfloat, VEC_SIZE>();

      // Accumulate over K dimension
      for (int32_t k = 0; k < K; k++) {
        // Load elements from A for all 4 rows and broadcast
        aie::vector<bfloat16, VEC_SIZE> a_vec0 = aie::broadcast<bfloat16, VEC_SIZE>(A[(i + 0) * K + k]);
        aie::vector<bfloat16, VEC_SIZE> a_vec1 = aie::broadcast<bfloat16, VEC_SIZE>(A[(i + 1) * K + k]);
        aie::vector<bfloat16, VEC_SIZE> a_vec2 = aie::broadcast<bfloat16, VEC_SIZE>(A[(i + 2) * K + k]);
        aie::vector<bfloat16, VEC_SIZE> a_vec3 = aie::broadcast<bfloat16, VEC_SIZE>(A[(i + 3) * K + k]);

        // Load B row once (shared across all A rows)
        aie::vector<bfloat16, VEC_SIZE> b_vec = aie::load_v<VEC_SIZE>(&B[k * N + j]);

        // Multiply and accumulate for all 4 rows
        acc0 = aie::mac(acc0, a_vec0, b_vec);
        acc1 = aie::mac(acc1, a_vec1, b_vec);
        acc2 = aie::mac(acc2, a_vec2, b_vec);
        acc3 = aie::mac(acc3, a_vec3, b_vec);
      }

      // Convert and store results for all 4 rows
      aie::store_v(&C[(i + 0) * N + j], acc0.to_vector<bfloat16>());
      aie::store_v(&C[(i + 1) * N + j], acc1.to_vector<bfloat16>());
      aie::store_v(&C[(i + 2) * N + j], acc2.to_vector<bfloat16>());
      aie::store_v(&C[(i + 3) * N + j], acc3.to_vector<bfloat16>());
    }
  }

  event1();
}

// Tile-based vectorized matmul with cache-friendly memory access
// Uses smaller tiles to fit in AIE local memory (~32KB)
void matmul_vectorized_tiled_bf16(bfloat16 *restrict A, bfloat16 *restrict B,
                                   bfloat16 *restrict C, int32_t M, int32_t N, int32_t K) {
  event0();

  constexpr int TILE_M = 8;   // Tile size in M dimension
  constexpr int TILE_N = 16;  // Tile size in N dimension (VEC_SIZE)
  constexpr int TILE_K = 64;  // Process full K dimension at once

  // Initialize output to zero
  for (int32_t i = 0; i < M * N; i += VEC_SIZE) {
    aie::vector<bfloat16, VEC_SIZE> zeros = aie::zeros<bfloat16, VEC_SIZE>();
    aie::store_v(&C[i], zeros);
  }

  // Process tiles
  for (int32_t i0 = 0; i0 < M; i0 += TILE_M) {
    for (int32_t j0 = 0; j0 < N; j0 += TILE_N) {
      // Process each row in the M tile
      for (int32_t i = i0; i < i0 + TILE_M && i < M; i++) {
        // Initialize accumulator for this row
        aie::accum<accfloat, VEC_SIZE> acc;
        acc = aie::zeros<accfloat, VEC_SIZE>();

        // Accumulate over entire K dimension
        for (int32_t k = 0; k < K; k++) {
          aie::vector<bfloat16, VEC_SIZE> a_vec = aie::broadcast<bfloat16, VEC_SIZE>(A[i * K + k]);
          aie::vector<bfloat16, VEC_SIZE> b_vec = aie::load_v<VEC_SIZE>(&B[k * N + j0]);

          acc = aie::mac(acc, a_vec, b_vec);
        }

        // Store result
        aie::store_v(&C[i * N + j0], acc.to_vector<bfloat16>());
      }
    }
  }

  event1();
}

extern "C" {

// Fixed-size 64x64 vectorized matrix multiplication for MLIR wrapper
// A: 64*64 BF16 matrix (8192 bytes)
// B: 64*64 BF16 matrix (8192 bytes)
// C: 64*64 BF16 matrix (8192 bytes, output)
void matmul_bf16_64x64(bfloat16 *restrict A, bfloat16 *restrict B,
                       bfloat16 *restrict C) {
  const int32_t size = 64;
  // Use unrolled version for best performance on 64x64
  matmul_vectorized_unrolled_bf16(A, B, C, size, size, size);
}

// Fixed-size 32x32 vectorized matrix multiplication
// A: 32*32 BF16 matrix (2048 bytes)
// B: 32*32 BF16 matrix (2048 bytes)
// C: 32*32 BF16 matrix (2048 bytes, output)
void matmul_bf16_32x32(bfloat16 *restrict A, bfloat16 *restrict B,
                       bfloat16 *restrict C) {
  const int32_t size = 32;
  matmul_vectorized_unrolled_bf16(A, B, C, size, size, size);
}

// General vectorized matrix multiplication
// M, N, K passed as parameters
void matmul_bf16(bfloat16 *restrict A, bfloat16 *restrict B,
                 bfloat16 *restrict C, int32_t M, int32_t N, int32_t K) {
  if (M <= 64 && N <= 64 && K <= 64) {
    // Small matrices: use unrolled vectorized version
    matmul_vectorized_unrolled_bf16(A, B, C, M, N, K);
  } else {
    // Larger matrices: use tiled approach
    matmul_vectorized_tiled_bf16(A, B, C, M, N, K);
  }
}

} // extern "C"
