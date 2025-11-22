//===- matmul_512_simple.cc ----------------------------------------*- C++ -*-===//
//
// Simple 512-element vector-matrix multiply for Whisper encoder
// Implements Y = X @ W where:
//   X: input vector (512 BF16)
//   W: weight matrix (512x512 BF16)
//   Y: output vector (512 BF16)
//
// This avoids complex tiling - processes one output element at a time
// Memory: 512*2 + 512*512*2 + 512*2 = 525,312 bytes (exceeds 32KB tile limit)
// Solution: Process in chunks, stream weights
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#define DIM 512

extern "C" {

// Simple matmul without AIE API - processes 64 outputs at a time
void matmul_512_simple(bfloat16* __restrict input,    // 512 elements
                       bfloat16* __restrict weights,  // 512x512 elements
                       bfloat16* __restrict output) {  // 512 elements

  // Process output in chunks of 64 to fit in memory
  for (int out_chunk = 0; out_chunk < DIM; out_chunk += 64) {
    int chunk_size = (out_chunk + 64 <= DIM) ? 64 : (DIM - out_chunk);

    // Compute each output element in this chunk
    for (int i = 0; i < chunk_size; i++) {
      int out_idx = out_chunk + i;
      float sum = 0.0f;

      // Dot product: output[out_idx] = sum(input[k] * weights[out_idx][k])
      for (int k = 0; k < DIM; k++) {
        sum += (float)input[k] * (float)weights[out_idx * DIM + k];
      }

      output[out_idx] = (bfloat16)sum;
    }
  }
}

// Simpler version for 64x64 tiles (fits in tile memory)
void matmul_64x64_simple(bfloat16* __restrict A,  // 64x64
                         bfloat16* __restrict B,  // 64x64
                         bfloat16* __restrict C) { // 64x64

  #define TILE_DIM 64

  for (int i = 0; i < TILE_DIM; i++) {
    for (int j = 0; j < TILE_DIM; j++) {
      float sum = 0.0f;

      for (int k = 0; k < TILE_DIM; k++) {
        sum += (float)A[i * TILE_DIM + k] * (float)B[k * TILE_DIM + j];
      }

      C[i * TILE_DIM + j] = (bfloat16)sum;
    }
  }
}

}  // extern "C"
