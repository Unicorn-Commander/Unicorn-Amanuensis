//===- matmul_64x64_bf16.cc --------------------------------------*- C++ -*-===//
//
// Matrix Multiplication 64×64 for tiled operations
// C = A × B where A is 64×64, B is 64×64, C is 64×64
//
// Input A: 64×64 bfloat16 (8192 bytes)
// Input B: 64×64 bfloat16 (8192 bytes)
// Output C: 64×64 bfloat16 (8192 bytes)
//
// Total memory: 24KB (fits comfortably in 32KB tile SRAM)
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 64

extern "C" {

void matmul_64x64_bf16(bfloat16* __restrict A,
                       bfloat16* __restrict B,
                       bfloat16* __restrict C) {
  // Use AIE2 vector units for acceleration
  constexpr int vec_factor = 16;  // Process 16 bf16 elements at a time

  // Initialize output to zero
  for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) {
    C[i] = (bfloat16)0.0f;
  }

  // Matrix multiplication: C[i][j] = sum over k of A[i][k] * B[k][j]
  for (int i = 0; i < TILE_SIZE; i++) {
    for (int j = 0; j < TILE_SIZE; j += vec_factor) {
      // Accumulate for this row/column block
      aie::vector<float, vec_factor> acc = aie::zeros<float, vec_factor>();

      for (int k = 0; k < TILE_SIZE; k++) {
        // Broadcast A[i][k] to all lanes
        float a_val = (float)A[i * TILE_SIZE + k];
        aie::vector<float, vec_factor> a_broadcast = aie::broadcast<float, vec_factor>(a_val);

        // Load B[k][j:j+vec_factor]
        aie::vector<bfloat16, vec_factor> b_vec = aie::load_v<vec_factor>(&B[k * TILE_SIZE + j]);

        // Convert B to float for accumulation
        aie::vector<float, vec_factor> b_float;
        for (int v = 0; v < vec_factor; v++) {
          b_float[v] = (float)b_vec[v];
        }

        // Multiply and accumulate
        acc = aie::mac(acc, a_broadcast, b_float);
      }

      // Store result, converting back to bfloat16
      aie::vector<bfloat16, vec_factor> result;
      for (int v = 0; v < vec_factor; v++) {
        result[v] = (bfloat16)acc[v];
      }
      aie::store_v(&C[i * TILE_SIZE + j], result);
    }
  }
}

}  // extern "C"
