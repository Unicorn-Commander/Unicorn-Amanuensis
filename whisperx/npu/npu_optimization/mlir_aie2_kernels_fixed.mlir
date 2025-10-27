// MLIR-AIE2 Kernels for WhisperX NPU Acceleration (FIXED)
// ==============================================
// Custom kernels for AMD NPU Phoenix (Ryzen AI)
// Target: AIE2 architecture with 1024-bit vector units

module @whisperx_npu_kernels {
  // Constants for AIE2 architecture - Fixed to use memref types
  memref.global "private" constant @VECTOR_WIDTH : memref<1xi32> = dense<32>
  memref.global "private" constant @AIE_TILES : memref<1xi32> = dense<20>
  memref.global "private" constant @DMA_CHANNELS : memref<1xi32> = dense<2>

  // Whisper Attention Kernel - Optimized for AIE2
  // This is the most compute-intensive part of Whisper
  aie.device(npu1_4col) {
    // Define AIE tile array layout (4x5 array)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    // Memory layout for efficient DMA
    %mem_a = aie.buffer(%tile_0_0) {sym_name = "query_buffer"} : memref<1024xi8>
    %mem_b = aie.buffer(%tile_0_1) {sym_name = "key_buffer"} : memref<1024xi8>
    %mem_c = aie.buffer(%tile_0_2) {sym_name = "value_buffer"} : memref<1024xi8>
    %mem_out = aie.buffer(%tile_0_3) {sym_name = "output_buffer"} : memref<1024xi8>

    // Attention Score Computation Kernel
    aie.core(%tile_0_0) {
      // Q @ K^T computation with INT8
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index

      scf.for %i = %c0 to %c1024 step %c32 {
        // Load 32 INT8 values using vector.load
        %q_vec = vector.load %mem_a[%i] : memref<1024xi8>, vector<32xi8>
        %k_vec = vector.load %mem_b[%i] : memref<1024xi8>, vector<32xi8>

        // INT8 multiply-accumulate using standard operations
        %q_ext = arith.extsi %q_vec : vector<32xi8> to vector<32xi32>
        %k_ext = arith.extsi %k_vec : vector<32xi8> to vector<32xi32>
        %acc = arith.muli %q_ext, %k_ext : vector<32xi32>

        // Quantization-aware scaling
        %scale = arith.constant 127 : i32
        %scale_vec = vector.splat %scale : vector<32xi32>
        %scaled = arith.muli %acc, %scale_vec : vector<32xi32>

        // Truncate back to INT8 for storage
        %result = arith.trunci %scaled : vector<32xi32> to vector<32xi8>
        vector.store %result, %mem_out[%i] : memref<1024xi8>, vector<32xi8>
      }
      aie.end
    }

    // Softmax Kernel - Simplified for INT8
    aie.core(%tile_0_1) {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index

      // Vectorized softmax approximation
      scf.for %i = %c0 to %c1024 step %c32 {
        %scores = vector.load %mem_out[%i] : memref<1024xi8>, vector<32xi8>

        // Simple normalization for INT8 range
        %min = arith.constant -128 : i8
        %max = arith.constant 127 : i8
        %min_vec = vector.splat %min : vector<32xi8>
        %max_vec = vector.splat %max : vector<32xi8>

        // Clamp values
        %clamped = arith.maxsi %scores, %min_vec : vector<32xi8>
        %normalized = arith.minsi %clamped, %max_vec : vector<32xi8>

        vector.store %normalized, %mem_out[%i] : memref<1024xi8>, vector<32xi8>
      }
      aie.end
    }

    // Matrix Multiply for Attention @ Values
    aie.core(%tile_0_2) {
      // Tiled matrix multiplication for V projection
      %M = arith.constant 64 : index
      %K = arith.constant 64 : index
      %N = arith.constant 64 : index

      // Triple nested loop with optimizations
      affine.for %m = 0 to 64 step 8 {
        affine.for %n = 0 to 64 step 8 {
          affine.for %k = 0 to 64 step 32 {
            // Load attention scores and values
            %offset_att = affine.apply affine_map<(d0, d1) -> (d0 * 64 + d1)>(%m, %k)
            %offset_val = affine.apply affine_map<(d0, d1) -> (d0 * 64 + d1)>(%k, %n)

            %att_vec = vector.load %mem_out[%offset_att] : memref<1024xi8>, vector<32xi8>
            %val_vec = vector.load %mem_c[%offset_val] : memref<1024xi8>, vector<32xi8>

            // Multiply and accumulate
            %att_ext = arith.extsi %att_vec : vector<32xi8> to vector<32xi32>
            %val_ext = arith.extsi %val_vec : vector<32xi8> to vector<32xi32>
            %prod = arith.muli %att_ext, %val_ext : vector<32xi32>

            // Quantize back to INT8
            %result = arith.trunci %prod : vector<32xi32> to vector<32xi8>

            // Store output
            %offset_out = affine.apply affine_map<(d0, d1) -> (d0 * 64 + d1)>(%m, %n)
            vector.store %result, %mem_out[%offset_out] : memref<1024xi8>, vector<32xi8>
          }
        }
      }
      aie.end
    }

    // DMA Configuration for Streaming
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_3, DMA : 0)
  }

  // Mel Spectrogram Feature Extraction Kernel
  func.func @mel_spectrogram_aie2(%audio: memref<16000xi16>, %mel_out: memref<80x3000xi8>) {
    // FFT window parameters
    %window_size = arith.constant 400 : index
    %hop_size = arith.constant 160 : index
    %n_mels = arith.constant 80 : index
    %c0 = arith.constant 0 : index
    %c3000 = arith.constant 3000 : index

    // Process audio in chunks suitable for AIE2
    affine.for %frame = 0 to 3000 {
      %offset = arith.muli %frame, %hop_size : index

      // Window and FFT preparation
      %windowed = memref.alloca() : memref<512xi16>

      // Apply Hanning window with vectors
      affine.for %i = 0 to 400 step 32 {
        %idx = arith.addi %offset, %i : index
        %audio_vec = vector.load %audio[%idx] : memref<16000xi16>, vector<32xi16>

        // Simple window approximation (can be improved with lookup table)
        %half = arith.constant 16384 : i16  // 0.5 in Q15
        %half_vec = vector.splat %half : vector<32xi16>
        %windowed_vec = arith.muli %audio_vec, %half_vec : vector<32xi16>

        vector.store %windowed_vec, %windowed[%i] : memref<512xi16>, vector<32xi16>
      }

      // FFT placeholder - would call optimized FFT function
      // For now, just copy to output (simplified)
      affine.for %mel = 0 to 80 {
        %dummy = arith.constant 0 : i8
        memref.store %dummy, %mel_out[%mel, %frame] : memref<80x3000xi8>
      }
    }
    return
  }

  // Convolution kernel for encoder layers
  func.func @conv1d_aie2(%input: memref<3000x512xi8>,
                         %weight: memref<3x512x512xi8>,
                         %output: memref<3000x512xi8>) {
    %M = arith.constant 3000 : index
    %C_in = arith.constant 512 : index
    %C_out = arith.constant 512 : index
    %K = arith.constant 3 : index
    %c1 = arith.constant 1 : index

    // Process multiple output channels in parallel
    affine.for %oc = 0 to 512 step 8 {
      affine.for %t = 1 to 2999 {
        affine.for %k = 0 to 3 {
          %t_in = arith.addi %t, %k : index
          %t_in_adj = arith.subi %t_in, %c1 : index

          affine.for %ic = 0 to 512 step 32 {
            // Load input vector
            %in_vec = vector.load %input[%t_in_adj, %ic] : memref<3000x512xi8>, vector<32xi8>

            // Load weight and multiply (simplified - should accumulate)
            affine.for %oc_off = 0 to 8 {
              %oc_idx = arith.addi %oc, %oc_off : index
              %w_vec = vector.load %weight[%k, %ic, %oc_idx] : memref<3x512x512xi8>, vector<32xi8>

              // Multiply
              %in_ext = arith.extsi %in_vec : vector<32xi8> to vector<32xi32>
              %w_ext = arith.extsi %w_vec : vector<32xi8> to vector<32xi32>
              %prod = arith.muli %in_ext, %w_ext : vector<32xi32>

              // Reduce to scalar (simplified)
              %c0_i32 = arith.constant 0 : i32
              %sum = vector.reduction <add>, %prod, %c0_i32 : vector<32xi32> into i32

              // Store result
              %sum_i8 = arith.trunci %sum : i32 to i8
              memref.store %sum_i8, %output[%t, %oc_idx] : memref<3000x512xi8>
            }
          }
        }
      }
    }
    return
  }

  // Positional encoding addition
  func.func @add_positional_encoding_aie2(%input: memref<3000x512xi8>,
                                          %pos_enc: memref<3000x512xi8>) {
    %M = arith.constant 3000 : index
    %D = arith.constant 512 : index

    // Vectorized addition with saturation
    affine.for %t = 0 to 3000 {
      affine.for %d = 0 to 512 step 32 {
        %in_vec = vector.load %input[%t, %d] : memref<3000x512xi8>, vector<32xi8>
        %pos_vec = vector.load %pos_enc[%t, %d] : memref<3000x512xi8>, vector<32xi8>

        // Saturating addition for INT8
        %in_ext = arith.extsi %in_vec : vector<32xi8> to vector<32xi16>
        %pos_ext = arith.extsi %pos_vec : vector<32xi8> to vector<32xi16>
        %sum = arith.addi %in_ext, %pos_ext : vector<32xi16>

        // Clamp to INT8 range
        %min = arith.constant -128 : i16
        %max = arith.constant 127 : i16
        %min_vec = vector.splat %min : vector<32xi16>
        %max_vec = vector.splat %max : vector<32xi16>
        %clamped = arith.maxsi %sum, %min_vec : vector<32xi16>
        %saturated = arith.minsi %clamped, %max_vec : vector<32xi16>

        %result = arith.trunci %saturated : vector<32xi16> to vector<32xi8>
        vector.store %result, %input[%t, %d] : memref<3000x512xi8>, vector<32xi8>
      }
    }
    return
  }

  // Layer normalization for INT8
  func.func @layer_norm_aie2(%input: memref<512xi8>, %gamma: memref<512xi8>,
                             %beta: memref<512xi8>, %output: memref<512xi8>) {
    %D = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32

    // Compute mean with reduction
    %sum = affine.for %d = 0 to 512 step 32 iter_args(%acc = %c0_i32) -> i32 {
      %vec = vector.load %input[%d] : memref<512xi8>, vector<32xi8>
      %vec_i32 = arith.extsi %vec : vector<32xi8> to vector<32xi32>
      %partial_sum = vector.reduction <add>, %vec_i32, %c0_i32 : vector<32xi32> into i32
      %new_acc = arith.addi %acc, %partial_sum : i32
      affine.yield %new_acc : i32
    }

    // Compute mean
    %mean = arith.divsi %sum, %c512_i32 : i32
    %mean_i8 = arith.trunci %mean : i32 to i8

    // Normalize (simplified - variance computation omitted for brevity)
    affine.for %d = 0 to 512 step 32 {
      %x_vec = vector.load %input[%d] : memref<512xi8>, vector<32xi8>
      %gamma_vec = vector.load %gamma[%d] : memref<512xi8>, vector<32xi8>
      %beta_vec = vector.load %beta[%d] : memref<512xi8>, vector<32xi8>

      %mean_vec = vector.splat %mean_i8 : vector<32xi8>
      %centered = arith.subi %x_vec, %mean_vec : vector<32xi8>

      // Apply gamma and beta
      %scaled = arith.muli %centered, %gamma_vec : vector<32xi8>

      %scaled_ext = arith.extsi %scaled : vector<32xi8> to vector<32xi16>
      %beta_ext = arith.extsi %beta_vec : vector<32xi8> to vector<32xi16>
      %shifted = arith.addi %scaled_ext, %beta_ext : vector<32xi16>

      %result = arith.trunci %shifted : vector<32xi16> to vector<32xi8>
      vector.store %result, %output[%d] : memref<512xi8>, vector<32xi8>
    }
    return
  }
}
