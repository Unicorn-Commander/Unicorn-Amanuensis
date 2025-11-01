/**
 * NPU Callback Interface for C++ Encoder
 *
 * Allows C++ to call back to Python for NPU matmuls
 */

#ifndef NPU_CALLBACK_H
#define NPU_CALLBACK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * NPU matmul callback function type (BFP16 format)
 *
 * This function will be called from C++ when a matmul needs to run on the NPU.
 * The Python side should implement this to dispatch to the NPU runtime.
 *
 * BFP16 Format:
 * - Uses 9 bytes per 8 values on average (1.125× compression)
 * - Buffer size calculation: ((N + 7) / 8) * 9 bytes per row
 * - All pointers are uint8_t* (byte arrays, not typed integers)
 *
 * @param user_data Opaque pointer passed to callback (e.g., Python runtime object)
 * @param A_bfp16 Input matrix A (M×K_bfp16), BFP16 format, shuffled for NPU
 * @param B_bfp16 Weight matrix B (N×K_bfp16), BFP16 format, shuffled for NPU
 * @param C_bfp16 Output matrix C (M×N_bfp16), BFP16 format, shuffled (to be filled)
 * @param M Number of rows in A
 * @param K Number of columns in A (original FP32 dimension)
 * @param N Number of columns in B (original FP32 dimension)
 * @return 0 on success, -1 on failure
 */
typedef int (*NPUMatmulCallback)(
    void* user_data,
    const uint8_t* A_bfp16,
    const uint8_t* B_bfp16,
    uint8_t* C_bfp16,
    size_t M,
    size_t K,
    size_t N
);

/**
 * Set NPU callback for an encoder layer
 *
 * @param handle Encoder layer handle
 * @param callback NPU matmul callback function
 * @param user_data User data to pass to callback (e.g., Python runtime object)
 * @return 0 on success, -1 on failure
 */
int encoder_layer_set_npu_callback(
    void* handle,
    NPUMatmulCallback callback,
    void* user_data
);

#ifdef __cplusplus
}
#endif

#endif // NPU_CALLBACK_H
