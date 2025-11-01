/**
 * Unit tests for BFP16 quantization functions
 *
 * Phase 2 Test Suite: BFP16Quantizer validation
 *
 * Tests:
 * 1. TestFindBlockExponent - Validate shared exponent calculation
 * 2. TestQuantizeDequantize - Test round-trip accuracy
 * 3. TestConvertToBFP16 - Large matrix conversion
 * 4. TestConvertFromBFP16 - BFP16 to FP32 conversion
 * 5. TestShuffleUnshuffle - NPU layout operations
 * 6. TestPrepareReadNPU - Full pipeline test
 *
 * Success criteria:
 * - Round-trip error < 1%
 * - All conversions complete without crashes
 * - Shuffle/unshuffle is exact reversal
 */

#include <gtest/gtest.h>
#include "bfp16_quantization.hpp"
#include <Eigen/Dense>
#include <random>
#include <cmath>

using namespace whisper_xdna2;

// Test utilities
namespace {

float compute_relative_error(const Eigen::MatrixXf& original, const Eigen::MatrixXf& reconstructed) {
    Eigen::MatrixXf diff = original - reconstructed;
    float mean_abs_error = diff.array().abs().mean();
    float original_mean_abs = original.array().abs().mean();
    return (original_mean_abs > 0.0f) ? (mean_abs_error / original_mean_abs) : 0.0f;
}

float compute_cosine_similarity(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) {
    float dot = (a.array() * b.array()).sum();
    float norm_a = std::sqrt((a.array() * a.array()).sum());
    float norm_b = std::sqrt((b.array() * b.array()).sum());
    return (norm_a > 0.0f && norm_b > 0.0f) ? (dot / (norm_a * norm_b)) : 1.0f;
}

Eigen::MatrixXf generate_random_matrix(int rows, int cols, float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(min_val, max_val);

    Eigen::MatrixXf mat(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat(i, j) = dist(gen);
        }
    }
    return mat;
}

} // anonymous namespace

/**
 * Test 1: TestFindBlockExponent
 *
 * Validates the shared exponent calculation for 8-value blocks.
 * Tests with known values and edge cases.
 */
TEST(BFP16QuantizationTest, FindBlockExponent) {
    // Test with simple values [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    // Note: BFP16 requires rows to be multiples of 8
    Eigen::MatrixXf test_matrix(8, 8);
    for (int i = 0; i < 8; i++) {
        test_matrix.row(i) << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f;
    }

    // Convert to BFP16
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> bfp16_data;
    BFP16Quantizer::convert_to_bfp16(test_matrix, bfp16_data);

    // Expected: 9 bytes per row (8 mantissas + 1 exponent)
    EXPECT_EQ(bfp16_data.cols(), 9);
    EXPECT_EQ(bfp16_data.rows(), 8);

    // Check exponent is reasonable (should be around 130 for values 1-8)
    uint8_t block_exp = bfp16_data(0, 8);  // Last byte of first row is the exponent
    EXPECT_GE(block_exp, 125);
    EXPECT_LE(block_exp, 135);

    // Test with all zeros
    Eigen::MatrixXf zeros = Eigen::MatrixXf::Zero(8, 8);
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> bfp16_zeros;
    BFP16Quantizer::convert_to_bfp16(zeros, bfp16_zeros);

    // Exponent for all zeros should be 0
    uint8_t zero_exp = bfp16_zeros(0, 8);
    EXPECT_EQ(zero_exp, 0);

    // Test with very small values
    Eigen::MatrixXf small_vals(8, 8);
    small_vals.setConstant(1e-6f);
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> bfp16_small;
    BFP16Quantizer::convert_to_bfp16(small_vals, bfp16_small);

    // Should not crash
    EXPECT_EQ(bfp16_small.cols(), 9);

    // Test with very large values
    Eigen::MatrixXf large_vals(8, 8);
    large_vals.setConstant(1000.0f);
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> bfp16_large;
    BFP16Quantizer::convert_to_bfp16(large_vals, bfp16_large);

    // Should not crash
    EXPECT_EQ(bfp16_large.cols(), 9);
}

/**
 * Test 2: TestQuantizeDequantize
 *
 * Tests quantization and dequantization accuracy.
 * Validates round-trip error is within acceptable bounds.
 */
TEST(BFP16QuantizationTest, QuantizeDequantize) {
    // Test single value 4.0 (8x8 matrix for BFP16 requirements)
    Eigen::MatrixXf single_val(8, 8);
    single_val.setConstant(4.0f);

    // Quantize
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> bfp16_data;
    BFP16Quantizer::convert_to_bfp16(single_val, bfp16_data);

    // Dequantize
    Eigen::MatrixXf reconstructed;
    BFP16Quantizer::convert_from_bfp16(bfp16_data, reconstructed, 8, 8);

    // Verify error < 1%
    float rel_error = compute_relative_error(single_val, reconstructed);
    EXPECT_LT(rel_error, 0.01f);  // < 1% error

    // Test 1024 random values (128 rows x 8 cols = 1024, multiple of 8)
    Eigen::MatrixXf random_vals = generate_random_matrix(128, 8, -1.0f, 1.0f);  // 1024 values

    // Round-trip
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> bfp16_random;
    BFP16Quantizer::convert_to_bfp16(random_vals, bfp16_random);

    Eigen::MatrixXf recon_random;
    BFP16Quantizer::convert_from_bfp16(bfp16_random, recon_random, 128, 8);

    // Verify error < 1%
    float random_error = compute_relative_error(random_vals, recon_random);
    EXPECT_LT(random_error, 0.01f);  // < 1% error

    // Verify shapes match
    EXPECT_EQ(recon_random.rows(), random_vals.rows());
    EXPECT_EQ(recon_random.cols(), random_vals.cols());
}

/**
 * Test 3: TestConvertToBFP16
 *
 * Tests conversion of large FP32 matrix to BFP16.
 * Validates output size and no crashes.
 */
TEST(BFP16QuantizationTest, ConvertToBFP16) {
    // Create 512x512 random FP32 matrix
    Eigen::MatrixXf input = generate_random_matrix(512, 512, -0.1f, 0.1f);

    // Convert to BFP16
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> bfp16_output;
    BFP16Quantizer::convert_to_bfp16(input, bfp16_output);

    // Verify output size: 512 x 576 (1.125x storage)
    EXPECT_EQ(bfp16_output.rows(), 512);

    // Calculate expected columns
    // 512 FP32 values -> 64 blocks of 8 -> 64 * 9 bytes = 576 bytes
    size_t expected_cols = BFP16Quantizer::calculate_bfp16_cols(512);
    EXPECT_EQ(bfp16_output.cols(), expected_cols);
    EXPECT_EQ(bfp16_output.cols(), 576);

    // Verify storage ratio is ~1.125x
    float storage_ratio = static_cast<float>(bfp16_output.size()) / (input.size() * sizeof(float));
    EXPECT_NEAR(storage_ratio, 0.28125f, 0.01f);  // 576/(512*4) = 0.28125

    // Check no crashes with different sizes
    Eigen::MatrixXf mat_128x128 = generate_random_matrix(128, 128, -1.0f, 1.0f);
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> bfp16_128;
    BFP16Quantizer::convert_to_bfp16(mat_128x128, bfp16_128);
    EXPECT_EQ(bfp16_128.rows(), 128);
    EXPECT_EQ(bfp16_128.cols(), 144);  // 128 / 8 * 9 = 144
}

/**
 * Test 4: TestConvertFromBFP16
 *
 * Tests BFP16 to FP32 conversion.
 * Validates round-trip error.
 */
TEST(BFP16QuantizationTest, ConvertFromBFP16) {
    // Create 512x512 FP32 matrix
    Eigen::MatrixXf original = generate_random_matrix(512, 512, -0.1f, 0.1f);

    // Convert to BFP16
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> bfp16_data;
    BFP16Quantizer::convert_to_bfp16(original, bfp16_data);

    // Convert back to FP32
    Eigen::MatrixXf reconstructed;
    BFP16Quantizer::convert_from_bfp16(bfp16_data, reconstructed, 512, 512);

    // Verify output size: 512 x 512
    EXPECT_EQ(reconstructed.rows(), 512);
    EXPECT_EQ(reconstructed.cols(), 512);

    // Measure round-trip error (expect < 1%)
    float rel_error = compute_relative_error(original, reconstructed);
    EXPECT_LT(rel_error, 0.01f);  // < 1% error

    // Verify cosine similarity > 99%
    float cos_sim = compute_cosine_similarity(original, reconstructed);
    EXPECT_GT(cos_sim, 0.99f);  // > 99% similarity
}

/**
 * Test 5: TestShuffleUnshuffle
 *
 * Tests shuffle and unshuffle operations for NPU layout.
 * Validates that unshuffle(shuffle(x)) == x.
 */
TEST(BFP16QuantizationTest, ShuffleUnshuffle) {
    // Create 512x576 BFP16 matrix (already in BFP16 format)
    Eigen::MatrixXf original_fp32 = generate_random_matrix(512, 512, -0.1f, 0.1f);
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> bfp16_original;
    BFP16Quantizer::convert_to_bfp16(original_fp32, bfp16_original);

    size_t rows = bfp16_original.rows();
    size_t cols_bytes = bfp16_original.cols();

    // Shuffle
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> shuffled;
    BFP16Quantizer::shuffle_bfp16(bfp16_original, shuffled, rows, cols_bytes);

    // Unshuffle
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> unshuffled;
    BFP16Quantizer::unshuffle_bfp16(shuffled, unshuffled, rows, cols_bytes);

    // Verify: unshuffle(shuffle(x)) == x
    EXPECT_EQ(unshuffled.rows(), bfp16_original.rows());
    EXPECT_EQ(unshuffled.cols(), bfp16_original.cols());

    // Check exact byte-for-byte match
    bool exact_match = (bfp16_original.array() == unshuffled.array()).all();
    EXPECT_TRUE(exact_match);

    if (!exact_match) {
        // Count differences
        int diff_count = (bfp16_original.array() != unshuffled.array()).count();
        FAIL() << "Shuffle/Unshuffle mismatch: " << diff_count << " / " << bfp16_original.size() << " bytes differ";
    }
}

/**
 * Test 6: TestPrepareReadNPU
 *
 * Tests the full pipeline: prepare_for_npu() and read_from_npu()
 * Validates round-trip error < 1%.
 */
TEST(BFP16QuantizationTest, PrepareReadNPU) {
    // Create 512x512 FP32 input
    Eigen::MatrixXf input = generate_random_matrix(512, 512, -0.1f, 0.1f);

    // Call prepare_for_npu() (FP32 -> BFP16 + shuffle)
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> npu_input;
    BFP16Quantizer::prepare_for_npu(input, npu_input);

    // Verify NPU input is shuffled BFP16
    EXPECT_EQ(npu_input.rows(), 512);
    EXPECT_EQ(npu_input.cols(), 576);

    // Call read_from_npu() (unshuffle + BFP16 -> FP32)
    Eigen::MatrixXf output;
    BFP16Quantizer::read_from_npu(npu_input, output, 512, 512);

    // Verify round-trip error < 1%
    float rel_error = compute_relative_error(input, output);
    EXPECT_LT(rel_error, 0.01f);  // < 1% error

    // Verify output dimensions
    EXPECT_EQ(output.rows(), 512);
    EXPECT_EQ(output.cols(), 512);

    // Verify cosine similarity > 99%
    float cos_sim = compute_cosine_similarity(input, output);
    EXPECT_GT(cos_sim, 0.99f);  // > 99% similarity

    // Test with different sizes
    Eigen::MatrixXf input_256 = generate_random_matrix(256, 256, -1.0f, 1.0f);
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> npu_256;
    BFP16Quantizer::prepare_for_npu(input_256, npu_256);

    Eigen::MatrixXf output_256;
    BFP16Quantizer::read_from_npu(npu_256, output_256, 256, 256);

    float error_256 = compute_relative_error(input_256, output_256);
    EXPECT_LT(error_256, 0.01f);
}

/**
 * Main test runner
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
