# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Test executables
add_executable(test_quantization test_quantization.cpp)
target_link_libraries(test_quantization whisper_encoder_cpp)
add_test(NAME QuantizationTest COMMAND test_quantization)

add_executable(test_encoder_layer test_encoder_layer.cpp)
target_link_libraries(test_encoder_layer whisper_encoder_cpp)
add_test(NAME EncoderLayerTest COMMAND test_encoder_layer)

add_executable(test_bfp16_converter test_bfp16_converter.cpp)
target_link_libraries(test_bfp16_converter whisper_encoder_cpp)
add_test(NAME BFP16ConverterTest COMMAND test_bfp16_converter)

# Phase 2 Tests - BFP16 Quantization
add_executable(test_bfp16_quantization test_bfp16_quantization.cpp)
target_link_libraries(test_bfp16_quantization
    whisper_encoder_cpp
    GTest::gtest
    GTest::gtest_main
)
add_test(NAME BFP16QuantizationTest COMMAND test_bfp16_quantization)

# Phase 2 Tests - Encoder Layer BFP16 Integration
add_executable(test_encoder_layer_bfp16 test_encoder_layer_bfp16.cpp)
target_link_libraries(test_encoder_layer_bfp16
    whisper_encoder_cpp
    GTest::gtest
    GTest::gtest_main
)
add_test(NAME EncoderLayerBFP16Test COMMAND test_encoder_layer_bfp16)

# XRT NPU Integration Test
add_executable(test_xrt_npu_integration test_xrt_npu_integration.cpp)
target_link_libraries(test_xrt_npu_integration
    whisper_xdna2_cpp
    ${Python3_LIBRARIES}
)
target_include_directories(test_xrt_npu_integration PRIVATE
    ${Python3_INCLUDE_DIRS}
)
# Note: Not added to CTest since it requires hardware and kernel files
