cmake_minimum_required(VERSION 3.20)
project(whisper_xdna2_cpp VERSION 1.0.0 LANGUAGES CXX)

# C++17 required for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Build options
option(BUILD_ENCODER "Build encoder components (requires Eigen3)" ON)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_NATIVE_XRT "Build native XRT library (NO Python dependency)" ON)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Build encoder: ${BUILD_ENCODER}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build native XRT: ${BUILD_NATIVE_XRT}")

# Find Eigen3 (for encoder implementation) - only if requested
set(HAVE_EIGEN3 FALSE)
if(BUILD_ENCODER)
    # Set CMAKE_MODULE_PATH to help find Eigen3
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/share/eigen3/cmake)
    find_package(Eigen3 3.3 QUIET NO_MODULE)
    if(EIGEN3_FOUND OR Eigen3_FOUND)
        message(STATUS "Eigen3 found: ${EIGEN3_VERSION}")
        message(STATUS "Eigen3 include: ${EIGEN3_INCLUDE_DIR}")
        set(HAVE_EIGEN3 TRUE)
    else()
        message(WARNING "Eigen3 not found - encoder components will not be built")
        set(HAVE_EIGEN3 FALSE)
    endif()
endif()

# Find Python3 (required for Python C API in runtime)
find_package(Python3 REQUIRED COMPONENTS Development)
message(STATUS "Python3 found: ${Python3_VERSION}")
message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")

# Find XRT (optional - for NPU acceleration)
# XRT headers are typically in /opt/xilinx/xrt/include
# XRT libraries are typically in /opt/xilinx/xrt/lib
set(XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT installation path")
set(XRT_INCLUDE_DIR "${XRT_ROOT}/include")
set(XRT_LIB_DIR "${XRT_ROOT}/lib")

set(HAVE_XRT FALSE)
if(EXISTS "${XRT_INCLUDE_DIR}/xrt/xrt_device.h")
    message(STATUS "XRT found: ${XRT_ROOT}")
    message(STATUS "XRT include: ${XRT_INCLUDE_DIR}")
    message(STATUS "XRT lib: ${XRT_LIB_DIR}")
    set(HAVE_XRT TRUE)
else()
    message(WARNING "XRT not found - NPU acceleration disabled")
    message(WARNING "Install XRT from: https://github.com/Xilinx/XRT")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
)

if(HAVE_EIGEN3)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

if(HAVE_XRT)
    include_directories(${XRT_INCLUDE_DIR})
endif()

# Encoder source files (only if Eigen3 available)
if(HAVE_EIGEN3)
    set(ENCODER_SOURCES
        src/quantization.cpp
        src/attention.cpp
        src/ffn.cpp
        src/encoder_layer.cpp
        src/encoder_c_api.cpp
        src/bfp16_converter.cpp
        src/bfp16_quantization.cpp
    )

    set(ENCODER_HEADERS
        include/quantization.hpp
        include/attention.hpp
        include/ffn.hpp
        include/encoder_layer.hpp
        include/encoder_c_api.h
        include/bfp16_converter.hpp
        include/bfp16_quantization.hpp
    )
endif()

# Runtime source files
set(RUNTIME_SOURCES
    src/whisper_xdna2_runtime.cpp
    src/buffer_manager.cpp
    src/kernel_loader.cpp
)

set(RUNTIME_HEADERS
    include/whisper_xdna2_runtime.hpp
    include/buffer_manager.hpp
    include/kernel_loader.hpp
)

# Native XRT source files (NO Python dependency!)
set(NATIVE_XRT_SOURCES
    src/xrt_native.cpp
    src/xrt_native_c_api.cpp
)

set(NATIVE_XRT_HEADERS
    include/xrt_native.hpp
    include/xrt_native_c_api.h
)

# Create encoder library (if Eigen3 available)
if(HAVE_EIGEN3)
    add_library(whisper_encoder_cpp SHARED ${ENCODER_SOURCES})
    target_include_directories(whisper_encoder_cpp PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
    )
    set_target_properties(whisper_encoder_cpp PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
    )
endif()

# Create runtime library
add_library(whisper_xdna2_cpp SHARED ${RUNTIME_SOURCES})

target_include_directories(whisper_xdna2_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(whisper_xdna2_cpp
    ${Python3_LIBRARIES}
    pthread
    stdc++fs
)

# Link XRT if available (note: we use Python pyxrt, not C++ XRT directly)
# But we add the library path for completeness
if(HAVE_XRT)
    target_link_directories(whisper_xdna2_cpp PUBLIC ${XRT_LIB_DIR})
endif()

# Link encoder if available
if(HAVE_EIGEN3)
    target_link_libraries(whisper_xdna2_cpp whisper_encoder_cpp)
endif()

set_target_properties(whisper_xdna2_cpp PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${RUNTIME_HEADERS}"
)

# Create native XRT library (NO Python dependency!)
if(BUILD_NATIVE_XRT AND HAVE_XRT)
    add_library(xrt_native SHARED ${NATIVE_XRT_SOURCES})

    target_include_directories(xrt_native PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${XRT_INCLUDE_DIR}
    )

    # Find XRT libraries
    find_library(XRT_CORE_LIBRARY NAMES xrt_core PATHS ${XRT_LIB_DIR} REQUIRED)
    find_library(XRT_COREUTIL_LIBRARY NAMES xrt_coreutil PATHS ${XRT_LIB_DIR} REQUIRED)

    target_link_libraries(xrt_native
        ${XRT_CORE_LIBRARY}
        ${XRT_COREUTIL_LIBRARY}
        pthread
        stdc++fs
    )

    set_target_properties(xrt_native PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER "${NATIVE_XRT_HEADERS}"
    )

    message(STATUS "Native XRT library: ENABLED")
    message(STATUS "  XRT Core: ${XRT_CORE_LIBRARY}")
    message(STATUS "  XRT Coreutil: ${XRT_COREUTIL_LIBRARY}")
elseif(BUILD_NATIVE_XRT AND NOT HAVE_XRT)
    message(WARNING "Native XRT requested but XRT not found - skipping")
endif()

# Tests (only if Eigen3 available and BUILD_TESTS enabled)
if(HAVE_EIGEN3 AND BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install targets
if(HAVE_EIGEN3)
    install(TARGETS whisper_encoder_cpp
        LIBRARY DESTINATION lib
    )
    install(FILES ${ENCODER_HEADERS}
        DESTINATION include/whisper_xdna2
    )
endif()

install(TARGETS whisper_xdna2_cpp
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/whisper_xdna2
)

if(BUILD_NATIVE_XRT AND HAVE_XRT)
    install(TARGETS xrt_native
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/whisper_xdna2
    )
endif()

# Print summary
message(STATUS "")
message(STATUS "===========================================")
message(STATUS "Whisper XDNA2 C++ Configuration")
message(STATUS "===========================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(HAVE_EIGEN3)
    message(STATUS "Encoder: ENABLED (Eigen3 ${EIGEN3_VERSION})")
else()
    message(STATUS "Encoder: DISABLED (Eigen3 not found)")
endif()
if(HAVE_XRT)
    message(STATUS "NPU/XRT: ENABLED (${XRT_ROOT})")
    if(BUILD_NATIVE_XRT)
        message(STATUS "Native XRT: ENABLED (NO Python dependency)")
    endif()
else()
    message(STATUS "NPU/XRT: DISABLED (XRT not found)")
endif()
message(STATUS "===========================================")
message(STATUS "")
