# 🎙️ whisper.cpp Intel iGPU Acceleration

[![GitHub Stars](https://img.shields.io/github/stars/ggerganov/whisper.cpp?style=flat-square)](https://github.com/ggerganov/whisper.cpp)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](/LICENSE)
[![Intel iGPU](https://img.shields.io/badge/Intel%20iGPU-Optimized-00C853?style=flat-square)](https://www.intel.com/content/www/us/en/products/docs/graphics/integrated-graphics.html)
[![SYCL](https://img.shields.io/badge/SYCL-2020-FF6F00?style=flat-square)](https://www.khronos.org/sycl/)
[![Performance](https://img.shields.io/badge/Performance-11.2x%20Realtime-FF4444?style=flat-square)](#performance-benchmarks)

**High-performance speech-to-text inference with Intel integrated GPU acceleration**

This optimization project leverages Intel integrated GPUs through SYCL and Intel Math Kernel Library (MKL) for dramatically improved transcription performance on Intel hardware.

## 🚀 **Key Features**

- **🎯 Intel iGPU Acceleration**: Native SYCL backend with Intel MKL optimization
- **⚡ 11.2x Realtime Performance**: 26-minute audio transcribed in 2.4 minutes
- **🔧 All Model Compatibility**: Works with tiny, base, small, medium, large, and large-v3
- **💾 Efficient Memory Usage**: Direct GPU memory loading (3GB for large-v3 model)
- **🔋 Low Power Consumption**: Utilizes integrated graphics efficiently

## 📊 **Performance Results**

| Hardware | Model | Audio Length | Processing Time | Speed | 
|----------|-------|--------------|----------------|-------|
| Intel UHD Graphics 770 | Base | 26.9 minutes | 2.4 minutes | **11.2x realtime** |
| Intel UHD Graphics 770 | Large-v3 | 26.9 minutes | ~8 minutes* | **3.4x realtime*** |

## 🔧 **Quick Setup**

```bash
# 1. Install Intel oneAPI Toolkit
sudo apt install -y intel-oneapi-toolkit intel-oneapi-mkl-sycl-devel

# 2. Build with SYCL support  
source /opt/intel/oneapi/setvars.sh
mkdir build_sycl && cd build_sycl
cmake .. -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
         -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
         -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/mkl/latest/lib/cmake
make -j$(nproc)

# 3. Run with iGPU acceleration
export ONEAPI_DEVICE_SELECTOR=level_zero:0
./bin/whisper-cli -m models/ggml-base.bin -f audio.wav
```

## 🗺️ **Optimization Roadmap**

### ✅ **Phase 1: Foundation (COMPLETED)**
- [x] Intel SYCL backend integration
- [x] Intel MKL library optimization  
- [x] Matrix operations on iGPU
- [x] Performance benchmarking (11.2x realtime achieved)

### 🎯 **Future Phases (PLANNED)**
- [ ] **Phase 2**: Audio preprocessing on GPU (target: 20x realtime)
- [ ] **Phase 3**: Complete encoder/decoder pipeline (target: 35x realtime) 
- [ ] **Phase 4**: Advanced optimizations & INT8 support (target: 50x realtime)

## 📈 **Expected Performance Improvements**

| Phase | Current Speed | Target Speed | CPU Usage |
|-------|--------------|--------------|-----------|
| **Phase 1 (Current)** | 11.2x realtime | ✅ Achieved | Medium |
| **Phase 2 (Planned)** | 11.2x → 20x | +79% improvement | Low |
| **Phase 3 (Future)** | 20x → 35x | +75% improvement | Minimal |
| **Phase 4 (Advanced)** | 35x → 50x | +43% improvement | Zero |

## 🛠️ **Supported Hardware**

- **Intel UHD Graphics** (Gen 9+) - Tested on UHD 770
- **Intel Iris Xe Graphics** - Compatible 
- **Intel Arc Graphics** - Compatible
- **Intel Data Center GPU Flex** - Compatible

## 📚 **Documentation**

- [Detailed Setup Guide](INTEL_IGPU_SETUP.md) - Complete installation instructions
- [Performance Benchmarks](PERFORMANCE_BENCHMARKS.md) - Detailed performance analysis
- [Intel oneAPI Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/sycl.html)

## 🏆 **Acknowledgments**

- **[Georgi Gerganov](https://github.com/ggerganov)** - Original whisper.cpp implementation
- **[OpenAI](https://openai.com)** - Whisper model architecture
- **[Intel](https://intel.com)** - oneAPI toolkit and SYCL support

## 📄 **License**

This project follows the same MIT License as the original whisper.cpp project.

---

*Built with ❤️ for high-performance speech recognition on Intel hardware*