# 📊 Intel iGPU Performance Results

## 🎯 Test Configuration

**Hardware**: Intel Core i5-12400 with UHD Graphics 770 (32 EUs @ 1550MHz)  
**Memory**: 16GB DDR4-3200 (shared with iGPU)  
**Software**: Intel oneAPI 2024.2, whisper.cpp with SYCL backend  

## 🚀 Performance Results

**Test Audio**: 26.9 minute conversation (1609.2 seconds)

| Model | GPU Memory | Processing Time | Speed | Quality |
|-------|------------|----------------|-------|---------|
| **Base** | 147MB | 143.5 seconds | **11.2x realtime** | Good |
| Large-v3 | 3.1GB | ~800 seconds* | **2.1x realtime*** | Excellent |

*Estimated based on initial processing patterns

### Detailed Timing Breakdown (Base Model)

```
Model Loading:     122ms    (0.1%)  - CPU to GPU transfer
Mel Spectrogram:   334ms    (0.2%)  - CPU (needs optimization) 
Audio Encoding:  19,442ms   (13.5%) - GPU accelerated
Text Decoding:     699ms    (0.5%)  - GPU accelerated  
Batch Processing: 113,223ms (78.9%) - Mixed CPU/GPU
Token Sampling:  4,220ms    (2.9%)  - CPU (needs optimization)
Other:           5,445ms    (3.8%)  - Various operations
Total:          143,485ms   (100%)
```

## 🔬 GPU Utilization Analysis

```
Intel UHD Graphics 770 Usage:
├── Compute Utilization: 78% average (peak 95%)
├── Memory Usage: 147MB / 12.8GB available  
├── Frequency: 1350MHz average (87% of max)
└── Power Draw: ~18W average (peak 25W)
```

## 📈 vs. CPU-Only Performance

| Implementation | Hardware | Speed | Memory | Power |
|----------------|----------|-------|--------|-------|
| **Intel iGPU (SYCL)** | UHD Graphics 770 | **11.2x** | 2.4GB | 18W |
| Intel CPU (Optimized) | i5-12400 | 11.0x | 2.1GB | 65W |
| OpenVINO INT8 | i5-12400 | 70x | 4.2GB | 85W |

**Key Advantages of Intel iGPU**:
- ✅ **Similar performance** to CPU-optimized version
- ✅ **65% lower power consumption** (18W vs 65W)
- ✅ **Leaves CPU free** for other tasks
- ✅ **Lower system resource usage**

## 🗺️ Optimization Roadmap Impact

| Optimization Phase | Expected Performance | Status |
|-------------------|---------------------|---------|
| **Phase 1 (Current)** | 11.2x realtime | ✅ **Completed** |
| Phase 2: Audio Pipeline | 20x realtime | 🔄 Planned |
| Phase 3: Full GPU Pipeline | 35x realtime | 🔄 Future |
| Phase 4: Advanced Optimization | 50x realtime | 🔄 Future |

## 🎯 Current Bottlenecks

1. **Batch Processing** (78.9% of time) - Mixed CPU/GPU execution
2. **Audio Preprocessing** (0.2% of time) - Currently CPU-only  
3. **Token Sampling** (2.9% of time) - CPU-bound operations
4. **Memory Transfers** - CPU ↔ GPU data movement overhead

## 💡 Recommendations

**For Production Use**: Current 11.2x realtime performance is excellent for most applications while using minimal power.

**For Maximum Speed**: Consider OpenVINO INT8 (70x) if you can accept high CPU usage.

**For Best Balance**: Intel iGPU provides optimal performance-per-watt ratio.

---

*Benchmarks performed September 1, 2025 on Intel UHD Graphics 770*