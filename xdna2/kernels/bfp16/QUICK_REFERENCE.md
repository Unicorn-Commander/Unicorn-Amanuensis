# BFP16 Kernels - Quick Reference

## Location
```
/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/kernels/bfp16/
```

## Quick Commands

### Generate MLIR (Fast, ~5 seconds)
```bash
cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/kernels/bfp16
./build_bfp16_kernels.sh
```

### Generate Single Kernel
```bash
source ~/mlir-aie/ironenv/bin/activate
python3 generate_whisper_bfp16.py --dev npu2 -M 512 -K 512 -N 512 \
  --dtype_in bf16 --dtype_out bf16 --emulate-bf16-mmul-with-bfp16 true
```

### View Generated MLIR
```bash
ls -lh build/mlir/
cat build/mlir/matmul_512x512x512_bfp16.mlir
```

## Whisper Dimensions

| Kernel | M | K | N | Usage |
|--------|---|---|-----|-------|
| Attention | 512 | 512 | 512 | Q/K/V/out (4x per layer) |
| FFN fc1 | 512 | 512 | 2048 | Expansion (1x per layer) |
| FFN fc2 | 512 | 2048 | 512 | Reduction (1x per layer) |

## Key Files

- **README.md** - Usage guide
- **BFP16_FORMAT.md** - Format documentation (11 KB)
- **SETUP_COMPLETE.md** - Setup report
- **mm_bfp.cc** - BFP16 kernel (C++)
- **generate_whisper_bfp16.py** - MLIR generator
- **build_bfp16_kernels.sh** - Build script

## BFP16 Format

- **Block size**: 8x8 elements
- **Storage**: 72 bytes per block (64 mantissas + 8 exponents)
- **Compression**: 56% vs BF16
- **Performance**: 2x faster MAC on XDNA2

## Performance Target

- **400-500x realtime Whisper Base**
- **13,000 matmuls/sec** (512x512x512)
- **60-75 µs per 30ms frame**
- **2.3% NPU utilization** (97% headroom)

## Next Steps

1. Implement FP32→BFP16 conversion (Python)
2. Create shuffle function bindings
3. Test with real Whisper weights
4. Integrate with encoder skeleton
5. Benchmark on NPU hardware

## Documentation

- `/home/ccadmin/CC-1L/docs/WHISPER_ARCHITECTURE.md`
- `/home/ccadmin/CC-1L/docs/WEEK3_COMPLETE.md`
- `BFP16_FORMAT.md` (this directory)
- `SETUP_COMPLETE.md` (this directory)

---
**Created**: October 30, 2025
**Status**: ✅ Infrastructure ready
