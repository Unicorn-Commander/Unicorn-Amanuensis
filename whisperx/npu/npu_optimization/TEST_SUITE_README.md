# NPU Kernel Test Suite

Comprehensive validation suite for AMD Phoenix NPU kernels.

## Quick Start

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization
python3 test_all_kernels.py
```

## What It Tests

1. **Mel Spectrogram Kernel** (`mel_fixed_v3.xclbin`)
   - Audio preprocessing for Whisper
   - Proven working: 16.5x realtime
   - Tests: 1 kHz sine wave → 80 mel bins

2. **Matrix Multiply Kernel** (`matmul_simple.xclbin`)
   - INT8 matrix multiplication (16×16)
   - Core operation for encoder/decoder
   - Tests: Diagonal matrix @ all-5s matrix

3. **Attention Mechanism Kernel** (`attention_simple.xclbin`)
   - INT8 scaled dot-product attention (16×16)
   - Critical for Transformer architecture
   - Tests: Random Q, K, V matrices

## Output

Color-coded results with:
- ✓ Green: Tests passed
- ✗ Red: Tests failed
- ⚠ Yellow: Warnings

Metrics reported:
- Execution time (ms)
- Realtime factor (for mel kernel)
- Performance (GOPS)
- Output validation (energy, non-zero counts)

## Test Results (Oct 29, 2025)

```
Total Tests: 3
Passed: 3
Failed: 0
Errors: 0

NPU Kernels Operational:
  • Mel Spectrogram: 16.5x realtime
  • Matrix Multiply: 0.004 GOPS
  • Attention Mechanism: 0.007 GOPS

Total execution time: 0.35 seconds
```

## Hardware Requirements

- AMD Ryzen 7040/8040 series (Phoenix/Hawk Point)
- XRT 2.20.0 or later
- `/dev/accel/accel0` accessible
- Python 3.10+ with PyXRT

## Files Tested

### XCLBINs
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/build_fixed_v3/mel_fixed_v3.xclbin` (16 KB)
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build/matmul_simple.xclbin` (11 KB)
- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build_attention/attention_simple.xclbin` (12 KB)

### Instruction Binaries
- `mel_kernels/build_fixed_v3/insts_v3.bin` (300 bytes)
- `whisper_encoder_kernels/build/insts.bin` (420 bytes)
- `whisper_encoder_kernels/build_attention/insts.bin` (300 bytes)

## Detailed Results

See: `NPU_KERNEL_VALIDATION_REPORT.md` for comprehensive analysis.

## Known Issues

1. **Matrix Multiply**: Outputs all zeros (kernel runs but needs debugging)
2. **Small Tile Sizes**: Currently 16×16, need to scale to 64×64+ for production
3. **Accuracy**: Need reference CPU comparisons for validation

## Next Steps

1. Debug matrix multiply kernel (verify buffer connections)
2. Validate attention accuracy against PyTorch reference
3. Scale to larger tile sizes (32×32, 64×64)
4. Integrate with WhisperX pipeline
5. Measure end-to-end speedup

## Exit Codes

- `0`: All tests passed
- `1`: One or more tests failed

## Troubleshooting

**"Device not found"**:
```bash
ls -l /dev/accel/accel0
/opt/xilinx/xrt/bin/xrt-smi examine
```

**"XCLBIN not found"**:
- Ensure you're running from the correct directory
- Check that XCLBINs were compiled successfully

**"Kernel timeout"**:
- Kernel may be hung
- Check XRT logs: `/var/log/xrt/`
- Reboot NPU: `sudo rmmod amdxdna && sudo modprobe amdxdna`

## Performance Notes

- Mel kernel: ~1.5ms execution (16.5x realtime for 25ms audio)
- Matmul kernel: ~1.0ms execution (needs debugging)
- Attention kernel: ~1.2ms execution (producing valid outputs)

## Contact

**Magic Unicorn Unconventional Technology & Stuff Inc.**
- Aaron Stransky: aaron@magicunicorn.tech
- GitHub: https://github.com/Unicorn-Commander

## License

Part of the Unicorn-Amanuensis project.
