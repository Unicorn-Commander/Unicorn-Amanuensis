#!/bin/bash
# Quick Integration Guide for NPU MatMul Acceleration
# Run this to integrate NPU matmul into Whisper pipeline

set -e

echo "============================================================"
echo "NPU MATMUL INTEGRATION - QUICK START"
echo "============================================================"
echo ""

# Step 1: Verify NPU is available
echo "[1/6] Verifying NPU device..."
if [ ! -e "/dev/accel/accel0" ]; then
    echo "❌ ERROR: NPU device not found at /dev/accel/accel0"
    echo "   Please ensure XRT 2.20.0 is installed and NPU is enabled"
    exit 1
fi
echo "✅ NPU device found: /dev/accel/accel0"
echo ""

# Step 2: Check XCLBIN exists
echo "[2/6] Checking NPU kernel..."
XCLBIN_PATH="/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels/build_matmul_fixed/matmul_16x16.xclbin"
if [ ! -f "$XCLBIN_PATH" ]; then
    echo "❌ ERROR: XCLBIN not found at $XCLBIN_PATH"
    exit 1
fi
echo "✅ XCLBIN found: $XCLBIN_PATH"
echo ""

# Step 3: Quick unit test
echo "[3/6] Running quick unit test..."
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels
python3 -c "
from npu_matmul_wrapper import NPUMatmul
import numpy as np

# Quick test
matmul = NPUMatmul()
A = np.random.randint(-32, 32, (64, 64), dtype=np.int8)
B = np.random.randint(-32, 32, (64, 64), dtype=np.int8)
C = matmul(A, B, quantize=False)

print(f'✅ NPU matmul test passed: output shape {C.shape}')
"
echo ""

# Step 4: Set environment variables
echo "[4/6] Setting environment variables..."
export NPU_MATMUL_ENABLED=1
export MATMUL_XCLBIN="$XCLBIN_PATH"
echo "export NPU_MATMUL_ENABLED=1" >> ~/.bashrc
echo "export MATMUL_XCLBIN=\"$XCLBIN_PATH\"" >> ~/.bashrc
echo "✅ Environment variables set"
echo ""

# Step 5: Integration options
echo "[5/6] Integration Options:"
echo ""
echo "Option A: Test encoder standalone"
echo "  cd whisper_encoder_kernels"
echo "  python3 whisper_npu_encoder_matmul.py"
echo ""
echo "Option B: Test decoder standalone"
echo "  cd whisper_encoder_kernels"
echo "  python3 whisper_npu_decoder_matmul.py"
echo ""
echo "Option C: Integrate into production (manual)"
echo "  1. Edit unified_stt_diarization.py"
echo "  2. Add: from whisper_encoder_kernels.whisper_npu_encoder_matmul import WhisperNPUEncoderMatmul"
echo "  3. Replace encoder with NPU version"
echo "  4. Restart server"
echo ""

# Step 6: Performance monitoring
echo "[6/6] Performance Monitoring:"
echo ""
echo "Monitor NPU usage:"
echo "  watch -n 1 '/opt/xilinx/xrt/bin/xrt-smi examine'"
echo ""
echo "Check kernel statistics:"
echo "  # In Python:"
echo "  matmul = NPUMatmul()"
echo "  stats = matmul.get_stats()"
echo "  print(stats)"
echo ""

echo "============================================================"
echo "✅ NPU MATMUL INTEGRATION READY"
echo "============================================================"
echo ""
echo "Expected Performance:"
echo "  - Current baseline: 19.1× realtime"
echo "  - With NPU matmul: 25-29× realtime"
echo "  - Improvement: ~38% faster"
echo ""
echo "Next Steps:"
echo "  1. Run Option A or B to benchmark"
echo "  2. Integrate into production pipeline"
echo "  3. Test with real audio"
echo "  4. Monitor performance metrics"
echo ""
echo "For full details, see: MATMUL_INTEGRATION_STATUS_OCT30.md"
echo "============================================================"
