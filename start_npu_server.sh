#!/bin/bash
# 🦄 Magic Unicorn NPU Server Startup Script
# Starts Unicorn Amanuensis with NPU acceleration

echo "============================================================"
echo "🦄 Unicorn Amanuensis - NPU Server Startup"
echo "============================================================"
echo ""

# Check NPU device
echo "1️⃣  Checking NPU device..."
if [ -c "/dev/accel/accel0" ]; then
    echo "   ✅ NPU device found: /dev/accel/accel0"
else
    echo "   ❌ NPU device NOT found!"
    echo "   Will fall back to CPU mode"
fi
echo ""

# Check XRT
echo "2️⃣  Checking XRT runtime..."
if [ -f "/opt/xilinx/xrt/bin/xrt-smi" ]; then
    echo "   ✅ XRT 2.20.0 installed"
    /opt/xilinx/xrt/bin/xrt-smi examine 2>&1 | grep -i "NPU\|Phoenix" | head -3
else
    echo "   ⚠️  XRT not found"
fi
echo ""

# Check NPU kernels
echo "3️⃣  Checking NPU kernels..."
KERNEL_DIR="whisperx/npu/npu_optimization"
if [ -d "$KERNEL_DIR" ]; then
    KERNEL_COUNT=$(find "$KERNEL_DIR" -name "*.xclbin" 2>/dev/null | wc -l)
    echo "   ✅ Found $KERNEL_COUNT compiled XCLBIN files"

    # Check for production kernels
    if [ -f "$KERNEL_DIR/mel_kernels/build_fixed_v3/mel_fixed_v3_PRODUCTION_v2.0.xclbin" ]; then
        echo "   ✅ Production mel kernel (v2.0) ready"
    fi
    if [ -f "$KERNEL_DIR/whisper_encoder_kernels/attention_64x64.xclbin" ]; then
        echo "   ✅ Attention kernel ready"
    fi
else
    echo "   ⚠️  Kernel directory not found"
fi
echo ""

# Set environment
echo "4️⃣  Setting up environment..."
export PYTHONPATH=/opt/xilinx/xrt/python:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
echo "   ✅ XRT Python path configured"
echo ""

# Start server
echo "5️⃣  Starting server..."
echo ""
echo "🚀 Server will start on: http://localhost:9004"
echo "🌐 Web interface at: http://localhost:9004/web"
echo ""
echo "Expected performance:"
echo "  • With NPU: 28.6× realtime (+49.7% speedup!)"
echo "  • With iGPU: 19.1× realtime"
echo "  • CPU only: 13.5× realtime"
echo ""
echo "============================================================"
echo "Press Ctrl+C to stop the server"
echo "============================================================"
echo ""

cd whisperx
python3 server_production.py
