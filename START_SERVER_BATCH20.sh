#!/bin/bash
# Start WhisperX Server with Batch-20 NPU Processor
# Magic Unicorn Unconventional Technology & Stuff Inc.
# November 1, 2025

echo "=========================================="
echo "🦄 Starting Unicorn Amanuensis Server"
echo "   With Batch-20 NPU Processor"
echo "=========================================="
echo ""

# Change to whisperx directory
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx || {
    echo "❌ Error: Could not find whisperx directory"
    exit 1
}

# Check if NPU device exists
if [ -e "/dev/accel/accel0" ]; then
    echo "✅ NPU Device: /dev/accel/accel0 found"
else
    echo "⚠️  NPU Device: Not found (will use CPU fallback)"
fi

# Check if XCLBIN exists
XCLBIN_PATH="npu/npu_optimization/mel_kernels/build_batch20/mel_batch20.xclbin"
if [ -f "$XCLBIN_PATH" ]; then
    echo "✅ Batch-20 Kernel: Found ($(ls -lh $XCLBIN_PATH | awk '{print $5}'))"
else
    echo "⚠️  Batch-20 Kernel: Not found at $XCLBIN_PATH"
fi

echo ""
echo "=========================================="
echo "Starting server on port 9004..."
echo "=========================================="
echo ""
echo "Logs will be written to: /tmp/server_batch20.log"
echo ""
echo "To monitor logs:"
echo "  tail -f /tmp/server_batch20.log"
echo ""
echo "To check status:"
echo "  curl http://localhost:9004/status"
echo ""
echo "To stop server:"
echo "  pkill -f server_dynamic"
echo ""
echo "=========================================="
echo ""

# Start server in background
nohup python3 server_dynamic.py > /tmp/server_batch20.log 2>&1 &
SERVER_PID=$!

# Wait a moment for server to start
sleep 3

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "✅ Server started successfully!"
    echo "   PID: $SERVER_PID"
    echo "   Port: 9004"
    echo ""

    # Try to get status
    echo "Testing server endpoint..."
    if curl -s http://localhost:9004/status > /dev/null 2>&1; then
        echo "✅ Server is responding!"
        echo ""
        curl -s http://localhost:9004/status | python3 -m json.tool 2>/dev/null || echo "(Status endpoint available)"
    else
        echo "⚠️  Server starting... (may take a few seconds)"
        echo "   Check logs: tail -f /tmp/server_batch20.log"
    fi
else
    echo "❌ Server failed to start!"
    echo "   Check logs: cat /tmp/server_batch20.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "🦄 Server is ready!"
echo "=========================================="
echo ""
echo "Web interface: http://localhost:9004/web"
echo "API docs: http://localhost:9004/docs"
echo ""
