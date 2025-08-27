#!/bin/bash

# Unicorn Amanuensis - Intel iGPU Optimized Server
# Defaults to Whisper Large v3 on Intel iGPU

echo "========================================="
echo "🦄 Unicorn Amanuensis - iGPU Server"
echo "========================================="

# Check if port 9000 is available
if netstat -tuln 2>/dev/null | grep -q ":9000 "; then
    echo "⚠️  Port 9000 is currently in use!"
    echo "To free port 9000, run: sudo kill $(lsof -t -i:9000)"
    echo ""
    echo "Starting on alternate port 8999..."
    PORT=8999
else
    PORT=9000
fi

# Set environment variables
export PATH="$HOME/.local/bin:$PATH"
export WHISPER_MODEL=large-v3
export WHISPER_DEVICE=igpu
export COMPUTE_TYPE=int8
export API_PORT=$PORT

# Navigate to whisperx directory
cd /home/ucadmin/Unicorn-Amanuensis/whisperx

echo ""
echo "Configuration:"
echo "  Model: Whisper Large v3"
echo "  Device: Intel iGPU (with OpenVINO)"
echo "  Port: $PORT"
echo "  Compute Type: INT8"
echo ""
echo "Starting server..."
echo "========================================="

# Start the iGPU optimized server
python3 -c "
import os
import sys
sys.path.insert(0, '.')
os.environ['API_PORT'] = '$PORT'
os.environ['WHISPER_MODEL'] = 'large-v3'
os.environ['WHISPER_DEVICE'] = 'igpu'
from server_igpu_optimized import app
import uvicorn

print(f'✅ Server starting on http://0.0.0.0:$PORT/')
print(f'📊 API docs: http://0.0.0.0:$PORT/docs')
print(f'🎨 Web interface: http://0.0.0.0:$PORT/')
uvicorn.run(app, host='0.0.0.0', port=$PORT)
"