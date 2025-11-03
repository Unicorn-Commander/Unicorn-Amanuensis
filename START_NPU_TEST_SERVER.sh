#!/bin/bash
# Start NPU Test Server on port 9005

cd /home/ucadmin/UC-1/Unicorn-Amanuensis

echo "🦄 Starting NPU Test Server..."
echo ""
echo "This will show you the NPU detection working!"
echo ""
echo "Open your browser to: http://localhost:9005/web"
echo ""

# Modify port to 9005
sed -i 's/port=9004/port=9005/g' test_npu_server.py

python3 test_npu_server.py
