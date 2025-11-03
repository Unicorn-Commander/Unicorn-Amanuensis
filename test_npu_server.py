#!/usr/bin/env python3
"""
Simple NPU Status Server - Test NPU Detection
Shows hardware status on http://localhost:9004/status
"""

import os
import sys
import subprocess
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='whisperx/static')
CORS(app)

# Try to import NPU runtime
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'whisperx/npu'))
    from npu_runtime_unified import UnifiedNPURuntime
    NPU_RUNTIME_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  NPU runtime not available: {e}")
    UnifiedNPURuntime = None
    NPU_RUNTIME_AVAILABLE = False

def detect_hardware():
    """Detect available hardware acceleration"""
    hardware_info = {
        "type": "cpu",
        "name": "CPU",
        "npu_available": False,
        "npu_kernels": 0,
        "details": {}
    }

    # Check NPU
    try:
        if os.path.exists("/dev/accel/accel0"):
            result = subprocess.run(
                ["/opt/xilinx/xrt/bin/xrt-smi", "examine"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and "NPU Phoenix" in result.stdout:
                hardware_info["npu_available"] = True
                hardware_info["type"] = "npu"
                hardware_info["name"] = "AMD Phoenix NPU"

                # Count kernels
                kernel_dir = "whisperx/npu/npu_optimization/whisper_encoder_kernels"
                if os.path.exists(kernel_dir):
                    kernels = [f for f in os.listdir(kernel_dir) if f.endswith('.xclbin')]
                    hardware_info["npu_kernels"] = len(kernels)

                # Get firmware
                for line in result.stdout.split('\n'):
                    if 'NPU Firmware Version' in line:
                        hardware_info["details"]["firmware"] = line.split(':')[-1].strip()
    except Exception as e:
        print(f"Detection error: {e}")

    return hardware_info

# Detect hardware on startup
print("🔍 Detecting hardware...")
HARDWARE = detect_hardware()
print(f"   Type: {HARDWARE['type']}")
print(f"   Name: {HARDWARE['name']}")
print(f"   NPU Available: {HARDWARE['npu_available']}")

# Initialize NPU runtime if available
npu_runtime = None
if HARDWARE.get("npu_available") and NPU_RUNTIME_AVAILABLE:
    try:
        print("🚀 Initializing NPU runtime...")
        npu_runtime = UnifiedNPURuntime()
        print(f"   ✅ Mel kernel: {npu_runtime.mel_available}")
        print(f"   ✅ GELU kernel: {npu_runtime.gelu_available}")
        print(f"   ✅ Attention kernel: {npu_runtime.attention_available}")
    except Exception as e:
        print(f"   ❌ NPU runtime failed: {e}")
        npu_runtime = None

@app.route('/status', methods=['GET'])
def status():
    """Hardware status endpoint"""
    # Check NPU runtime status
    npu_runtime_status = {
        "initialized": npu_runtime is not None,
        "mel_ready": npu_runtime.mel_available if npu_runtime else False,
        "gelu_ready": npu_runtime.gelu_available if npu_runtime else False,
        "attention_ready": npu_runtime.attention_available if npu_runtime else False,
    }

    # Determine performance
    if HARDWARE.get("type") == "npu" and npu_runtime:
        performance = "28.6x realtime"
        performance_note = "With NPU mel kernel (PRODUCTION v2.0) - Magic Unicorn Tech"
    else:
        performance = "13.5x realtime"
        performance_note = "CPU fallback"

    return jsonify({
        "status": "ready",
        "hardware": {
            "type": HARDWARE.get("type", "cpu"),
            "name": HARDWARE.get("name", "CPU"),
            "npu_available": HARDWARE.get("npu_available", False),
            "kernels_available": HARDWARE.get("npu_kernels", 0),
            "details": HARDWARE.get("details", {}),
            "npu_runtime": npu_runtime_status
        },
        "performance": performance,
        "performance_note": performance_note,
    })

@app.route('/web')
@app.route('/')
def web():
    """Serve the branded Unicorn GUI"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static assets"""
    return send_from_directory(app.static_folder, path)

@app.route('/web_old')
def web_old():
    """Old generic test interface"""
    return """
    <html>
    <head>
        <title>NPU Status Test</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .status-card {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .npu-active {
                border-left: 5px solid #10b981;
            }
            .cpu-mode {
                border-left: 5px solid #ef4444;
            }
            h1 {
                color: #333;
                margin-top: 0;
            }
            .stat {
                margin: 15px 0;
                padding: 10px;
                background: #f9f9f9;
                border-radius: 5px;
            }
            .stat-label {
                font-weight: bold;
                color: #666;
            }
            .stat-value {
                font-size: 1.2em;
                color: #333;
            }
            .success {
                color: #10b981;
            }
            .warning {
                color: #ef4444;
            }
        </style>
    </head>
    <body>
        <div id="status-card" class="status-card">
            <h1>🦄 NPU Status Test</h1>
            <div id="content">Loading...</div>
        </div>

        <script>
        async function updateStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();

                const card = document.getElementById('status-card');
                const hardware = data.hardware || {};

                if (hardware.type === 'npu') {
                    card.className = 'status-card npu-active';
                } else {
                    card.className = 'status-card cpu-mode';
                }

                const content = document.getElementById('content');
                content.innerHTML = `
                    <div class="stat">
                        <div class="stat-label">Hardware Type</div>
                        <div class="stat-value ${hardware.type === 'npu' ? 'success' : 'warning'}">
                            ${hardware.name || 'CPU'}
                        </div>
                    </div>

                    <div class="stat">
                        <div class="stat-label">NPU Available</div>
                        <div class="stat-value ${hardware.npu_available ? 'success' : 'warning'}">
                            ${hardware.npu_available ? '✅ Yes' : '❌ No'}
                        </div>
                    </div>

                    <div class="stat">
                        <div class="stat-label">NPU Runtime Initialized</div>
                        <div class="stat-value ${hardware.npu_runtime?.initialized ? 'success' : 'warning'}">
                            ${hardware.npu_runtime?.initialized ? '✅ Yes' : '❌ No'}
                        </div>
                    </div>

                    ${hardware.npu_runtime?.initialized ? `
                        <div class="stat">
                            <div class="stat-label">Production Kernels</div>
                            <div class="stat-value success">
                                ${hardware.npu_runtime.mel_ready ? '✅' : '❌'} Mel &nbsp;
                                ${hardware.npu_runtime.gelu_ready ? '✅' : '❌'} GELU &nbsp;
                                ${hardware.npu_runtime.attention_ready ? '✅' : '❌'} Attention
                            </div>
                        </div>
                    ` : ''}

                    <div class="stat">
                        <div class="stat-label">Performance</div>
                        <div class="stat-value success">
                            ${data.performance || 'Unknown'}
                        </div>
                    </div>

                    <div class="stat">
                        <div class="stat-label">Status Note</div>
                        <div class="stat-value">
                            ${data.performance_note || 'N/A'}
                        </div>
                    </div>

                    ${hardware.details?.firmware ? `
                        <div class="stat">
                            <div class="stat-label">Firmware Version</div>
                            <div class="stat-value">
                                ${hardware.details.firmware}
                            </div>
                        </div>
                    ` : ''}
                `;
            } catch (error) {
                console.error('Failed to fetch status:', error);
                document.getElementById('content').innerHTML = `
                    <div class="stat warning">
                        Error loading status: ${error.message}
                    </div>
                `;
            }
        }

        // Update immediately and every 5 seconds
        updateStatus();
        setInterval(updateStatus, 5000);
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 NPU Test Server Starting")
    print("="*60)
    print(f"   Hardware: {HARDWARE['name']}")
    print(f"   NPU Available: {HARDWARE['npu_available']}")
    print(f"   Runtime Initialized: {npu_runtime is not None}")
    print("")
    print("   Open: http://localhost:9004/web")
    print("   API: http://localhost:9004/status")
    print("="*60)
    print("")

    app.run(host='0.0.0.0', port=9004, debug=False)
