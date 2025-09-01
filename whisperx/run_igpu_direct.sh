#!/bin/bash
# Kill any existing servers
pkill -f server_igpu

# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Set environment for iGPU-only execution
export SYCL_DEVICE_FILTER=gpu
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK=0  # Use first GPU (iGPU)

# Disable CPU fallback completely
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Force OpenVINO to use GPU only
export OV_CACHE_DIR=/tmp/ov_cache
export OPENVINO_DEVICE=GPU

echo "🦄 Starting TRUE iGPU-only server..."
echo "📍 All operations on Intel UHD Graphics 770"
echo "🚫 CPU fallback disabled"

# Run our custom SYCL server (not OpenVINO)
cd /home/ucadmin/Unicorn-Amanuensis/whisperx
python3 server_igpu_pipeline.py