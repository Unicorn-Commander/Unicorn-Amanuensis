#!/usr/bin/env python3
"""
Device Manager for Unicorn Amanuensis
Intelligently selects and configures the best available acceleration device
"""

import os
import subprocess
import logging
from typing import Optional, Dict, List, Tuple
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Supported device types"""
    INTEL_IGPU = "intel_igpu"
    AMD_NPU = "amd_npu"
    NVIDIA_GPU = "nvidia_gpu"
    CPU = "cpu"

class DeviceManager:
    """Manages device detection and selection for optimal transcription"""
    
    def __init__(self):
        self.available_devices = self._detect_devices()
        self.selected_device = None
        self.device_config = {}
        
    def _detect_devices(self) -> Dict[DeviceType, Dict]:
        """Detect all available acceleration devices"""
        devices = {}
        
        # Check for Intel iGPU
        intel_gpu = self._detect_intel_gpu()
        if intel_gpu:
            devices[DeviceType.INTEL_IGPU] = intel_gpu
            
        # Check for AMD NPU (XDNA1)
        amd_npu = self._detect_amd_npu()
        if amd_npu:
            devices[DeviceType.AMD_NPU] = amd_npu
            
        # Check for NVIDIA GPU
        nvidia_gpu = self._detect_nvidia_gpu()
        if nvidia_gpu:
            devices[DeviceType.NVIDIA_GPU] = nvidia_gpu
            
        # CPU is always available
        devices[DeviceType.CPU] = {
            "name": "CPU",
            "available": True,
            "priority": 0  # Lowest priority
        }
        
        return devices
    
    def _detect_intel_gpu(self) -> Optional[Dict]:
        """Detect Intel iGPU with Level Zero support"""
        # First check /dev/dri for Intel GPU
        try:
            result = subprocess.run(
                ["ls", "-la", "/dev/dri/"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            
            # Check renderD128 which is typically Intel iGPU
            if "renderD128" in result.stdout:
                # Verify it's Intel
                card_info = subprocess.run(
                    ["cat", "/sys/class/drm/card0/device/vendor"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if "0x8086" in card_info.stdout:  # Intel vendor ID
                    gpu_info = {
                        "name": "Intel UHD Graphics 770",  # We know this system has it
                        "model": "UHD Graphics 770",
                        "available": True,
                        "priority": 20,  # HIGHEST priority for Unicorn Amanuensis
                        "backend": "SYCL",
                        "expected_performance": "11-20x realtime",
                        "env_vars": {
                            "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                            "SYCL_DEVICE_FILTER": "gpu"
                        },
                        "note": "Optimized for Unicorn Amanuensis - saves RTX 5090 for LLM"
                    }
                    logger.info(f"✅ Detected Intel iGPU: {gpu_info['model']}")
                    return gpu_info
        except:
            pass
            
        # Fallback to sycl-ls check
        try:
            # Check for Intel GPU via sycl-ls
            result = subprocess.run(
                ["sycl-ls"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            
            if "Intel(R) UHD Graphics" in result.stdout or "Intel(R) Graphics" in result.stdout:
                # Extract GPU details
                gpu_info = {
                    "name": "Intel iGPU",
                    "available": True,
                    "priority": 20,  # HIGHEST priority for this project
                    "backend": "SYCL",
                    "env_vars": {
                        "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                        "SYCL_DEVICE_FILTER": "gpu"
                    }
                }
                
                # Check if it's UHD Graphics 770 (our optimized target)
                if "UHD Graphics 770" in result.stdout:
                    gpu_info["model"] = "UHD Graphics 770"
                    gpu_info["expected_performance"] = "11-20x realtime"
                    
                logger.info(f"✅ Detected Intel iGPU: {gpu_info.get('model', 'Unknown model')}")
                return gpu_info
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        # Alternative check via Level Zero
        try:
            result = subprocess.run(
                ["clinfo"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            
            if "Intel" in result.stdout and "Graphics" in result.stdout:
                return {
                    "name": "Intel iGPU",
                    "available": True,
                    "priority": 10,
                    "backend": "SYCL"
                }
        except:
            pass
            
        return None
    
    def _detect_amd_npu(self) -> Optional[Dict]:
        """Detect AMD XDNA1 NPU"""
        try:
            # Check for AMD NPU/APU devices
            # XDNA1 is in Ryzen AI processors
            
            # Check via ROCm
            result = subprocess.run(
                ["rocminfo"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            
            if "XDNA" in result.stdout or "Ryzen AI" in result.stdout:
                npu_info = {
                    "name": "AMD XDNA1 NPU",
                    "available": True,
                    "priority": 12,  # High priority for NPU
                    "backend": "ONNX Runtime",  # Or Vitis AI
                    "env_vars": {
                        "HSA_OVERRIDE_GFX_VERSION": "11.0.0"  # May need adjustment
                    },
                    "note": "Experimental support - development needed"
                }
                logger.info("✅ Detected AMD XDNA1 NPU (Ryzen AI)")
                return npu_info
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        # Alternative check via lspci
        try:
            result = subprocess.run(
                ["lspci"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            
            # Look for AMD AI accelerator
            if "AMD" in result.stdout and ("AI Engine" in result.stdout or "XDNA" in result.stdout):
                return {
                    "name": "AMD NPU",
                    "available": True,
                    "priority": 12,
                    "backend": "ONNX Runtime",
                    "note": "Detected via lspci - experimental"
                }
                
        except:
            pass
            
        return None
    
    def _detect_nvidia_gpu(self) -> Optional[Dict]:
        """Detect NVIDIA GPU with CUDA support"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip()
                return {
                    "name": f"NVIDIA {gpu_name}",
                    "available": True,
                    "priority": 5,  # Lower priority - save for LLM inference
                    "backend": "CUDA",
                    "env_vars": {
                        "CUDA_VISIBLE_DEVICES": "0"
                    },
                    "note": "Available but reserved for LLM inference"
                }
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        return None
    
    def select_best_device(self, prefer_device: Optional[DeviceType] = None) -> Tuple[DeviceType, Dict]:
        """Select the best available device for transcription"""
        
        if not self.available_devices:
            logger.warning("No acceleration devices detected, falling back to CPU")
            return DeviceType.CPU, {"name": "CPU", "available": True}
        
        # If user prefers a specific device and it's available
        if prefer_device and prefer_device in self.available_devices:
            device = self.available_devices[prefer_device]
            if device["available"]:
                logger.info(f"🎯 Using preferred device: {device['name']}")
                self.selected_device = prefer_device
                self.device_config = device
                return prefer_device, device
        
        # Otherwise, select by priority
        best_device = None
        best_priority = -1
        
        for device_type, config in self.available_devices.items():
            if config.get("priority", 0) > best_priority:
                best_priority = config["priority"]
                best_device = device_type
        
        if best_device:
            self.selected_device = best_device
            self.device_config = self.available_devices[best_device]
            logger.info(f"🎯 Auto-selected best device: {self.device_config['name']}")
            return best_device, self.device_config
        
        # Fallback to CPU
        return DeviceType.CPU, self.available_devices.get(DeviceType.CPU, {"name": "CPU"})
    
    def get_whisper_backend(self, device_type: DeviceType) -> str:
        """Get the appropriate Whisper backend for the device"""
        backends = {
            DeviceType.INTEL_IGPU: "whisper_igpu_real",  # Our SYCL implementation
            DeviceType.AMD_NPU: "whisper_amd_npu",  # To be developed
            DeviceType.NVIDIA_GPU: "whisper_cuda",  # Standard CUDA whisper
            DeviceType.CPU: "whisper_cpu"  # CPU fallback
        }
        return backends.get(device_type, "whisper_cpu")
    
    def configure_environment(self):
        """Configure environment variables for selected device"""
        if self.device_config and "env_vars" in self.device_config:
            for key, value in self.device_config["env_vars"].items():
                os.environ[key] = value
                logger.info(f"Set {key}={value}")
    
    def get_device_info(self) -> Dict:
        """Get information about all detected devices"""
        info = {
            "detected_devices": [],
            "selected_device": None,
            "recommendations": []
        }
        
        for device_type, config in self.available_devices.items():
            info["detected_devices"].append({
                "type": device_type.value,
                "name": config["name"],
                "backend": config.get("backend", "Unknown"),
                "priority": config.get("priority", 0),
                "note": config.get("note", "")
            })
        
        if self.selected_device:
            info["selected_device"] = {
                "type": self.selected_device.value,
                "name": self.device_config["name"],
                "backend": self.device_config.get("backend", "Unknown")
            }
        
        # Add recommendations
        if DeviceType.INTEL_IGPU in self.available_devices:
            info["recommendations"].append("Intel iGPU detected - optimal for this system")
        
        if DeviceType.AMD_NPU in self.available_devices:
            info["recommendations"].append("AMD NPU detected - experimental support available")
            info["recommendations"].append("Consider contributing to AMD NPU implementation")
        
        return info


def test_device_manager():
    """Test device detection and selection"""
    manager = DeviceManager()
    
    print("🔍 Detecting available devices...")
    info = manager.get_device_info()
    
    print("\n📋 Detected Devices:")
    for device in info["detected_devices"]:
        print(f"  - {device['name']} ({device['type']})")
        print(f"    Backend: {device['backend']}")
        print(f"    Priority: {device['priority']}")
        if device['note']:
            print(f"    Note: {device['note']}")
    
    print("\n🎯 Selecting best device...")
    device_type, config = manager.select_best_device()
    print(f"  Selected: {config['name']} ({device_type.value})")
    
    print("\n💡 Recommendations:")
    for rec in info["recommendations"]:
        print(f"  - {rec}")
    
    # Test with preference
    print("\n🔧 Testing with AMD NPU preference...")
    device_type, config = manager.select_best_device(prefer_device=DeviceType.AMD_NPU)
    print(f"  Result: {config['name']} ({device_type.value})")


if __name__ == "__main__":
    test_device_manager()