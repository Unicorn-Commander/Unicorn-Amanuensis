#!/usr/bin/env python3
"""
NPU Mel Preprocessing Configuration
Sign-Fixed Production Kernel Settings

This module provides centralized configuration for NPU mel preprocessing,
including kernel paths, performance settings, and fallback options.

Usage:
    from whisperx.npu.npu_mel_config import NPU_MEL_CONFIG

    # Enable NPU mel
    if NPU_MEL_CONFIG['enabled']:
        from whisperx.npu.npu_mel_production import NPUMelProcessor
        processor = NPUMelProcessor(**NPU_MEL_CONFIG['processor_settings'])

Author: Team Lead 2 - WhisperX NPU Integration Expert
Date: October 31, 2025
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


# Default kernel paths
DEFAULT_KERNEL_DIR = Path(__file__).parent / "npu_optimization" / "mel_kernels" / "production_kernels"
DEFAULT_XCLBIN_PATH = DEFAULT_KERNEL_DIR / "mel_signfix_production.xclbin"
DEFAULT_INSTS_PATH = DEFAULT_KERNEL_DIR / "insts_signfix_production.bin"


# NPU Mel Configuration
NPU_MEL_CONFIG: Dict[str, Any] = {
    # Enable/disable NPU mel preprocessing
    "enabled": True,

    # Fallback to CPU if NPU unavailable
    "fallback_to_cpu": True,

    # Minimum correlation threshold for validation
    "correlation_threshold": 0.5,

    # Minimum non-zero bin percentage
    "nonzero_threshold": 80.0,  # percent

    # Performance monitoring
    "enable_monitoring": True,

    # NPU device ID
    "device_id": 0,

    # Kernel paths (None = use defaults)
    "xclbin_path": None,
    "insts_path": None,

    # Processor initialization settings
    "processor_settings": {
        "xclbin_path": None,  # None = use default
        "insts_path": None,   # None = use default
        "device_id": 0,
        "fallback_to_cpu": True,
        "enable_performance_monitoring": True,
    },

    # Performance expectations
    "performance": {
        "min_realtime_factor": 20.0,  # Minimum x realtime
        "max_frame_time_ms": 0.1,     # Maximum ms per frame
        "target_correlation": 0.62,    # Expected correlation with librosa
    },

    # Audio settings (Whisper defaults)
    "audio": {
        "sample_rate": 16000,
        "n_mels": 80,
        "frame_size": 400,     # samples (25ms @ 16kHz)
        "hop_length": 160,     # samples (10ms @ 16kHz)
        "n_fft": 512,
        "fmin": 0,
        "fmax": 8000,
    },
}


def get_config() -> Dict[str, Any]:
    """
    Get NPU mel configuration with environment variable overrides.

    Environment variables:
        NPU_MEL_ENABLED: "1" or "0" to enable/disable NPU mel
        NPU_MEL_FALLBACK: "1" or "0" to enable/disable CPU fallback
        NPU_MEL_CORRELATION_THRESHOLD: Float value for correlation threshold
        NPU_MEL_XCLBIN_PATH: Custom path to xclbin file
        NPU_MEL_INSTS_PATH: Custom path to instructions file
        NPU_MEL_DEVICE_ID: NPU device ID (default: 0)

    Returns:
        config: Configuration dictionary with overrides applied
    """
    config = NPU_MEL_CONFIG.copy()

    # Override from environment variables
    if "NPU_MEL_ENABLED" in os.environ:
        config["enabled"] = os.environ["NPU_MEL_ENABLED"] == "1"

    if "NPU_MEL_FALLBACK" in os.environ:
        config["fallback_to_cpu"] = os.environ["NPU_MEL_FALLBACK"] == "1"

    if "NPU_MEL_CORRELATION_THRESHOLD" in os.environ:
        try:
            config["correlation_threshold"] = float(os.environ["NPU_MEL_CORRELATION_THRESHOLD"])
        except ValueError:
            pass

    if "NPU_MEL_XCLBIN_PATH" in os.environ:
        config["xclbin_path"] = os.environ["NPU_MEL_XCLBIN_PATH"]

    if "NPU_MEL_INSTS_PATH" in os.environ:
        config["insts_path"] = os.environ["NPU_MEL_INSTS_PATH"]

    if "NPU_MEL_DEVICE_ID" in os.environ:
        try:
            config["device_id"] = int(os.environ["NPU_MEL_DEVICE_ID"])
        except ValueError:
            pass

    # Update processor settings from config
    config["processor_settings"]["xclbin_path"] = config["xclbin_path"]
    config["processor_settings"]["insts_path"] = config["insts_path"]
    config["processor_settings"]["device_id"] = config["device_id"]
    config["processor_settings"]["fallback_to_cpu"] = config["fallback_to_cpu"]
    config["processor_settings"]["enable_performance_monitoring"] = config["enable_monitoring"]

    return config


def validate_config(config: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
    """
    Validate NPU mel configuration.

    Args:
        config: Configuration dictionary to validate (None = use default)

    Returns:
        (valid, message): Validation result
    """
    if config is None:
        config = get_config()

    issues = []

    # Check if enabled
    if not config["enabled"]:
        return True, "NPU mel disabled in configuration"

    # Check kernel files if custom paths specified
    if config["xclbin_path"]:
        if not Path(config["xclbin_path"]).exists():
            issues.append(f"Custom XCLBIN not found: {config['xclbin_path']}")
    else:
        # Check default path
        if not DEFAULT_XCLBIN_PATH.exists():
            issues.append(f"Default XCLBIN not found: {DEFAULT_XCLBIN_PATH}")

    if config["insts_path"]:
        if not Path(config["insts_path"]).exists():
            issues.append(f"Custom instructions not found: {config['insts_path']}")
    else:
        # Check default path
        if not DEFAULT_INSTS_PATH.exists():
            issues.append(f"Default instructions not found: {DEFAULT_INSTS_PATH}")

    # Check NPU device
    if not Path("/dev/accel/accel0").exists():
        if not config["fallback_to_cpu"]:
            issues.append("NPU device not found and CPU fallback disabled")
        else:
            issues.append("NPU device not found (will use CPU fallback)")

    # Check thresholds
    if config["correlation_threshold"] < 0.0 or config["correlation_threshold"] > 1.0:
        issues.append(f"Invalid correlation threshold: {config['correlation_threshold']}")

    if config["nonzero_threshold"] < 0.0 or config["nonzero_threshold"] > 100.0:
        issues.append(f"Invalid nonzero threshold: {config['nonzero_threshold']}")

    if issues:
        return False, "\n".join(f"  - {issue}" for issue in issues)

    return True, "Configuration valid"


def print_config(config: Optional[Dict[str, Any]] = None):
    """
    Print NPU mel configuration in a formatted way.

    Args:
        config: Configuration dictionary to print (None = use default)
    """
    if config is None:
        config = get_config()

    print("\n" + "="*70)
    print("NPU Mel Preprocessing Configuration")
    print("="*70)

    print(f"\nStatus:")
    print(f"  Enabled:             {config['enabled']}")
    print(f"  Fallback to CPU:     {config['fallback_to_cpu']}")
    print(f"  Performance monitoring: {config['enable_monitoring']}")

    print(f"\nKernel Settings:")
    print(f"  Device ID:           {config['device_id']}")
    print(f"  XCLBIN path:         {config['xclbin_path'] or 'default'}")
    print(f"  Instructions path:   {config['insts_path'] or 'default'}")

    if not config['xclbin_path']:
        print(f"  Default XCLBIN:      {DEFAULT_XCLBIN_PATH}")
        if DEFAULT_XCLBIN_PATH.exists():
            size_kb = DEFAULT_XCLBIN_PATH.stat().st_size / 1024
            print(f"                       ({size_kb:.1f} KB)")

    if not config['insts_path']:
        print(f"  Default instructions: {DEFAULT_INSTS_PATH}")
        if DEFAULT_INSTS_PATH.exists():
            size_b = DEFAULT_INSTS_PATH.stat().st_size
            print(f"                       ({size_b} bytes)")

    print(f"\nValidation Thresholds:")
    print(f"  Correlation:         ≥ {config['correlation_threshold']:.2f}")
    print(f"  Non-zero bins:       ≥ {config['nonzero_threshold']:.1f}%")

    print(f"\nPerformance Targets:")
    print(f"  Realtime factor:     ≥ {config['performance']['min_realtime_factor']:.1f}x")
    print(f"  Frame time:          ≤ {config['performance']['max_frame_time_ms']:.3f} ms")
    print(f"  Target correlation:  {config['performance']['target_correlation']:.2f}")

    print(f"\nAudio Settings:")
    print(f"  Sample rate:         {config['audio']['sample_rate']} Hz")
    print(f"  Mel bins:            {config['audio']['n_mels']}")
    print(f"  Frame size:          {config['audio']['frame_size']} samples")
    print(f"  Hop length:          {config['audio']['hop_length']} samples")
    print(f"  FFT size:            {config['audio']['n_fft']}")
    print(f"  Frequency range:     {config['audio']['fmin']}-{config['audio']['fmax']} Hz")

    print("="*70)

    # Validate and show status
    valid, message = validate_config(config)
    if valid:
        print("✓ Configuration valid")
    else:
        print("✗ Configuration issues:")
        print(message)
    print()


def create_processor_from_config(config: Optional[Dict[str, Any]] = None):
    """
    Create NPU mel processor from configuration.

    Args:
        config: Configuration dictionary (None = use default with env overrides)

    Returns:
        processor: Initialized NPUMelProcessor or None if disabled/failed
    """
    if config is None:
        config = get_config()

    if not config["enabled"]:
        print("NPU mel preprocessing disabled in configuration")
        return None

    # Validate configuration
    valid, message = validate_config(config)
    if not valid:
        print(f"Configuration validation failed:\n{message}")
        if not config["fallback_to_cpu"]:
            return None
        print("Attempting to initialize with CPU fallback...")

    # Import processor
    try:
        from whisperx.npu.npu_mel_production import NPUMelProcessor
    except ImportError as e:
        print(f"Failed to import NPUMelProcessor: {e}")
        return None

    # Create processor with config settings
    try:
        processor = NPUMelProcessor(**config["processor_settings"])

        if processor.npu_available:
            print("✓ NPU mel processor initialized successfully")
        else:
            print("⚠ NPU not available, using CPU fallback")

        return processor

    except Exception as e:
        print(f"Failed to initialize NPU mel processor: {e}")
        return None


if __name__ == "__main__":
    """Configuration validation and testing"""
    import sys

    print("NPU Mel Configuration Utility\n")

    # Get configuration with environment overrides
    config = get_config()

    # Print configuration
    print_config(config)

    # Try to create processor
    print("\nTesting processor initialization...")
    processor = create_processor_from_config(config)

    if processor:
        print("✓ Processor created successfully")

        # Quick test
        import numpy as np
        test_audio = np.random.randint(-32768, 32767, 400, dtype=np.int16)
        try:
            mel = processor.process_frame(test_audio)
            print(f"✓ Test frame processed: {mel.shape}")

            stats = processor.get_statistics()
            print(f"✓ Statistics available: {stats['npu_calls']} NPU calls")

        except Exception as e:
            print(f"✗ Test failed: {e}")
            sys.exit(1)
    else:
        print("✗ Failed to create processor")
        sys.exit(1)

    print("\n✓ All configuration tests passed")
    sys.exit(0)
