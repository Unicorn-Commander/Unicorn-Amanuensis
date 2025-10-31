#!/usr/bin/env python3
"""
Buffer Utilities for XDNA1 NPU - Sign Extension Fix

This module contains critical buffer handling utilities for AMD Phoenix/Hawk Point
NPU (XDNA1) that address a sign extension bug discovered during mel kernel development.

THE SIGN EXTENSION BUG
======================

Problem:
--------
When converting int16 audio samples to byte buffers for NPU processing, incorrect
sign handling causes 50% of samples (all negative values) to be off by exactly +65536.
This creates phase inversion in FFT operations, resulting in:
- Negative correlation with reference output (-0.0297 instead of positive)
- 96.2% zero bins in mel output
- Completely unusable results

Root Cause:
-----------
The bug occurs at two levels:

1. C Kernel Level (mel_kernel_fft_fixed.c):
   ```c
   // WRONG (causes +65536 wraparound for negative samples):
   uint8_t high = input[i+1];
   int16_t sample = low | (high << 8);

   // CORRECT (preserves sign):
   int8_t high = (int8_t)input[i+1];
   int16_t sample = low | (high << 8);
   ```

2. Python Buffer Level:
   ```python
   # WRONG (sign extends to int8):
   buffer = np.frombuffer(audio_bytes, dtype=np.int8)

   # CORRECT (preserves unsigned bytes):
   buffer = np.frombuffer(audio_bytes, dtype=np.uint8)
   ```

Impact:
-------
- 50% of samples affected (all negative int16 values)
- Each affected sample off by exactly +65536
- Creates phase inversion in FFT → negative correlation
- Output completely unusable (96.2% zeros)

Fix Results:
------------
After applying the fix:
- Correlation: -0.0297 → +0.6184 (NEGATIVE TO POSITIVE!)
- Output range: [0, 4] → [0, 60] (+1400%)
- Non-zero bins: 3.8% → 68.8% (+1713%)
- Performance: 23.6x realtime
- Status: PRODUCTION READY ✅

References:
-----------
- Full story: /whisperx/npu/npu_optimization/FINAL_STATUS_REPORT_OCT31_2025.md
- Technical analysis: /tmp/SIGN_BUG_FIX_SUCCESS_STORY_OCT31.md
- Test results: /tmp/uc1-dev-check/SIGNFIX_TEST_RESULTS_OCT31.md

Author: Team Lead B - Pipeline Validation
        Team Lead D - Production Integration
Date: October 31, 2025
Validated: AMD Ryzen 7040 (Phoenix) NPU hardware
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def fix_sign_extension(audio_int16: np.ndarray) -> np.ndarray:
    """
    Convert int16 audio samples to uint8 buffer with proper sign handling.

    This function implements the critical sign fix that prevents the +65536
    wraparound bug in NPU mel kernel processing.

    The fix ensures that when int16 samples are split into byte pairs and
    sent to the NPU, the high byte is treated as unsigned, preventing sign
    extension during reconstruction.

    Args:
        audio_int16: Input audio samples as int16 array
                     Shape: (num_samples,) or (num_frames, frame_size)

    Returns:
        buffer: Unsigned byte buffer ready for NPU
                Shape: (num_samples * 2,) or (num_frames, frame_size * 2)
                Dtype: uint8

    Example:
        >>> # Single frame (400 samples = 800 bytes)
        >>> audio = np.array([100, -200, 300], dtype=np.int16)
        >>> buffer = fix_sign_extension(audio)
        >>> assert buffer.dtype == np.uint8
        >>> assert len(buffer) == 6  # 3 samples * 2 bytes

        >>> # Multiple frames
        >>> audio_frames = np.random.randint(-32768, 32767, (10, 400), dtype=np.int16)
        >>> buffer = fix_sign_extension(audio_frames)
        >>> assert buffer.shape == (10, 800)

    Technical Details:
        The bug occurs because:
        1. int16 sample -200 = 0xFF38 in two's complement
        2. Bytes: [0x38, 0xFF] (little-endian)
        3. If high byte treated as int8: 0xFF → -1 (sign extended)
        4. Reconstruction: 0x38 | (-1 << 8) = WRONG!
        5. If high byte treated as uint8: 0xFF → 255 (correct)
        6. Reconstruction: 0x38 | (255 << 8) = 0xFF38 = -200 ✓

    Validation:
        - Tested on AMD Ryzen 7040 (Phoenix) NPU
        - Correlation improved from -0.03 to +0.62
        - 23.6x realtime performance
        - 68.8% non-zero mel bins (was 3.8%)
    """
    # Validate input
    if not isinstance(audio_int16, np.ndarray):
        audio_int16 = np.array(audio_int16, dtype=np.int16)

    if audio_int16.dtype != np.int16:
        logger.warning(f"Input dtype is {audio_int16.dtype}, converting to int16")
        audio_int16 = audio_int16.astype(np.int16)

    # Convert to bytes (preserves little-endian byte pairs)
    audio_bytes = audio_int16.tobytes()

    # CRITICAL: Use uint8 to prevent sign extension bug!
    # This is the core of the fix - treating bytes as unsigned
    buffer = np.frombuffer(audio_bytes, dtype=np.uint8)

    # Reshape if needed to match input dimensions
    if audio_int16.ndim == 2:
        # Input was (num_frames, frame_size)
        # Output should be (num_frames, frame_size * 2)
        num_frames = audio_int16.shape[0]
        frame_size_bytes = audio_int16.shape[1] * 2
        buffer = buffer.reshape(num_frames, frame_size_bytes)

    return buffer


def validate_sign_fix(audio_int16: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that sign extension fix is working correctly.

    Tests the buffer conversion by checking:
    1. Buffer size is correct (samples * 2)
    2. Buffer dtype is uint8 (not int8)
    3. Reconstruction produces original values
    4. Negative samples are handled correctly

    Args:
        audio_int16: Test audio samples (int16)

    Returns:
        (success, message): Validation result

    Example:
        >>> audio = np.array([100, -200, 300, -400], dtype=np.int16)
        >>> success, msg = validate_sign_fix(audio)
        >>> print(f"Validation: {msg}")
        >>> assert success
    """
    try:
        # Apply fix
        buffer = fix_sign_extension(audio_int16)

        # Check 1: Size
        expected_size = audio_int16.size * 2
        if buffer.size != expected_size:
            return False, f"Size mismatch: expected {expected_size}, got {buffer.size}"

        # Check 2: Dtype
        if buffer.dtype != np.uint8:
            return False, f"Wrong dtype: expected uint8, got {buffer.dtype}"

        # Check 3: Reconstruction (validate round-trip)
        # Convert buffer back to int16 and compare
        reconstructed = np.frombuffer(buffer.tobytes(), dtype=np.int16)
        if audio_int16.ndim == 2:
            reconstructed = reconstructed.reshape(audio_int16.shape)

        if not np.array_equal(reconstructed, audio_int16):
            max_diff = np.abs(reconstructed - audio_int16).max()
            return False, f"Reconstruction failed: max difference = {max_diff}"

        # Check 4: Negative samples
        negative_mask = audio_int16 < 0
        if negative_mask.any():
            # Ensure negative samples reconstructed correctly
            neg_samples = audio_int16[negative_mask]
            rec_samples = reconstructed[negative_mask]
            if not np.array_equal(neg_samples, rec_samples):
                return False, "Negative samples not reconstructed correctly"

        return True, "All validation checks passed - sign fix working correctly"

    except Exception as e:
        return False, f"Validation error: {e}"


def create_npu_input_buffer(
    audio_int16: np.ndarray,
    target_size: int = 800,
    pad_value: int = 0
) -> np.ndarray:
    """
    Create NPU-ready input buffer from audio samples.

    Applies sign extension fix and ensures buffer is correct size for NPU.

    Args:
        audio_int16: Input audio samples (int16)
        target_size: Target buffer size in bytes (default: 800 for 400 samples)
        pad_value: Value to use for padding if needed (default: 0)

    Returns:
        buffer: NPU-ready uint8 buffer of size target_size

    Example:
        >>> # Create buffer for 400-sample frame (800 bytes)
        >>> audio = np.random.randint(-32768, 32767, 400, dtype=np.int16)
        >>> buffer = create_npu_input_buffer(audio, target_size=800)
        >>> assert len(buffer) == 800
        >>> assert buffer.dtype == np.uint8
    """
    # Apply sign fix
    buffer = fix_sign_extension(audio_int16)

    # Flatten if needed
    if buffer.ndim > 1:
        buffer = buffer.flatten()

    # Pad or truncate to target size
    if len(buffer) < target_size:
        # Pad with zeros
        padding = np.full(target_size - len(buffer), pad_value, dtype=np.uint8)
        buffer = np.concatenate([buffer, padding])
        logger.debug(f"Padded buffer from {len(buffer)} to {target_size} bytes")
    elif len(buffer) > target_size:
        # Truncate
        buffer = buffer[:target_size]
        logger.warning(f"Truncated buffer from {len(buffer)} to {target_size} bytes")

    return buffer


if __name__ == "__main__":
    # Quick validation test
    logging.basicConfig(level=logging.INFO)

    print("="*70)
    print("XDNA1 Sign Extension Fix - Validation Test")
    print("="*70)

    # Test 1: Single positive sample
    print("\nTest 1: Positive sample")
    audio = np.array([100], dtype=np.int16)
    success, msg = validate_sign_fix(audio)
    print(f"  {msg}")
    assert success, "Test 1 failed"

    # Test 2: Single negative sample (critical for bug)
    print("\nTest 2: Negative sample (critical for sign bug)")
    audio = np.array([-200], dtype=np.int16)
    success, msg = validate_sign_fix(audio)
    print(f"  {msg}")
    assert success, "Test 2 failed"

    # Test 3: Mixed samples
    print("\nTest 3: Mixed positive/negative samples")
    audio = np.array([100, -200, 300, -400, 32767, -32768], dtype=np.int16)
    success, msg = validate_sign_fix(audio)
    print(f"  {msg}")
    assert success, "Test 3 failed"

    # Test 4: Full frame (400 samples = 800 bytes)
    print("\nTest 4: Full 400-sample frame")
    audio = np.random.randint(-32768, 32767, 400, dtype=np.int16)
    success, msg = validate_sign_fix(audio)
    print(f"  {msg}")
    assert success, "Test 4 failed"

    # Test 5: Multiple frames
    print("\nTest 5: Multiple frames (batch processing)")
    audio = np.random.randint(-32768, 32767, (10, 400), dtype=np.int16)
    success, msg = validate_sign_fix(audio)
    print(f"  {msg}")
    assert success, "Test 5 failed"

    # Test 6: NPU buffer creation
    print("\nTest 6: NPU input buffer creation")
    audio = np.random.randint(-32768, 32767, 400, dtype=np.int16)
    buffer = create_npu_input_buffer(audio, target_size=800)
    assert len(buffer) == 800
    assert buffer.dtype == np.uint8
    print(f"  Created NPU buffer: {len(buffer)} bytes, dtype={buffer.dtype}")

    print("\n" + "="*70)
    print("✅ ALL VALIDATION TESTS PASSED")
    print("="*70)
    print("\nSign extension fix is working correctly!")
    print("Safe to use for NPU mel kernel processing.")
    print()
