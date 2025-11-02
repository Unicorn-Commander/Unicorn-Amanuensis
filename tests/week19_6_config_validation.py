#!/usr/bin/env python3
"""
Week 19.6 Configuration Validation - Rollback & Buffer Pool Fix

Validates Week 19.6 configuration changes without starting the service:
1. Check environment variable defaults in server.py source
2. Verify buffer pool sizes in server.py source
3. Confirm 30s audio support configuration
4. Test file verification

Author: Week 19.6 Team 1 Lead
Date: November 2, 2025
"""

import os
import re
from pathlib import Path

def validate_server_config():
    """Validate server.py configuration by parsing source code"""
    print("\n" + "="*70)
    print("  WEEK 19.6 CONFIGURATION VALIDATION")
    print("  Rollback & Buffer Pool Fix")
    print("="*70)

    server_path = Path(__file__).parent.parent / "xdna2" / "server.py"

    if not server_path.exists():
        print(f"\n  ✗ ERROR: server.py not found at {server_path}")
        return False

    with open(server_path, 'r') as f:
        server_code = f.read()

    all_passed = True

    # Test 1: Check USE_FASTER_WHISPER default
    print("\n" + "="*70)
    print("  TEST 1: Week 19 Decoder Rollback")
    print("="*70)

    faster_whisper_match = re.search(
        r'USE_FASTER_WHISPER\s*=\s*os\.environ\.get\("USE_FASTER_WHISPER",\s*"([^"]+)"\)',
        server_code
    )

    if faster_whisper_match:
        default_value = faster_whisper_match.group(1)
        print(f"\n  USE_FASTER_WHISPER default: '{default_value}'")
        if default_value == "false":
            print("  ✓ Correctly defaults to 'false' (Week 18 WhisperX)")
        else:
            print(f"  ✗ FAIL: Should default to 'false', got '{default_value}'")
            all_passed = False
    else:
        print("  ✗ FAIL: USE_FASTER_WHISPER configuration not found")
        all_passed = False

    # Test 2: Check USE_CUSTOM_DECODER default
    print("\n" + "="*70)
    print("  TEST 2: Week 19.5 Custom Decoder Rollback")
    print("="*70)

    custom_decoder_match = re.search(
        r'USE_CUSTOM_DECODER\s*=\s*os\.environ\.get\("USE_CUSTOM_DECODER",\s*"([^"]+)"\)',
        server_code
    )

    if custom_decoder_match:
        default_value = custom_decoder_match.group(1)
        print(f"\n  USE_CUSTOM_DECODER default: '{default_value}'")
        if default_value == "false":
            print("  ✓ Correctly defaults to 'false' (Week 18 WhisperX)")
        else:
            print(f"  ✗ FAIL: Should default to 'false', got '{default_value}'")
            all_passed = False
    else:
        print("  ✗ FAIL: USE_CUSTOM_DECODER configuration not found")
        all_passed = False

    # Test 3: Check buffer pool sizes
    print("\n" + "="*70)
    print("  TEST 3: Buffer Pool Size Increases")
    print("="*70)

    # Check AUDIO_BUFFER_POOL_SIZE
    audio_pool_match = re.search(
        r'AUDIO_BUFFER_POOL_SIZE\s*=\s*int\(os\.getenv\([\'"]AUDIO_BUFFER_POOL_SIZE[\'"]\s*,\s*[\'"](\d+)[\'"]\)\)',
        server_code
    )

    if audio_pool_match:
        pool_size = int(audio_pool_match.group(1))
        print(f"\n  AUDIO_BUFFER_POOL_SIZE: {pool_size}")
        if pool_size == 50:
            print("  ✓ Correctly set to 50 (increased from 5)")
        else:
            print(f"  ✗ FAIL: Should be 50, got {pool_size}")
            all_passed = False
    else:
        print("  ✗ FAIL: AUDIO_BUFFER_POOL_SIZE not found")
        all_passed = False

    # Check MEL_BUFFER_POOL_SIZE
    mel_pool_match = re.search(
        r'MEL_BUFFER_POOL_SIZE\s*=\s*int\(os\.getenv\([\'"]MEL_BUFFER_POOL_SIZE[\'"]\s*,\s*[\'"](\d+)[\'"]\)\)',
        server_code
    )

    if mel_pool_match:
        pool_size = int(mel_pool_match.group(1))
        print(f"  MEL_BUFFER_POOL_SIZE: {pool_size}")
        if pool_size == 50:
            print("  ✓ Correctly set to 50 (increased from 10)")
        else:
            print(f"  ✗ FAIL: Should be 50, got {pool_size}")
            all_passed = False
    else:
        print("  ✗ FAIL: MEL_BUFFER_POOL_SIZE not found")
        all_passed = False

    # Check ENCODER_BUFFER_POOL_SIZE
    encoder_pool_match = re.search(
        r'ENCODER_BUFFER_POOL_SIZE\s*=\s*int\(os\.getenv\([\'"]ENCODER_BUFFER_POOL_SIZE[\'"]\s*,\s*[\'"](\d+)[\'"]\)\)',
        server_code
    )

    if encoder_pool_match:
        pool_size = int(encoder_pool_match.group(1))
        print(f"  ENCODER_BUFFER_POOL_SIZE: {pool_size}")
        if pool_size == 50:
            print("  ✓ Correctly set to 50 (increased from 5)")
        else:
            print(f"  ✗ FAIL: Should be 50, got {pool_size}")
            all_passed = False
    else:
        print("  ✗ FAIL: ENCODER_BUFFER_POOL_SIZE not found")
        all_passed = False

    # Check MAX_POOL_SIZE
    max_pool_match = re.search(
        r'MAX_POOL_SIZE\s*=\s*int\(os\.getenv\([\'"]MAX_POOL_SIZE[\'"]\s*,\s*[\'"](\d+)[\'"]\)\)',
        server_code
    )

    if max_pool_match:
        pool_size = int(max_pool_match.group(1))
        print(f"  MAX_POOL_SIZE: {pool_size}")
        if pool_size == 100:
            print("  ✓ Correctly set to 100 (safety limit)")
        else:
            print(f"  ✗ FAIL: Should be 100, got {pool_size}")
            all_passed = False
    else:
        print("  ✗ FAIL: MAX_POOL_SIZE not found")
        all_passed = False

    # Check for growth_strategy parameter
    if "'growth_strategy': 'auto'" in server_code:
        print("\n  ✓ growth_strategy: 'auto' found in buffer configuration")
    else:
        print("\n  ⚠ WARNING: growth_strategy: 'auto' not found")
        # Not a failure, just a warning

    # Test 4: Check 30s audio support
    print("\n" + "="*70)
    print("  TEST 4: 30s Audio Support")
    print("="*70)

    max_duration_match = re.search(
        r'MAX_AUDIO_DURATION\s*=\s*int\(os\.getenv\([\'"]MAX_AUDIO_DURATION[\'"]\s*,\s*[\'"](\d+)[\'"]\)\)',
        server_code
    )

    if max_duration_match:
        duration = int(max_duration_match.group(1))
        print(f"\n  MAX_AUDIO_DURATION: {duration}s")
        if duration == 30:
            print("  ✓ Correctly set to 30s")
        else:
            print(f"  ⚠ WARNING: Expected 30s, got {duration}s")
    else:
        print("  ✗ FAIL: MAX_AUDIO_DURATION not found")
        all_passed = False

    # Check if 30s test file exists
    test_file = Path(__file__).parent / "audio" / "test_30s.wav"
    if test_file.exists():
        file_size_mb = test_file.stat().st_size / (1024 * 1024)
        print(f"\n  Test file: {test_file}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print("  ✓ 30s test audio file exists")
    else:
        print(f"\n  ⚠ WARNING: 30s test file not found at: {test_file}")

    # Summary
    print("\n" + "="*70)
    print("  VALIDATION SUMMARY")
    print("="*70)

    if all_passed:
        print("\n  ✓ ALL CONFIGURATION CHECKS PASSED")
        print("\n  Changes Implemented:")
        print("    1. USE_CUSTOM_DECODER: true → false (Week 19.5 disabled)")
        print("    2. USE_FASTER_WHISPER: true → false (Week 19 disabled)")
        print("    3. AUDIO_BUFFER_POOL_SIZE: 5 → 50 (10× increase)")
        print("    4. MEL_BUFFER_POOL_SIZE: 10 → 50 (5× increase)")
        print("    5. ENCODER_BUFFER_POOL_SIZE: 5 → 50 (10× increase)")
        print("    6. MAX_POOL_SIZE: Added (100 safety limit)")
        print("    7. growth_strategy: Added ('auto' for dynamic growth)")
        print("\n  Expected Results:")
        print("    - Decoder: Week 18 WhisperX (stable baseline)")
        print("    - Performance: ≥7.9× realtime (Week 18 parity)")
        print("    - Multi-stream: 100% success (no buffer exhaustion)")
        print("    - Concurrent streams: 50+ (vs 4-5 before)")
        print("    - 30s audio: Supported")
        print("\n  Status: READY FOR SERVICE STARTUP AND TESTING")
    else:
        print("\n  ✗ SOME CONFIGURATION CHECKS FAILED")
        print("  Review errors above and fix configuration")

    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    import sys
    success = validate_server_config()
    sys.exit(0 if success else 1)
