#!/usr/bin/env python3
"""
Test Native XRT Library Loading

Quick smoke test to verify libxrt_native.so loads and basic API works.
"""

import sys
from pathlib import Path

# Add xdna2 directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_library_load():
    """Test 1: Library loads without errors"""
    print("=" * 60)
    print("TEST 1: Library Loading")
    print("=" * 60)

    try:
        from encoder_cpp_native import WhisperEncoderNative
        print("SUCCESS: encoder_cpp_native module imported")
        return True
    except Exception as e:
        print(f"FAILED: Could not import module: {e}")
        return False

def test_create_instance():
    """Test 2: Can create WhisperEncoderNative instance"""
    print("\n" + "=" * 60)
    print("TEST 2: Instance Creation")
    print("=" * 60)

    try:
        from encoder_cpp_native import WhisperEncoderNative
        encoder = WhisperEncoderNative(model_size="base", use_4tile=False)
        print("SUCCESS: WhisperEncoderNative instance created")
        print(f"  Model size: base")
        print(f"  Use 4-tile: False")
        return True
    except Exception as e:
        print(f"FAILED: Could not create instance: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_get_model_dims():
    """Test 3: Can query model dimensions"""
    print("\n" + "=" * 60)
    print("TEST 3: Model Dimensions")
    print("=" * 60)

    try:
        from encoder_cpp_native import WhisperEncoderNative
        encoder = WhisperEncoderNative(model_size="base", use_4tile=False)
        dims = encoder.get_model_dims()
        print("SUCCESS: Got model dimensions")
        print(f"  n_mels: {dims['n_mels']}")
        print(f"  n_ctx: {dims['n_ctx']}")
        print(f"  n_state: {dims['n_state']}")
        print(f"  n_head: {dims['n_head']}")
        print(f"  n_layer: {dims['n_layer']}")

        # Verify dimensions match base model
        assert dims['n_mels'] == 80, f"Expected n_mels=80, got {dims['n_mels']}"
        assert dims['n_ctx'] == 1500, f"Expected n_ctx=1500, got {dims['n_ctx']}"
        assert dims['n_state'] == 512, f"Expected n_state=512, got {dims['n_state']}"
        assert dims['n_head'] == 8, f"Expected n_head=8, got {dims['n_head']}"
        assert dims['n_layer'] == 6, f"Expected n_layer=6, got {dims['n_layer']}"

        print("SUCCESS: All dimensions correct for base model")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Native XRT Library Load Test")
    print("=" * 60)
    print()

    results = []

    # Test 1: Library load
    results.append(("Library Loading", test_library_load()))

    # Test 2: Instance creation
    results.append(("Instance Creation", test_create_instance()))

    # Test 3: Model dimensions
    results.append(("Model Dimensions", test_get_model_dims()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    print()
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    success = all(p for _, p in results)
    if success:
        print("\nALL TESTS PASSED! Native XRT library is working correctly.")
        return 0
    else:
        print("\nSOME TESTS FAILED! Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
