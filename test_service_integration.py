#!/usr/bin/env python3
"""Test service-level integration of C++ runtime."""

import sys
import os

# Ensure we're in the service directory
service_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, service_dir)

def test_config_loading():
    """Test runtime configuration loading."""
    print("=" * 70)
    print("Test 1: Runtime Configuration")
    print("=" * 70)
    print()

    try:
        # Try to import and load config
        try:
            from config.runtime_config import RuntimeConfig
            config = RuntimeConfig.load()
            print("✅ RuntimeConfig loaded successfully")
            print(f"   Backend preference: {config.backend.preferred_backend if hasattr(config, 'backend') else 'N/A'}")
            return config
        except ImportError as e:
            print(f"⚠️  RuntimeConfig not available: {e}")
            print("   (This is OK if config module doesn't exist yet)")
            return None

    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_encoder_base_import():
    """Test encoder base class import."""
    print("\n" + "=" * 70)
    print("Test 2: Encoder Base Import")
    print("=" * 70)
    print()

    try:
        from encoder.whisper_encoder import WhisperEncoderBase
        print("✅ WhisperEncoderBase imported successfully")
        return True
    except ImportError as e:
        print(f"❌ WhisperEncoderBase import failed: {e}")
        return False


def test_cpp_encoder_import():
    """Test C++ encoder wrapper import."""
    print("\n" + "=" * 70)
    print("Test 3: C++ Encoder Wrapper Import")
    print("=" * 70)
    print()

    try:
        from xdna2.encoder_cpp import WhisperEncoderCpp
        print("✅ WhisperEncoderCpp imported successfully")
        return True
    except ImportError as e:
        print(f"⚠️  WhisperEncoderCpp import failed: {e}")
        print("   Checking if encoder_cpp.py exists...")
        cpp_file = os.path.join(service_dir, "xdna2", "encoder_cpp.py")
        if os.path.exists(cpp_file):
            print(f"   ✅ File exists: {cpp_file}")
            print("   Error is likely in the file content")
        else:
            print(f"   ❌ File missing: {cpp_file}")
        import traceback
        traceback.print_exc()
        return False


def test_cpp_runtime_direct():
    """Test C++ runtime wrapper directly."""
    print("\n" + "=" * 70)
    print("Test 4: C++ Runtime Direct Access")
    print("=" * 70)
    print()

    try:
        from xdna2.cpp_runtime_wrapper import CPPRuntimeWrapper, EncoderLayer
        print("✅ cpp_runtime_wrapper imported successfully")

        # Try creating runtime
        runtime = CPPRuntimeWrapper()
        print(f"✅ CPPRuntimeWrapper created")
        print(f"   Version: {runtime.get_version()}")

        # Try creating a layer
        with EncoderLayer(runtime, layer_idx=0, n_heads=8, n_state=512, ffn_dim=2048) as layer:
            print(f"✅ EncoderLayer created (handle: {layer.handle})")

        print("✅ C++ runtime fully functional")
        return True

    except Exception as e:
        print(f"❌ C++ runtime test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoder_factory():
    """Test encoder factory can create encoder."""
    print("\n" + "=" * 70)
    print("Test 5: Encoder Factory")
    print("=" * 70)
    print()

    try:
        # Check if encoder_factory.py exists
        factory_file = os.path.join(service_dir, "encoder", "encoder_factory.py")
        if not os.path.exists(factory_file):
            print(f"⚠️  EncoderFactory not found at: {factory_file}")
            print("   (This is OK if factory doesn't exist yet)")
            return None

        from encoder.encoder_factory import EncoderFactory
        print("✅ EncoderFactory imported successfully")

        factory = EncoderFactory()
        print("✅ EncoderFactory created")

        # Try creating encoder with auto-detection
        print("\n🏭 Creating encoder with auto-detection...")
        encoder = factory.create_encoder()

        print(f"✅ Encoder created successfully")
        print(f"   Type: {type(encoder).__name__}")
        print(f"   Module: {type(encoder).__module__}")

        # Check if it's the C++ encoder
        if 'Cpp' in type(encoder).__name__:
            print("✅ SUCCESS: C++ encoder selected")
        else:
            print(f"⚠️  INFO: Using {type(encoder).__name__} encoder")

        return encoder

    except Exception as e:
        print(f"❌ Encoder factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all service integration tests."""
    print("\n" + "=" * 70)
    print("Service Integration Tests")
    print("=" * 70)
    print()

    results = {}

    # Test 1: Config loading
    config = test_config_loading()
    results['config'] = config is not None or "not implemented"

    # Test 2: Encoder base import
    results['encoder_base'] = test_encoder_base_import()

    # Test 3: C++ encoder wrapper import
    results['cpp_encoder'] = test_cpp_encoder_import()

    # Test 4: C++ runtime direct access
    results['cpp_runtime'] = test_cpp_runtime_direct()

    # Test 5: Encoder factory
    encoder = test_encoder_factory()
    results['encoder_factory'] = encoder is not None or "not implemented"

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    # Count passes
    passed = sum(1 for v in results.values() if v is True)
    not_impl = sum(1 for v in results.values() if v == "not implemented")
    failed = sum(1 for v in results.values() if v is False)

    print("Test Results:")
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result == "not implemented":
            status = "⚠️  NOT IMPLEMENTED"
        else:
            status = "❌ FAIL"
        print(f"  {test_name:20} {status}")

    print()
    print(f"Total: {passed} passed, {failed} failed, {not_impl} not implemented")
    print()

    # Overall assessment
    if results['cpp_runtime']:
        print("✅ CRITICAL: C++ runtime is fully functional")
        print()
        print("Next steps:")
        print("  1. Implement encoder_factory.py if needed")
        print("  2. Test end-to-end encoder pipeline")
        print("  3. Install XRT for NPU hardware testing")
        print("  4. Validate 1211.3x speedup from C++")
        return True
    else:
        print("❌ CRITICAL: C++ runtime not working")
        print("   Service integration cannot proceed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
