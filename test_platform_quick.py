#!/usr/bin/env python3
"""Quick test of platform detection."""

import sys
import os

# Ensure we're in the service directory
service_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, service_dir)

def main():
    print("=" * 70)
    print("Platform Detection Quick Test")
    print("=" * 70)
    print()

    try:
        from runtime.platform_detector import PlatformDetector, Platform

        # Create detector
        detector = PlatformDetector()
        print("✅ PlatformDetector created")

        # Detect platform
        platform = detector.detect()
        print(f"\n🔍 Detected Platform: {platform.name} ({platform.value})")

        # Check available platforms
        print("\n📋 Checking platform availability:")

        # Check C++ runtime
        print("\n1. Checking C++ runtime (xdna2_cpp):")
        has_cpp = detector._has_cpp_runtime()
        print(f"   {'✅' if has_cpp else '❌'} C++ runtime available: {has_cpp}")

        if has_cpp:
            # Find the library
            xdna2_dir = os.path.join(service_dir, "xdna2")
            cpp_build_dir = os.path.join(xdna2_dir, "cpp", "build")
            lib_files = [
                os.path.join(cpp_build_dir, "libwhisper_encoder_cpp.so"),
                os.path.join(cpp_build_dir, "libwhisper_xdna2_cpp.so"),
            ]
            for lib_file in lib_files:
                exists = os.path.exists(lib_file)
                print(f"   {'✅' if exists else '❌'} {os.path.basename(lib_file)}: {exists}")

        # Check XDNA2
        print("\n2. Checking XDNA2 hardware:")
        has_xdna2 = detector._has_xdna2()
        print(f"   {'✅' if has_xdna2 else '❌'} XDNA2 NPU available: {has_xdna2}")

        # Check XDNA1
        print("\n3. Checking XDNA1 hardware:")
        has_xdna1 = detector._has_xdna1()
        print(f"   {'✅' if has_xdna1 else '❌'} XDNA1 NPU available: {has_xdna1}")

        # Summary
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"\n✅ Selected Platform: {platform.name}")
        print(f"\nPriority order:")
        print(f"  1. xdna2_cpp (C++ runtime)  {'✅ SELECTED' if platform == Platform.XDNA2_CPP else '❌'}")
        print(f"  2. xdna2 (Python runtime)   {'✅ SELECTED' if platform == Platform.XDNA2 else '❌'}")
        print(f"  3. xdna1 (Legacy NPU)       {'✅ SELECTED' if platform == Platform.XDNA1 else '❌'}")
        print(f"  4. cpu (Fallback)           {'✅ SELECTED' if platform == Platform.CPU else '❌'}")

        if platform == Platform.XDNA2_CPP:
            print("\n✅ SUCCESS: C++ runtime selected (highest priority)")
        elif has_cpp and platform != Platform.XDNA2_CPP:
            print("\n⚠️  WARNING: C++ runtime available but not selected")
            print(f"   Selected: {platform.name}")
        else:
            print(f"\n⚠️  INFO: Using {platform.name} platform")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
