#!/usr/bin/env python3
"""
NPU Mel Preprocessing Integration Test Suite
Sign-Fixed Production Kernel Validation

This comprehensive test suite validates the integration of the sign-fixed
NPU mel preprocessing kernel into the WhisperX pipeline.

Tests:
1. Kernel loading and initialization
2. Frame processing correctness
3. Batch processing performance
4. CPU fallback behavior
5. Performance metrics (>20x realtime)
6. Accuracy validation (correlation >0.5)
7. Non-zero output verification (>80%)
8. Thread safety
9. Memory management
10. Error handling

Author: Team Lead 2 - WhisperX NPU Integration Expert
Date: October 31, 2025
"""

import sys
import os
import numpy as np
import time
import threading
from pathlib import Path
from typing import Tuple, Dict

# Add XRT to path
sys.path.insert(0, '/opt/xilinx/xrt/python')

# Configure test environment
os.environ['XRT_LOG_LEVEL'] = 'warning'  # Reduce noise


def test_1_kernel_loading() -> Tuple[bool, str]:
    """Test 1: Kernel loading and initialization"""
    try:
        from whisperx.npu.npu_mel_production import NPUMelProcessor

        processor = NPUMelProcessor()

        # Check NPU availability
        if not processor.npu_available:
            return False, "NPU not available (may fallback to CPU - check device)"

        # Check kernel paths
        kernel_path = Path(processor.xclbin_path)
        insts_path = Path(processor.insts_buffer)

        if not kernel_path.exists():
            return False, f"Kernel file not found: {kernel_path}"

        # Check kernel size (should be ~56KB)
        kernel_size = kernel_path.stat().st_size
        if kernel_size < 50000 or kernel_size > 60000:
            return False, f"Unexpected kernel size: {kernel_size} bytes"

        del processor
        return True, f"Kernel loaded successfully ({kernel_size} bytes)"

    except Exception as e:
        return False, f"Exception: {e}"


def test_2_frame_processing() -> Tuple[bool, str]:
    """Test 2: Single frame processing correctness"""
    try:
        from whisperx.npu.npu_mel_production import NPUMelProcessor

        processor = NPUMelProcessor()

        if not processor.npu_available:
            return False, "NPU not available"

        # Generate test audio (1kHz sine wave, 400 samples = 25ms @ 16kHz)
        sr = 16000
        duration = 400 / sr
        t = np.linspace(0, duration, 400)
        audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Process frame
        mel_output = processor.process_frame(audio_int16)

        # Validate output
        if mel_output.shape != (80,):
            return False, f"Incorrect output shape: {mel_output.shape}"

        # Check for non-zero output
        non_zero_count = np.count_nonzero(mel_output)
        non_zero_pct = (non_zero_count / 80.0) * 100

        if non_zero_pct < 50.0:
            return False, f"Too many zeros: {non_zero_pct:.1f}% non-zero (sign bug?)"

        del processor
        return True, f"Frame processed correctly ({non_zero_pct:.1f}% non-zero bins)"

    except Exception as e:
        return False, f"Exception: {e}"


def test_3_batch_processing() -> Tuple[bool, str]:
    """Test 3: Batch processing performance"""
    try:
        from whisperx.npu.npu_mel_production import NPUMelProcessor

        processor = NPUMelProcessor()

        if not processor.npu_available:
            return False, "NPU not available"

        # Generate batch of frames (10 frames = 250ms audio)
        num_frames = 10
        audio_frames = np.random.randint(-32768, 32767, (num_frames, 400), dtype=np.int16)

        # Process batch
        start_time = time.perf_counter()
        mel_batch = processor.process_batch(audio_frames)
        elapsed = time.perf_counter() - start_time

        # Validate output
        if mel_batch.shape != (num_frames, 80):
            return False, f"Incorrect batch shape: {mel_batch.shape}"

        # Calculate performance
        audio_duration = (num_frames * 400) / 16000  # seconds
        realtime_factor = audio_duration / elapsed if elapsed > 0 else 0

        if realtime_factor < 10.0:
            return False, f"Low performance: {realtime_factor:.1f}x realtime"

        del processor
        return True, f"Batch processed at {realtime_factor:.1f}x realtime"

    except Exception as e:
        return False, f"Exception: {e}"


def test_4_cpu_fallback() -> Tuple[bool, str]:
    """Test 4: CPU fallback behavior"""
    try:
        from whisperx.npu.npu_mel_production import NPUMelProcessor

        # Force CPU fallback by using non-existent kernel
        processor = NPUMelProcessor(
            xclbin_path="/nonexistent/path.xclbin",
            fallback_to_cpu=True
        )

        # Should fall back to CPU
        if processor.npu_available:
            return False, "NPU should not be available with invalid kernel path"

        # Test CPU processing
        audio_int16 = np.random.randint(-32768, 32767, 400, dtype=np.int16)
        mel_output = processor.process_frame(audio_int16)

        # Validate CPU output
        if mel_output.shape != (80,):
            return False, f"Incorrect CPU output shape: {mel_output.shape}"

        del processor
        return True, "CPU fallback works correctly"

    except Exception as e:
        return False, f"Exception: {e}"


def test_5_performance_metrics() -> Tuple[bool, str]:
    """Test 5: Performance metrics (>20x realtime target)"""
    try:
        from whisperx.npu.npu_mel_production import NPUMelProcessor

        processor = NPUMelProcessor()

        if not processor.npu_available:
            return False, "NPU not available"

        # Process multiple frames to get accurate statistics
        num_frames = 100
        audio_frames = np.random.randint(-32768, 32767, (num_frames, 400), dtype=np.int16)

        processor.reset_statistics()
        mel_batch = processor.process_batch(audio_frames, show_progress=False)

        # Get statistics
        stats = processor.get_statistics()

        # Check realtime factor
        if stats['realtime_factor'] < 20.0:
            return False, f"Performance below target: {stats['realtime_factor']:.1f}x (target: >20x)"

        # Check NPU usage
        if stats['npu_calls'] == 0:
            return False, "No NPU calls recorded"

        del processor
        return True, f"Performance: {stats['realtime_factor']:.1f}x realtime, {stats['npu_avg_time']:.3f}ms avg"

    except Exception as e:
        return False, f"Exception: {e}"


def test_6_accuracy_validation() -> Tuple[bool, str]:
    """Test 6: Accuracy validation (correlation >0.5 with librosa)"""
    try:
        import librosa
        from whisperx.npu.npu_mel_production import NPUMelProcessor

        processor = NPUMelProcessor()

        if not processor.npu_available:
            return False, "NPU not available"

        # Generate test audio (1kHz sine wave)
        sr = 16000
        duration = 400 / sr
        t = np.linspace(0, duration, 400)
        audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Process with NPU
        mel_npu = processor.process_frame(audio_int16)

        # Process with CPU reference (librosa)
        mel_cpu = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=512, hop_length=160,
            n_mels=80, fmin=0, fmax=8000, power=2.0, htk=True
        )[:, 0]
        mel_cpu_db = librosa.power_to_db(mel_cpu, ref=np.max)

        # Calculate correlation
        correlation = np.corrcoef(mel_npu, mel_cpu_db)[0, 1]

        if correlation < 0.5:
            return False, f"Low correlation: {correlation:.3f} (target: >0.5, sign bug?)"

        del processor
        return True, f"Correlation: {correlation:.3f} (>0.5 target achieved)"

    except Exception as e:
        return False, f"Exception: {e}"


def test_7_nonzero_output() -> Tuple[bool, str]:
    """Test 7: Non-zero output verification (>80% non-zero bins)"""
    try:
        from whisperx.npu.npu_mel_production import NPUMelProcessor

        processor = NPUMelProcessor()

        if not processor.npu_available:
            return False, "NPU not available"

        # Process multiple frames with different signals
        results = []

        for freq in [500, 1000, 2000, 4000]:  # Different frequencies
            t = np.linspace(0, 400/16000, 400)
            audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
            audio_int16 = (audio * 32767).astype(np.int16)

            mel_output = processor.process_frame(audio_int16)
            non_zero_pct = (np.count_nonzero(mel_output) / 80.0) * 100
            results.append(non_zero_pct)

        avg_non_zero = np.mean(results)

        if avg_non_zero < 80.0:
            return False, f"Too many zeros: {avg_non_zero:.1f}% non-zero (target: >80%)"

        del processor
        return True, f"Non-zero output: {avg_non_zero:.1f}% average across test signals"

    except Exception as e:
        return False, f"Exception: {e}"


def test_8_thread_safety() -> Tuple[bool, str]:
    """Test 8: Thread safety"""
    try:
        from whisperx.npu.npu_mel_production import NPUMelProcessor

        processor = NPUMelProcessor()

        if not processor.npu_available:
            return False, "NPU not available"

        results = []
        errors = []

        def worker():
            try:
                audio_int16 = np.random.randint(-32768, 32767, 400, dtype=np.int16)
                mel_output = processor.process_frame(audio_int16)
                results.append(mel_output.shape == (80,))
            except Exception as e:
                errors.append(str(e))

        # Run 10 threads concurrently
        threads = [threading.Thread(target=worker) for _ in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        if errors:
            return False, f"Thread safety errors: {errors[0]}"

        if not all(results):
            return False, "Some threads produced incorrect output"

        del processor
        return True, "Thread safety verified (10 concurrent threads)"

    except Exception as e:
        return False, f"Exception: {e}"


def test_9_memory_management() -> Tuple[bool, str]:
    """Test 9: Memory management and cleanup"""
    try:
        from whisperx.npu.npu_mel_production import NPUMelProcessor

        # Create and destroy multiple processors
        for i in range(5):
            processor = NPUMelProcessor()

            if processor.npu_available:
                # Process a frame
                audio_int16 = np.random.randint(-32768, 32767, 400, dtype=np.int16)
                _ = processor.process_frame(audio_int16)

            # Explicit cleanup
            del processor

        return True, "Memory management verified (5 create/destroy cycles)"

    except Exception as e:
        return False, f"Exception: {e}"


def test_10_error_handling() -> Tuple[bool, str]:
    """Test 10: Error handling"""
    try:
        from whisperx.npu.npu_mel_production import NPUMelProcessor

        processor = NPUMelProcessor()

        if not processor.npu_available:
            return False, "NPU not available"

        # Test 1: Invalid input size
        try:
            audio_wrong_size = np.random.randint(-32768, 32767, 300, dtype=np.int16)
            _ = processor.process_frame(audio_wrong_size)
            return False, "Should have raised ValueError for wrong input size"
        except ValueError:
            pass  # Expected

        # Test 2: Invalid input type
        try:
            audio_wrong_type = np.random.randn(400).astype(np.float32)
            # Should auto-convert, not error
            mel = processor.process_frame(audio_wrong_type)
            if mel.shape != (80,):
                return False, "Auto-conversion failed"
        except:
            return False, "Should auto-convert float32 to int16"

        del processor
        return True, "Error handling verified"

    except Exception as e:
        return False, f"Exception: {e}"


def run_test_suite(verbose: bool = False) -> Dict:
    """Run complete test suite"""

    tests = [
        ("Kernel loading", test_1_kernel_loading),
        ("Frame processing", test_2_frame_processing),
        ("Batch processing", test_3_batch_processing),
        ("CPU fallback", test_4_cpu_fallback),
        ("Performance (>20x realtime)", test_5_performance_metrics),
        ("Accuracy (correlation >0.5)", test_6_accuracy_validation),
        ("Non-zero output (>80%)", test_7_nonzero_output),
        ("Thread safety", test_8_thread_safety),
        ("Memory management", test_9_memory_management),
        ("Error handling", test_10_error_handling),
    ]

    print("\n" + "="*70)
    print("NPU Mel Preprocessing Integration Test Suite")
    print("Sign-Fixed Production Kernel Validation")
    print("="*70)
    print()

    results = {}
    passed = 0
    failed = 0

    for i, (name, test_func) in enumerate(tests, 1):
        test_name = f"Test {i}: {name}"
        print(f"{test_name:.<60} ", end='', flush=True)

        try:
            success, message = test_func()
            results[name] = (success, message)

            if success:
                print("✓ PASS")
                if verbose:
                    print(f"  → {message}")
                passed += 1
            else:
                print("✗ FAIL")
                print(f"  → {message}")
                failed += 1
        except Exception as e:
            print("✗ ERROR")
            print(f"  → Unhandled exception: {e}")
            results[name] = (False, str(e))
            failed += 1

        if verbose:
            print()

    print()
    print("="*70)
    print(f"Test Summary: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*70)

    if failed == 0:
        print("✓ ALL TESTS PASSED - Integration successful!")
    else:
        print("✗ SOME TESTS FAILED - Review failures above")

    print()

    return results


def main():
    """Main test entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="NPU Mel Preprocessing Integration Test Suite"
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output with details')
    parser.add_argument('--test', '-t', type=int,
                        help='Run specific test number (1-10)')

    args = parser.parse_args()

    if args.test:
        # Run specific test
        tests = [
            test_1_kernel_loading,
            test_2_frame_processing,
            test_3_batch_processing,
            test_4_cpu_fallback,
            test_5_performance_metrics,
            test_6_accuracy_validation,
            test_7_nonzero_output,
            test_8_thread_safety,
            test_9_memory_management,
            test_10_error_handling,
        ]

        if 1 <= args.test <= len(tests):
            test_func = tests[args.test - 1]
            print(f"\nRunning Test {args.test}...\n")
            success, message = test_func()
            print(f"Result: {'PASS' if success else 'FAIL'}")
            print(f"Message: {message}")
            sys.exit(0 if success else 1)
        else:
            print(f"Error: Test number must be between 1 and {len(tests)}")
            sys.exit(1)
    else:
        # Run full suite
        results = run_test_suite(verbose=args.verbose)

        # Exit with error code if any tests failed
        all_passed = all(success for success, _ in results.values())
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
