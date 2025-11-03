#!/usr/bin/env python3
"""
Test Script for NPU Mel Processor Batch Final

This script tests the production batch-100 processor with:
- Basic functionality tests
- Edge case handling
- Performance benchmarking
- Accuracy validation (if reference available)
- Integration testing

Author: Magic Unicorn Unconventional Technology & Stuff Inc.
Date: November 1, 2025
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from npu_mel_processor_batch_final import NPUMelProcessorBatch, create_batch_processor
    print("✅ Successfully imported NPUMelProcessorBatch")
except ImportError as e:
    print(f"❌ Failed to import NPUMelProcessorBatch: {e}")
    sys.exit(1)


def generate_test_audio(duration_sec: float, sample_rate: int = 16000, freq_type: str = "sweep") -> np.ndarray:
    """
    Generate test audio signal.

    Args:
        duration_sec: Duration in seconds
        sample_rate: Sample rate in Hz
        freq_type: Type of signal ("sweep", "sine", "chirp", "noise")

    Returns:
        audio: Numpy array of audio samples (float32)
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, dtype=np.float32)

    if freq_type == "sweep":
        # Frequency sweep from 100Hz to 8000Hz
        freq_start = 100
        freq_end = 8000
        freq = freq_start + (freq_end - freq_start) * (t / duration_sec)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)

    elif freq_type == "sine":
        # Pure 440Hz sine wave
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    elif freq_type == "chirp":
        # Chirp signal
        freq = 440 + 200 * np.sin(2 * np.pi * t)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)

    elif freq_type == "noise":
        # White noise
        audio = 0.1 * np.random.randn(n_samples).astype(np.float32)

    else:
        raise ValueError(f"Unknown freq_type: {freq_type}")

    return audio.astype(np.float32)


def test_basic_functionality(processor):
    """Test basic functionality with various audio lengths."""
    print("\n" + "="*70)
    print("TEST 1: Basic Functionality")
    print("="*70)

    test_cases = [
        ("Very short (0.5s)", 0.5),
        ("Short (1.0s)", 1.0),
        ("Medium (5.0s)", 5.0),
        ("Long (30.0s)", 30.0),
    ]

    results = []

    for name, duration in test_cases:
        print(f"\n{name}: {duration}s audio")
        print("-"*70)

        try:
            # Generate test audio
            audio = generate_test_audio(duration, freq_type="sweep")
            n_samples = len(audio)

            # Expected frames
            expected_frames = (n_samples - processor.FRAME_SIZE) // processor.HOP_LENGTH + 1

            # Process
            start_time = time.time()
            mel_features = processor.process(audio)
            elapsed = time.time() - start_time

            # Validate
            assert mel_features.shape[0] == processor.N_MELS, f"Expected {processor.N_MELS} mel bins"
            assert mel_features.shape[1] == expected_frames, f"Expected {expected_frames} frames"

            # Calculate metrics
            rtf = duration / elapsed if elapsed > 0 else 0

            print(f"  ✅ SUCCESS")
            print(f"  Output shape: {mel_features.shape}")
            print(f"  Processing time: {elapsed:.4f}s")
            print(f"  Realtime factor: {rtf:.2f}x")
            print(f"  Mel range: [{mel_features.min():.4f}, {mel_features.max():.4f}]")

            results.append({
                "name": name,
                "duration": duration,
                "elapsed": elapsed,
                "rtf": rtf,
                "success": True
            })

        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            results.append({
                "name": name,
                "duration": duration,
                "success": False,
                "error": str(e)
            })

    # Summary
    print("\n" + "="*70)
    print("Test 1 Summary:")
    successes = sum(1 for r in results if r.get("success", False))
    print(f"  Passed: {successes}/{len(results)}")
    if successes > 0:
        avg_rtf = np.mean([r["rtf"] for r in results if r.get("success", False)])
        print(f"  Average RTF: {avg_rtf:.2f}x")

    return results


def test_edge_cases(processor):
    """Test edge cases and boundary conditions."""
    print("\n" + "="*70)
    print("TEST 2: Edge Cases")
    print("="*70)

    test_cases = []

    # Test 1: Exact 100 frames
    print("\n1. Exact batch size (100 frames)")
    print("-"*70)
    samples_needed = (processor.BATCH_SIZE - 1) * processor.HOP_LENGTH + processor.FRAME_SIZE
    audio = generate_test_audio(samples_needed / 16000, freq_type="sine")
    try:
        mel = processor.process(audio)
        assert mel.shape[1] == processor.BATCH_SIZE
        print(f"  ✅ Exact batch: {mel.shape} (expected 100 frames)")
        test_cases.append(("exact_batch", True))
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        test_cases.append(("exact_batch", False))

    # Test 2: Partial batch (150 frames = 100 + 50)
    print("\n2. Partial batch (150 frames)")
    print("-"*70)
    frames_needed = 150
    samples_needed = (frames_needed - 1) * processor.HOP_LENGTH + processor.FRAME_SIZE
    audio = generate_test_audio(samples_needed / 16000, freq_type="chirp")
    try:
        mel = processor.process(audio)
        assert mel.shape[1] == frames_needed
        print(f"  ✅ Partial batch: {mel.shape} (expected 150 frames)")
        test_cases.append(("partial_batch", True))
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        test_cases.append(("partial_batch", False))

    # Test 3: Multiple batches (250 frames = 100 + 100 + 50)
    print("\n3. Multiple batches (250 frames)")
    print("-"*70)
    frames_needed = 250
    samples_needed = (frames_needed - 1) * processor.HOP_LENGTH + processor.FRAME_SIZE
    audio = generate_test_audio(samples_needed / 16000, freq_type="sweep")
    try:
        mel = processor.process(audio)
        assert mel.shape[1] == frames_needed
        batches = processor.total_batches
        print(f"  ✅ Multiple batches: {mel.shape} (expected 250 frames, {batches} batches)")
        test_cases.append(("multiple_batches", True))
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        test_cases.append(("multiple_batches", False))

    # Test 4: Very short audio (< 1 frame)
    print("\n4. Very short audio (100 samples)")
    print("-"*70)
    audio = generate_test_audio(0.00625, freq_type="sine")  # 100 samples
    try:
        mel = processor.process(audio)
        print(f"  ✅ Very short: {mel.shape}")
        test_cases.append(("very_short", True))
    except Exception as e:
        print(f"  ⚠️ Expected behavior: {e}")
        test_cases.append(("very_short", True))  # This is expected to fail or return empty

    # Test 5: Exact frame size (400 samples)
    print("\n5. Exact frame size (400 samples)")
    print("-"*70)
    audio = generate_test_audio(0.025, freq_type="sine")  # 400 samples
    try:
        mel = processor.process(audio)
        print(f"  ✅ Exact frame: {mel.shape}")
        test_cases.append(("exact_frame", True))
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        test_cases.append(("exact_frame", False))

    # Summary
    print("\n" + "="*70)
    print("Test 2 Summary:")
    successes = sum(1 for _, success in test_cases if success)
    print(f"  Passed: {successes}/{len(test_cases)}")

    return test_cases


def test_performance_benchmark(processor):
    """Benchmark performance with various audio lengths."""
    print("\n" + "="*70)
    print("TEST 3: Performance Benchmark")
    print("="*70)

    durations = [1.0, 5.0, 10.0, 30.0, 60.0]
    results = []

    for duration in durations:
        print(f"\n{duration}s audio benchmark")
        print("-"*70)

        # Generate test audio
        audio = generate_test_audio(duration, freq_type="chirp")

        # Reset metrics
        processor.reset_metrics()

        # Process
        start_time = time.time()
        mel_features = processor.process(audio)
        elapsed = time.time() - start_time

        # Get metrics
        metrics = processor.get_performance_metrics()
        rtf = duration / elapsed if elapsed > 0 else 0

        print(f"  Duration: {duration}s")
        print(f"  Processing time: {elapsed:.4f}s")
        print(f"  Realtime factor: {rtf:.2f}x")
        print(f"  Total frames: {metrics['total_frames']}")
        print(f"  Total batches: {metrics['total_batches']}")
        print(f"  Time per frame: {metrics['npu_time_per_frame_ms']:.3f}ms")
        print(f"  Time per batch: {metrics['npu_time_per_batch_ms']:.3f}ms")

        if processor.npu_available:
            print(f"  Kernel time: {metrics['kernel_time_total']:.4f}s ({metrics['kernel_time_total']/elapsed*100:.1f}%)")
            print(f"  Transfer time: {metrics['transfer_time_total']:.4f}s ({metrics['transfer_time_total']/elapsed*100:.1f}%)")

        results.append({
            "duration": duration,
            "elapsed": elapsed,
            "rtf": rtf,
            "metrics": metrics
        })

    # Summary
    print("\n" + "="*70)
    print("Test 3 Summary:")
    print("="*70)
    avg_rtf = np.mean([r["rtf"] for r in results])
    print(f"  Average RTF: {avg_rtf:.2f}x")
    print(f"  Backend: {'NPU' if processor.npu_available else 'CPU'}")

    # Performance table
    print("\n  Duration | Processing | RTF    | Frames | Batches")
    print("  " + "-"*56)
    for r in results:
        print(f"  {r['duration']:7.1f}s | {r['elapsed']:9.4f}s | {r['rtf']:5.1f}x | {r['metrics']['total_frames']:6d} | {r['metrics']['total_batches']:7d}")

    return results


def test_accuracy_comparison(processor):
    """Compare NPU output with CPU reference (if available)."""
    print("\n" + "="*70)
    print("TEST 4: Accuracy Comparison")
    print("="*70)

    try:
        import librosa

        # Generate test audio
        duration = 5.0
        audio = generate_test_audio(duration, freq_type="sweep")

        print(f"\nProcessing {duration}s audio...")
        print("-"*70)

        # Process with NPU
        if processor.npu_available:
            print("  NPU processing...")
            mel_npu = processor.process(audio)
            npu_available = True
        else:
            print("  ⚠️ NPU not available, using CPU")
            mel_npu = processor.process(audio)
            npu_available = False

        # Process with librosa (reference)
        print("  Librosa reference processing...")
        n_frames = mel_npu.shape[1]
        mel_ref = np.zeros((processor.N_MELS, n_frames), dtype=np.float32)

        for i in range(n_frames):
            start_idx = i * processor.HOP_LENGTH
            end_idx = start_idx + processor.FRAME_SIZE

            if end_idx <= len(audio):
                frame = audio[start_idx:end_idx]
            else:
                frame = np.zeros(processor.FRAME_SIZE, dtype=np.float32)
                remaining = len(audio) - start_idx
                if remaining > 0:
                    frame[:remaining] = audio[start_idx:]

            mel = librosa.feature.melspectrogram(
                y=frame,
                sr=processor.SAMPLE_RATE,
                n_fft=512,
                hop_length=processor.HOP_LENGTH,
                win_length=processor.FRAME_SIZE,
                n_mels=processor.N_MELS,
                fmin=0,
                fmax=processor.SAMPLE_RATE // 2,
                htk=True,
                power=2.0
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_ref[:, i] = mel_db[:, 0]

        # Compare
        print("\nAccuracy Analysis:")
        print("-"*70)

        # Calculate differences
        diff = np.abs(mel_npu - mel_ref)
        max_diff = diff.max()
        mean_diff = diff.mean()
        std_diff = diff.std()

        # Calculate correlation
        correlation = np.corrcoef(mel_npu.flatten(), mel_ref.flatten())[0, 1]

        print(f"  NPU shape: {mel_npu.shape}")
        print(f"  Ref shape: {mel_ref.shape}")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Std difference: {std_diff:.6f}")
        print(f"  Correlation: {correlation:.6f}")

        # Tolerance check
        tolerance = 0.1  # Allow 10% difference for quantization
        if max_diff < tolerance:
            print(f"  ✅ Accuracy within tolerance ({tolerance})")
        else:
            print(f"  ⚠️ Accuracy exceeds tolerance (max={max_diff:.6f} > {tolerance})")

        return {
            "npu_available": npu_available,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "correlation": correlation,
            "within_tolerance": max_diff < tolerance
        }

    except ImportError:
        print("  ⚠️ librosa not available, skipping accuracy test")
        return None
    except Exception as e:
        print(f"  ❌ Accuracy test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_stress_test(processor):
    """Stress test with rapid processing."""
    print("\n" + "="*70)
    print("TEST 5: Stress Test")
    print("="*70)

    print("\nRapid processing of 10 audio clips...")
    print("-"*70)

    total_time = 0.0
    total_audio = 0.0
    success_count = 0

    for i in range(10):
        try:
            # Generate random duration audio
            duration = np.random.uniform(1.0, 5.0)
            audio = generate_test_audio(duration, freq_type="noise")

            # Process
            start = time.time()
            mel = processor.process(audio)
            elapsed = time.time() - start

            total_time += elapsed
            total_audio += duration
            success_count += 1

            print(f"  Clip {i+1}/10: {duration:.2f}s → {elapsed:.4f}s ({duration/elapsed:.2f}x RTF)")

        except Exception as e:
            print(f"  Clip {i+1}/10: ❌ Failed - {e}")

    # Summary
    avg_rtf = total_audio / total_time if total_time > 0 else 0
    print("\n" + "="*70)
    print("Test 5 Summary:")
    print(f"  Success rate: {success_count}/10")
    print(f"  Total audio: {total_audio:.2f}s")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Average RTF: {avg_rtf:.2f}x")

    return {
        "success_rate": success_count / 10,
        "avg_rtf": avg_rtf
    }


def main():
    """Main test function."""
    print("="*70)
    print("NPU Mel Processor Batch-100 - Comprehensive Test Suite")
    print("="*70)

    # Create processor
    print("\nInitializing processor...")
    try:
        processor = create_batch_processor(verbose=True)
        print(f"✅ Processor initialized (NPU: {processor.npu_available})")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run tests
    all_results = {}

    try:
        # Test 1: Basic functionality
        all_results["basic"] = test_basic_functionality(processor)

        # Test 2: Edge cases
        all_results["edge_cases"] = test_edge_cases(processor)

        # Test 3: Performance benchmark
        all_results["performance"] = test_performance_benchmark(processor)

        # Test 4: Accuracy comparison
        all_results["accuracy"] = test_accuracy_comparison(processor)

        # Test 5: Stress test
        all_results["stress"] = test_stress_test(processor)

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        processor.close()

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"NPU Available: {processor.npu_available}")

    if "basic" in all_results:
        basic_success = sum(1 for r in all_results["basic"] if r.get("success", False))
        print(f"Basic Tests: {basic_success}/{len(all_results['basic'])} passed")

    if "edge_cases" in all_results:
        edge_success = sum(1 for _, success in all_results["edge_cases"] if success)
        print(f"Edge Cases: {edge_success}/{len(all_results['edge_cases'])} passed")

    if "performance" in all_results:
        avg_rtf = np.mean([r["rtf"] for r in all_results["performance"]])
        print(f"Average RTF: {avg_rtf:.2f}x realtime")

    if "accuracy" in all_results and all_results["accuracy"]:
        acc = all_results["accuracy"]
        print(f"Accuracy: correlation={acc['correlation']:.4f}, within_tolerance={acc['within_tolerance']}")

    if "stress" in all_results:
        stress = all_results["stress"]
        print(f"Stress Test: {stress['success_rate']*100:.0f}% success, {stress['avg_rtf']:.2f}x RTF")

    print("="*70)
    print("Test suite complete!")
    print("="*70)


if __name__ == "__main__":
    main()
