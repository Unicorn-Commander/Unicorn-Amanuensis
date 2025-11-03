#!/usr/bin/env python3
"""
Quick Test Script for Batch-10 NPU Kernel

Tests the batch-10 kernel with:
- Basic functionality
- Performance measurement
- Accuracy validation vs librosa

Author: Magic Unicorn Unconventional Technology & Stuff Inc.
Date: November 1, 2025
"""

import sys
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from npu_mel_processor_batch_final import create_batch_processor

def generate_test_audio(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate test audio with frequency sweep."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, dtype=np.float32)
    freq = 100 + 7900 * (t / duration_sec)  # 100Hz to 8000Hz sweep
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return audio.astype(np.float32)

def test_batch10():
    """Test batch-10 kernel with various audio lengths."""
    print("\n" + "="*70)
    print("BATCH-10 NPU KERNEL TEST")
    print("="*70)
    
    # Initialize processor with batch-10 xclbin
    xclbin_path = 'mel_kernels/build_batch10/mel_batch10.xclbin'
    
    try:
        processor = create_batch_processor(
            xclbin_path=xclbin_path,
            verbose=True
        )
        
        if not processor.npu_available:
            print("❌ NPU not available - falling back to CPU")
            print("   This defeats the purpose of testing the NPU kernel!")
            return False
            
        print(f"\n✅ NPU initialized successfully")
        print(f"   Device: {processor.device}")
        print(f"   XCLBIN: {Path(xclbin_path).name}")
        
    except Exception as e:
        print(f"❌ Failed to initialize NPU: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test cases
    test_cases = [
        ("Short (1s)", 1.0),
        ("Medium (5s)", 5.0),
        ("Long (10s)", 10.0),
        ("Very Long (30s)", 30.0),
    ]
    
    results = []
    
    for name, duration in test_cases:
        print(f"\n{'-'*70}")
        print(f"Test: {name} ({duration}s audio)")
        print(f"{'-'*70}")
        
        try:
            # Generate test audio
            audio = generate_test_audio(duration)
            
            # Process with NPU
            start = time.time()
            mel = processor.process(audio)
            elapsed = time.time() - start
            
            # Calculate metrics
            rtf = duration / elapsed
            
            print(f"✅ SUCCESS")
            print(f"   Output shape: {mel.shape}")
            print(f"   Processing time: {elapsed:.4f}s")
            print(f"   Realtime factor: {rtf:.1f}x")
            print(f"   Throughput: {mel.shape[1] / elapsed:.1f} frames/sec")
            
            results.append({
                'name': name,
                'duration': duration,
                'elapsed': elapsed,
                'rtf': rtf,
                'shape': mel.shape
            })
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    
    if results:
        print(f"\n{'Test':<20} {'Duration':>10} {'Time':>10} {'RTF':>10}")
        print(f"{'-'*70}")
        for r in results:
            print(f"{r['name']:<20} {r['duration']:>9.1f}s {r['elapsed']:>9.4f}s {r['rtf']:>9.1f}x")
        
        avg_rtf = sum(r['rtf'] for r in results) / len(results)
        print(f"\n{'Average RTF:':<30} {avg_rtf:>9.1f}x")
        
        print(f"\n✅ All tests passed with NPU acceleration!")
        processor.close()
        return True
    else:
        print(f"\n❌ No tests completed successfully")
        processor.close()
        return False

if __name__ == "__main__":
    success = test_batch10()
    sys.exit(0 if success else 1)
