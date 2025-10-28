#!/usr/bin/env python3
"""
NPU Mel Spectrogram Accuracy Benchmarking Suite

Compares NPU fixed-point FFT implementation against librosa reference.
Generates comprehensive accuracy metrics and visual comparisons.

Author: Magic Unicorn Inc.
Date: October 28, 2025
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import numpy as np
import pyxrt as xrt
from pathlib import Path
import json
from datetime import datetime

# Optional imports
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa not available - install with: pip install librosa")

try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available - some metrics will be skipped")


class NPUMelBenchmark:
    """Benchmark NPU mel spectrogram accuracy"""

    def __init__(self, xclbin_path="build_fixed/mel_fixed.xclbin",
                 insts_path="build_fixed/insts_fixed.bin"):
        """Initialize NPU benchmark

        Args:
            xclbin_path: Path to NPU XCLBIN file
            insts_path: Path to instruction binary
        """
        self.xclbin_path = xclbin_path
        self.insts_path = insts_path

        # Initialize NPU
        print("Initializing NPU...")
        self.device = xrt.device(0)
        self.xclbin = xrt.xclbin(xclbin_path)
        self.device.register_xclbin(self.xclbin)

        uuid = self.xclbin.get_uuid()
        self.hw_ctx = xrt.hw_context(self.device, uuid)
        self.kernel = xrt.kernel(self.hw_ctx, "MLIR_AIE")

        # Read instruction binary
        with open(insts_path, 'rb') as f:
            self.insts_bin = f.read()
        self.n_insts = len(self.insts_bin)

        # Allocate buffers
        self.instr_bo = xrt.bo(self.device, self.n_insts,
                               xrt.bo.flags.cacheable, self.kernel.group_id(1))
        self.input_bo = xrt.bo(self.device, 800,
                               xrt.bo.flags.host_only, self.kernel.group_id(3))
        self.output_bo = xrt.bo(self.device, 80,
                                xrt.bo.flags.host_only, self.kernel.group_id(4))

        # Write instructions
        self.instr_bo.write(self.insts_bin, 0)
        self.instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                           self.n_insts, 0)

        print(f"✅ NPU initialized: {xclbin_path}")

    def run_npu(self, audio_int16):
        """Run mel computation on NPU

        Args:
            audio_int16: INT16 audio samples (400 samples)

        Returns:
            mel_bins: INT8 mel spectrogram (80 bins)
        """
        # Convert to bytes (little-endian)
        input_data = audio_int16.astype(np.int16).tobytes()

        # Write to NPU
        self.input_bo.write(input_data, 0)
        self.input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 800, 0)

        # Execute kernel
        opcode = 3
        run = self.kernel(opcode, self.instr_bo, self.n_insts,
                         self.input_bo, self.output_bo)
        state = run.wait(5000)

        if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"NPU kernel failed with state: {state}")

        # Read output
        self.output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 80, 0)
        mel_bins = np.frombuffer(self.output_bo.read(80, 0), dtype=np.int8)

        return mel_bins

    def compute_cpu_mel(self, audio_int16):
        """Compute mel spectrogram on CPU using librosa

        Args:
            audio_int16: INT16 audio samples (400 samples)

        Returns:
            mel_bins: Float mel spectrogram (80 bins), scaled to [0, 127]
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for CPU reference")

        # Convert INT16 to float [-1, 1]
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Pad to 512 samples (match NPU behavior)
        audio_padded = np.pad(audio_float, (0, 512 - len(audio_float)), mode='constant')

        # Compute STFT manually to match NPU
        from scipy.signal import get_window
        window = get_window('hann', 512)
        windowed = audio_padded * window

        # FFT
        fft_result = np.fft.rfft(windowed, n=512)
        magnitude = np.abs(fft_result)

        # Compute mel filterbank
        mel_filters = librosa.filters.mel(
            sr=16000,
            n_fft=512,
            n_mels=80,
            fmin=0,
            fmax=8000,
            htk=True
        )

        # Apply mel filters
        mel_spec = mel_filters @ magnitude

        # Apply log scaling (like Whisper)
        log_mel = np.log10(mel_spec + 1e-10)

        # Normalize to [0, 127] range (match NPU INT8 output)
        mel_min = np.min(log_mel)
        mel_max = np.max(log_mel)

        if mel_max > mel_min:
            mel_normalized = (log_mel - mel_min) / (mel_max - mel_min) * 127
        else:
            mel_normalized = np.zeros(80)

        return mel_normalized

    def compare_mel_spectrograms(self, npu_mel, cpu_mel):
        """Compare NPU and CPU mel spectrograms

        Args:
            npu_mel: NPU mel bins (INT8, 80 bins)
            cpu_mel: CPU mel bins (float, 80 bins, scaled to [0, 127])

        Returns:
            metrics: Dictionary of comparison metrics
        """
        # Convert NPU INT8 to float for comparison
        npu_float = npu_mel.astype(np.float32)

        # Compute metrics
        metrics = {}

        # Mean Squared Error
        mse = np.mean((npu_float - cpu_mel) ** 2)
        metrics['mse'] = float(mse)

        # Mean Absolute Error
        mae = np.mean(np.abs(npu_float - cpu_mel))
        metrics['mae'] = float(mae)

        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        metrics['rmse'] = float(rmse)

        # Pearson Correlation
        if SCIPY_AVAILABLE and len(npu_float) > 1:
            try:
                corr, p_value = pearsonr(npu_float, cpu_mel)
                metrics['correlation'] = float(corr)
                metrics['correlation_pvalue'] = float(p_value)
            except:
                metrics['correlation'] = None
        else:
            # Simple correlation coefficient
            mean_npu = np.mean(npu_float)
            mean_cpu = np.mean(cpu_mel)
            npu_centered = npu_float - mean_npu
            cpu_centered = cpu_mel - mean_cpu

            numerator = np.sum(npu_centered * cpu_centered)
            denominator = np.sqrt(np.sum(npu_centered**2) * np.sum(cpu_centered**2))

            if denominator > 0:
                metrics['correlation'] = float(numerator / denominator)
            else:
                metrics['correlation'] = 0.0

        # Signal-to-Noise Ratio
        # SNR = 10 * log10(signal_power / noise_power)
        signal_power = np.mean(cpu_mel ** 2)
        noise_power = np.mean((npu_float - cpu_mel) ** 2)

        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            metrics['snr_db'] = float(snr)
        else:
            metrics['snr_db'] = float('inf')

        # Per-bin error statistics
        per_bin_error = np.abs(npu_float - cpu_mel)
        metrics['per_bin_error_mean'] = float(np.mean(per_bin_error))
        metrics['per_bin_error_std'] = float(np.std(per_bin_error))
        metrics['per_bin_error_max'] = float(np.max(per_bin_error))
        metrics['per_bin_error_min'] = float(np.min(per_bin_error))

        # Store per-bin errors for analysis
        metrics['per_bin_errors'] = per_bin_error.tolist()

        return metrics

    def run_test_file(self, test_file_path, test_name):
        """Run benchmark on single test file

        Args:
            test_file_path: Path to raw audio file (800 bytes)
            test_name: Name of test for reporting

        Returns:
            results: Dictionary with test results
        """
        print(f"  Testing: {test_name}...", end=' ')

        # Load test audio (800 bytes = 400 INT16 samples)
        with open(test_file_path, 'rb') as f:
            audio_bytes = f.read(800)

        # Convert to INT16
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

        # Run on NPU
        npu_mel = self.run_npu(audio_int16)

        # Run on CPU (if available)
        if LIBROSA_AVAILABLE:
            cpu_mel = self.compute_cpu_mel(audio_int16)

            # Compare
            metrics = self.compare_mel_spectrograms(npu_mel, cpu_mel)
        else:
            cpu_mel = None
            metrics = {}

        # Package results
        results = {
            'test_name': test_name,
            'test_file': str(test_file_path),
            'npu_mel': npu_mel.tolist(),
            'cpu_mel': cpu_mel.tolist() if cpu_mel is not None else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        if LIBROSA_AVAILABLE and metrics:
            corr_pct = metrics['correlation'] * 100
            print(f"Corr: {corr_pct:.2f}%, MSE: {metrics['mse']:.4f}, "
                  f"SNR: {metrics['snr_db']:.1f} dB")
        else:
            print("NPU only (no CPU reference)")

        return results


def run_benchmark_suite(test_audio_dir="test_audio",
                       xclbin_path="build_fixed/mel_fixed.xclbin",
                       output_dir="benchmark_results"):
    """Run complete benchmark suite

    Args:
        test_audio_dir: Directory with test audio files
        xclbin_path: Path to NPU XCLBIN
        output_dir: Output directory for results

    Returns:
        all_results: List of all test results
    """
    print("=" * 70)
    print("NPU Mel Spectrogram Accuracy Benchmark")
    print("=" * 70)
    print()

    # Check for librosa
    if not LIBROSA_AVAILABLE:
        print("⚠️  WARNING: librosa not available")
        print("   NPU outputs will be collected but not compared")
        print()

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Initialize benchmark
    benchmark = NPUMelBenchmark(xclbin_path)

    # Find all test files
    test_files = sorted(Path(test_audio_dir).glob("*.raw"))

    if not test_files:
        print(f"❌ No test files found in {test_audio_dir}")
        print("   Run: python3 generate_test_signals.py")
        return []

    print(f"Found {len(test_files)} test files")
    print()

    # Run tests
    all_results = []
    for test_file in test_files:
        test_name = test_file.stem
        try:
            results = benchmark.run_test_file(test_file, test_name)
            all_results.append(results)
        except Exception as e:
            print(f"❌ FAILED: {e}")
            continue

    print()
    print("=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
    print()

    # Save results
    results_file = Path(output_dir) / "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Results saved: {results_file}")

    # Compute aggregate statistics
    if LIBROSA_AVAILABLE and all_results:
        print()
        print("Aggregate Statistics:")
        print("-" * 70)

        correlations = [r['metrics']['correlation'] for r in all_results
                       if 'correlation' in r['metrics']]
        mses = [r['metrics']['mse'] for r in all_results
               if 'mse' in r['metrics']]
        snrs = [r['metrics']['snr_db'] for r in all_results
               if 'snr_db' in r['metrics'] and r['metrics']['snr_db'] != float('inf')]

        if correlations:
            print(f"  Correlation:  Mean={np.mean(correlations)*100:.2f}%, "
                  f"Min={np.min(correlations)*100:.2f}%, "
                  f"Max={np.max(correlations)*100:.2f}%")
        if mses:
            print(f"  MSE:          Mean={np.mean(mses):.4f}, "
                  f"Min={np.min(mses):.4f}, Max={np.max(mses):.4f}")
        if snrs:
            print(f"  SNR (dB):     Mean={np.mean(snrs):.1f}, "
                  f"Min={np.min(snrs):.1f}, Max={np.max(snrs):.1f}")

        # Overall verdict
        print()
        print("Verdict:")
        print("-" * 70)
        avg_corr = np.mean(correlations) if correlations else 0
        avg_mse = np.mean(mses) if mses else float('inf')

        if avg_corr > 0.99 and avg_mse < 0.01:
            verdict = "EXCELLENT (>99% correlation, <0.01 MSE)"
            status = "✅ PASS"
        elif avg_corr > 0.95 and avg_mse < 0.1:
            verdict = "GOOD (>95% correlation, <0.1 MSE)"
            status = "✅ PASS"
        elif avg_corr > 0.90:
            verdict = "ACCEPTABLE (>90% correlation)"
            status = "⚠️  MARGINAL"
        else:
            verdict = f"NEEDS IMPROVEMENT ({avg_corr*100:.1f}% correlation)"
            status = "❌ FAIL"

        print(f"  {status}: {verdict}")

    print()
    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark NPU mel spectrogram accuracy")
    parser.add_argument('--test-dir', default='test_audio',
                       help='Directory with test audio files')
    parser.add_argument('--xclbin', default='build_fixed/mel_fixed.xclbin',
                       help='Path to NPU XCLBIN file')
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark_suite(
        test_audio_dir=args.test_dir,
        xclbin_path=args.xclbin,
        output_dir=args.output_dir
    )

    if results:
        print()
        print("✨ Benchmark complete!")
        print()
        print("Next steps:")
        print("  1. Review: benchmark_results/benchmark_results.json")
        print("  2. Generate visuals: python3 visual_comparison.py")
        print("  3. Create report: python3 accuracy_report.py")
    else:
        print()
        print("❌ No results generated")
        sys.exit(1)
