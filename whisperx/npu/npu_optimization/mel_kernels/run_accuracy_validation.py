#!/usr/bin/env python3
"""
Comprehensive Accuracy Validation for Simple vs Optimized Mel Kernels

Runs both kernels, compares results, and generates comprehensive report.
Team 1: Accuracy Benchmarking Lead

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

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("❌ librosa not available - cannot compute CPU reference!")
    sys.exit(1)

try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class MelKernelTester:
    """Test mel kernel accuracy"""

    def __init__(self, xclbin_path, insts_path, kernel_name):
        """Initialize kernel tester

        Args:
            xclbin_path: Path to XCLBIN file
            insts_path: Path to instruction binary
            kernel_name: Name for reporting
        """
        self.kernel_name = kernel_name
        self.xclbin_path = xclbin_path
        self.insts_path = insts_path

        print(f"Initializing {kernel_name}...")
        print(f"  XCLBIN: {xclbin_path}")
        print(f"  Insts:  {insts_path}")

        # Initialize NPU
        self.device = xrt.device(0)
        self.xclbin = xrt.xclbin(xclbin_path)
        self.device.register_xclbin(self.xclbin)

        uuid = self.xclbin.get_uuid()
        self.hw_ctx = xrt.hw_context(self.device, uuid)
        self.kernel = xrt.kernel(self.hw_ctx, "MLIR_AIE")

        # Read instructions
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

        print(f"✅ {kernel_name} initialized!")

    def run_npu(self, audio_int16):
        """Run mel computation on NPU"""
        # Convert to bytes
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
            raise RuntimeError(f"Kernel failed with state: {state}")

        # Read output
        self.output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 80, 0)
        mel_bins = np.frombuffer(self.output_bo.read(80, 0), dtype=np.int8)

        return mel_bins


def compute_cpu_mel(audio_int16):
    """Compute mel spectrogram using librosa (CPU reference)"""
    # Convert INT16 to float
    audio_float = audio_int16.astype(np.float32) / 32768.0

    # Pad to 512
    audio_padded = np.pad(audio_float, (0, 512 - len(audio_float)), mode='constant')

    # Window and FFT
    from scipy.signal import get_window
    window = get_window('hann', 512)
    windowed = audio_padded * window
    fft_result = np.fft.rfft(windowed, n=512)
    magnitude = np.abs(fft_result)

    # Mel filterbank
    mel_filters = librosa.filters.mel(
        sr=16000,
        n_fft=512,
        n_mels=80,
        fmin=0,
        fmax=8000,
        htk=True
    )

    # Apply filters
    mel_spec = mel_filters @ magnitude

    # Log scaling
    log_mel = np.log10(mel_spec + 1e-10)

    # Normalize to [0, 127]
    mel_min = np.min(log_mel)
    mel_max = np.max(log_mel)

    if mel_max > mel_min:
        mel_normalized = (log_mel - mel_min) / (mel_max - mel_min) * 127
    else:
        mel_normalized = np.zeros(80)

    return mel_normalized


def compute_metrics(npu_mel, cpu_mel):
    """Compute accuracy metrics"""
    npu_float = npu_mel.astype(np.float32)

    metrics = {}

    # MSE, MAE, RMSE
    mse = np.mean((npu_float - cpu_mel) ** 2)
    mae = np.mean(np.abs(npu_float - cpu_mel))
    rmse = np.sqrt(mse)

    metrics['mse'] = float(mse)
    metrics['mae'] = float(mae)
    metrics['rmse'] = float(rmse)

    # Correlation
    if SCIPY_AVAILABLE:
        try:
            corr, p_value = pearsonr(npu_float, cpu_mel)
            metrics['correlation'] = float(corr)
            metrics['correlation_pvalue'] = float(p_value)
        except:
            metrics['correlation'] = None
    else:
        # Manual correlation
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

    # SNR
    signal_power = np.mean(cpu_mel ** 2)
    noise_power = np.mean((npu_float - cpu_mel) ** 2)

    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
        metrics['snr_db'] = float(snr)
    else:
        metrics['snr_db'] = float('inf')

    # Per-bin errors
    per_bin_error = np.abs(npu_float - cpu_mel)
    metrics['per_bin_errors'] = per_bin_error.tolist()
    metrics['per_bin_error_mean'] = float(np.mean(per_bin_error))
    metrics['per_bin_error_std'] = float(np.std(per_bin_error))
    metrics['per_bin_error_max'] = float(np.max(per_bin_error))

    return metrics


def run_validation():
    """Run complete validation suite"""
    print("=" * 70)
    print("MEL KERNEL ACCURACY VALIDATION")
    print("Simple Kernel vs Optimized Kernel")
    print("=" * 70)
    print()

    # Initialize both kernels
    simple_tester = MelKernelTester(
        xclbin_path="build_fixed/mel_fixed_new.xclbin",
        insts_path="build_fixed/insts_new.bin",
        kernel_name="Simple Kernel"
    )
    print()

    optimized_tester = MelKernelTester(
        xclbin_path="build_optimized/mel_optimized_new.xclbin",
        insts_path="build_optimized/insts_optimized_new.bin",
        kernel_name="Optimized Kernel"
    )
    print()

    # Find test files
    test_files = sorted(Path("test_audio").glob("*.raw"))
    print(f"Found {len(test_files)} test files")
    print()

    # Run tests
    results_simple = []
    results_optimized = []

    for test_file in test_files:
        test_name = test_file.stem
        print(f"Testing: {test_name}")

        # Load audio
        with open(test_file, 'rb') as f:
            audio_bytes = f.read(800)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

        # Run simple kernel
        npu_mel_simple = simple_tester.run_npu(audio_int16)

        # Run optimized kernel
        npu_mel_opt = optimized_tester.run_npu(audio_int16)

        # Compute CPU reference
        cpu_mel = compute_cpu_mel(audio_int16)

        # Compute metrics
        metrics_simple = compute_metrics(npu_mel_simple, cpu_mel)
        metrics_opt = compute_metrics(npu_mel_opt, cpu_mel)

        # Store results
        results_simple.append({
            'test_name': test_name,
            'npu_mel': npu_mel_simple.tolist(),
            'cpu_mel': cpu_mel.tolist(),
            'metrics': metrics_simple
        })

        results_optimized.append({
            'test_name': test_name,
            'npu_mel': npu_mel_opt.tolist(),
            'cpu_mel': cpu_mel.tolist(),
            'metrics': metrics_opt
        })

        # Print quick summary
        corr_s = metrics_simple.get('correlation', 0) * 100 if metrics_simple.get('correlation') else 0
        corr_o = metrics_opt.get('correlation', 0) * 100 if metrics_opt.get('correlation') else 0
        print(f"  Simple: {corr_s:6.2f}% | Optimized: {corr_o:6.2f}%")

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE!")
    print("=" * 70)
    print()

    # Save results
    Path("benchmark_results_simple").mkdir(exist_ok=True)
    Path("benchmark_results_optimized").mkdir(exist_ok=True)

    with open("benchmark_results_simple/benchmark_results.json", 'w') as f:
        json.dump(results_simple, f, indent=2)

    with open("benchmark_results_optimized/benchmark_results.json", 'w') as f:
        json.dump(results_optimized, f, indent=2)

    print("✅ Results saved!")
    print()

    # Compute aggregate statistics
    correlations_s = [r['metrics']['correlation'] for r in results_simple
                     if r['metrics'].get('correlation') is not None]
    correlations_o = [r['metrics']['correlation'] for r in results_optimized
                     if r['metrics'].get('correlation') is not None]

    mses_s = [r['metrics']['mse'] for r in results_simple]
    mses_o = [r['metrics']['mse'] for r in results_optimized]

    print("AGGREGATE STATISTICS:")
    print("=" * 70)
    print()
    print("SIMPLE KERNEL (Linear Downsampling):")
    if correlations_s:
        print(f"  Correlation: {np.mean(correlations_s)*100:.2f}% (min: {np.min(correlations_s)*100:.2f}%, max: {np.max(correlations_s)*100:.2f}%)")
    print(f"  MSE:         {np.mean(mses_s):.4f}")
    print()

    print("OPTIMIZED KERNEL (Triangular Mel Filters):")
    if correlations_o:
        print(f"  Correlation: {np.mean(correlations_o)*100:.2f}% (min: {np.min(correlations_o)*100:.2f}%, max: {np.max(correlations_o)*100:.2f}%)")
    print(f"  MSE:         {np.mean(mses_o):.4f}")
    print()

    # Verdict
    avg_corr_s = np.mean(correlations_s) if correlations_s else 0
    avg_corr_o = np.mean(correlations_o) if correlations_o else 0

    print("VERDICT:")
    print("=" * 70)
    if avg_corr_o > 0.95:
        print(f"✅ PASS: Optimized kernel achieves {avg_corr_o*100:.2f}% correlation (target: >95%)")
    else:
        print(f"❌ FAIL: Optimized kernel achieves {avg_corr_o*100:.2f}% correlation (target: >95%)")

    improvement = (avg_corr_o - avg_corr_s) / abs(avg_corr_s) * 100 if avg_corr_s != 0 else 0
    print(f"📈 Improvement: {abs(improvement):.1f}x better correlation")
    print()


if __name__ == '__main__':
    run_validation()
