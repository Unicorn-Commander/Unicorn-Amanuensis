#!/usr/bin/env python3
"""
Visual comparison of NPU vs CPU mel spectrograms

Generates side-by-side comparison plots and difference maps.

Author: Magic Unicorn Inc.
Date: October 28, 2025
"""

import numpy as np
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - install with: pip install matplotlib")


def plot_mel_comparison(npu_mel, cpu_mel, test_name, metrics, save_path):
    """Create side-by-side mel spectrogram comparison

    Args:
        npu_mel: NPU mel bins (80 values)
        cpu_mel: CPU mel bins (80 values)
        test_name: Name of test
        metrics: Accuracy metrics dictionary
        save_path: Path to save plot
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Mel Spectrogram Comparison: {test_name}', fontsize=16, fontweight='bold')

    # Plot 1: NPU mel spectrogram
    ax1 = axes[0, 0]
    ax1.plot(npu_mel, 'b-', linewidth=2, label='NPU')
    ax1.set_title('NPU Mel Spectrogram (Fixed-Point FFT)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Mel Bin')
    ax1.set_ylabel('Energy (INT8: 0-127)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 127])

    # Plot 2: CPU mel spectrogram
    ax2 = axes[0, 1]
    if cpu_mel is not None:
        ax2.plot(cpu_mel, 'r-', linewidth=2, label='CPU (librosa)')
        ax2.set_title('CPU Mel Spectrogram (Reference)', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'CPU reference not available',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('CPU Mel Spectrogram (N/A)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Mel Bin')
    ax2.set_ylabel('Energy (Scaled: 0-127)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 127])

    # Plot 3: Overlay comparison
    ax3 = axes[1, 0]
    ax3.plot(npu_mel, 'b-', linewidth=2, label='NPU', alpha=0.7)
    if cpu_mel is not None:
        ax3.plot(cpu_mel, 'r--', linewidth=2, label='CPU', alpha=0.7)
    ax3.set_title('Overlay Comparison', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Mel Bin')
    ax3.set_ylabel('Energy')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 127])

    # Plot 4: Difference map and metrics
    ax4 = axes[1, 1]
    if cpu_mel is not None and metrics:
        difference = np.array(npu_mel) - np.array(cpu_mel)
        ax4.bar(range(80), difference, color='purple', alpha=0.6)
        ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax4.set_title('Difference (NPU - CPU)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Mel Bin')
        ax4.set_ylabel('Error')
        ax4.grid(True, alpha=0.3)

        # Add metrics text
        metrics_text = f"""Accuracy Metrics:
Correlation: {metrics.get('correlation', 0)*100:.2f}%
MSE: {metrics.get('mse', 0):.4f}
MAE: {metrics.get('mae', 0):.4f}
RMSE: {metrics.get('rmse', 0):.4f}
SNR: {metrics.get('snr_db', 0):.1f} dB"""

        ax4.text(0.98, 0.98, metrics_text,
                transform=ax4.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax4.text(0.5, 0.5, 'No CPU reference for comparison',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Difference Map (N/A)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Saved: {save_path}")


def plot_aggregate_comparison(all_results, save_path):
    """Create aggregate comparison across all tests

    Args:
        all_results: List of all test results
        save_path: Path to save plot
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NPU Mel Spectrogram: Aggregate Accuracy Analysis',
                 fontsize=16, fontweight='bold')

    # Extract metrics
    test_names = []
    correlations = []
    mses = []
    maes = []
    snrs = []

    for result in all_results:
        if 'metrics' in result and result['metrics']:
            test_names.append(result['test_name'])
            correlations.append(result['metrics'].get('correlation', 0) * 100)
            mses.append(result['metrics'].get('mse', 0))
            maes.append(result['metrics'].get('mae', 0))
            snr = result['metrics'].get('snr_db', 0)
            snrs.append(snr if snr != float('inf') else 100)

    if not test_names:
        print("⚠️  No metrics to plot")
        return

    # Plot 1: Correlation per test
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(test_names)), correlations, color='skyblue', alpha=0.7)
    ax1.axhline(y=99, color='g', linestyle='--', linewidth=1, label='Excellent (99%)')
    ax1.axhline(y=95, color='orange', linestyle='--', linewidth=1, label='Good (95%)')
    ax1.set_title('Correlation by Test', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Correlation (%)')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    ax1.set_xticks([])

    # Color bars based on quality
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        if corr >= 99:
            bar.set_color('green')
        elif corr >= 95:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    # Plot 2: MSE per test
    ax2 = axes[0, 1]
    ax2.bar(range(len(test_names)), mses, color='salmon', alpha=0.7)
    ax2.axhline(y=0.01, color='g', linestyle='--', linewidth=1, label='Excellent (0.01)')
    ax2.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, label='Good (0.1)')
    ax2.set_title('Mean Squared Error by Test', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MSE')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    ax2.set_xticks([])

    # Plot 3: SNR per test
    ax3 = axes[1, 0]
    ax3.bar(range(len(test_names)), snrs, color='lightgreen', alpha=0.7)
    ax3.axhline(y=40, color='g', linestyle='--', linewidth=1, label='Excellent (40 dB)')
    ax3.axhline(y=30, color='orange', linestyle='--', linewidth=1, label='Good (30 dB)')
    ax3.set_title('Signal-to-Noise Ratio by Test', fontsize=12, fontweight='bold')
    ax3.set_ylabel('SNR (dB)')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    ax3.set_xticks([])

    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""Summary Statistics ({len(test_names)} tests)

Correlation:
  Mean: {np.mean(correlations):.2f}%
  Min:  {np.min(correlations):.2f}%
  Max:  {np.max(correlations):.2f}%

Mean Squared Error:
  Mean: {np.mean(mses):.4f}
  Min:  {np.min(mses):.4f}
  Max:  {np.max(mses):.4f}

Signal-to-Noise Ratio:
  Mean: {np.mean(snrs):.1f} dB
  Min:  {np.min(snrs):.1f} dB
  Max:  {np.max(snrs):.1f} dB

Mean Absolute Error:
  Mean: {np.mean(maes):.4f}
  Min:  {np.min(maes):.4f}
  Max:  {np.max(maes):.4f}"""

    ax4.text(0.1, 0.9, summary_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Saved: {save_path}")


def generate_visualizations(results_file="benchmark_results/benchmark_results.json",
                           output_dir="benchmark_results/plots"):
    """Generate all visualizations from benchmark results

    Args:
        results_file: Path to benchmark results JSON
        output_dir: Output directory for plots
    """
    print("=" * 70)
    print("Generating Visual Comparisons")
    print("=" * 70)
    print()

    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available - cannot generate plots")
        print("   Install with: pip install matplotlib")
        return

    # Load results
    results_path = Path(results_file)
    if not results_path.exists():
        print(f"❌ Results file not found: {results_file}")
        print("   Run: python3 benchmark_accuracy.py")
        return

    with open(results_path, 'r') as f:
        all_results = json.load(f)

    print(f"Loaded {len(all_results)} test results")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate individual comparisons
    print("Generating individual comparisons:")
    for result in all_results:
        test_name = result['test_name']
        npu_mel = result['npu_mel']
        cpu_mel = result['cpu_mel']
        metrics = result.get('metrics', {})

        plot_path = output_path / f"{test_name}_comparison.png"
        plot_mel_comparison(npu_mel, cpu_mel, test_name, metrics, plot_path)

    print()

    # Generate aggregate comparison
    print("Generating aggregate analysis:")
    aggregate_path = output_path / "aggregate_analysis.png"
    plot_aggregate_comparison(all_results, aggregate_path)

    print()
    print("=" * 70)
    print("Visualization Complete!")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate visual comparisons")
    parser.add_argument('--results', default='benchmark_results/benchmark_results.json',
                       help='Path to benchmark results JSON')
    parser.add_argument('--output-dir', default='benchmark_results/plots',
                       help='Output directory for plots')

    args = parser.parse_args()

    generate_visualizations(args.results, args.output_dir)

    print("Next step:")
    print("  python3 accuracy_report.py")
