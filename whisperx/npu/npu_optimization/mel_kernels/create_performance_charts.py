#!/usr/bin/env python3
"""
Create Performance Comparison Charts

Generates visualizations from benchmark results:
- Processing time comparison (bar chart)
- Timing distribution (violin plot)
- Throughput comparison
- Overhead analysis
- Timing variance over iterations

Author: Magic Unicorn Inc. - Performance Metrics Lead
Date: October 28, 2025
"""

import json
import numpy as np
from pathlib import Path
import sys

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("❌ matplotlib not available - install with: pip install matplotlib")
    sys.exit(1)


def load_results(results_file: Path):
    """Load benchmark results from JSON

    Args:
        results_file: Path to results JSON file

    Returns:
        results: Dictionary of benchmark results
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results


def create_processing_time_chart(results, output_dir: Path):
    """Create processing time comparison chart

    Args:
        results: Benchmark results dictionary
        output_dir: Output directory
    """
    simple = results['simple']
    optimized = results['optimized']

    fig, ax = plt.subplots(figsize=(10, 6))

    kernels = ['Simple', 'Optimized']
    mean_times = [simple['mean_time_us'], optimized['mean_time_us']]
    std_times = [simple['std_time_us'], optimized['std_time_us']]

    colors = ['#4CAF50', '#2196F3']
    bars = ax.bar(kernels, mean_times, yerr=std_times, capsize=10, color=colors, alpha=0.8)

    # Add value labels on bars
    for bar, mean, std in zip(bars, mean_times, std_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f} µs\n± {std:.2f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Processing Time (microseconds)', fontsize=12)
    ax.set_title('NPU Kernel Processing Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'processing_time_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Created: {output_file}")


def create_timing_distribution_chart(results, output_dir: Path):
    """Create timing distribution chart (violin plot)

    Args:
        results: Benchmark results dictionary
        output_dir: Output directory
    """
    simple = results['simple']
    optimized = results['optimized']

    fig, ax = plt.subplots(figsize=(10, 6))

    data = [
        np.array(simple['execution_times']) * 1e6,  # Convert to microseconds
        np.array(optimized['execution_times']) * 1e6
    ]

    parts = ax.violinplot(data, positions=[1, 2], widths=0.7,
                          showmeans=True, showmedians=True, showextrema=True)

    # Color the violin plots
    colors = ['#4CAF50', '#2196F3']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Simple', 'Optimized'])
    ax.set_ylabel('Processing Time (microseconds)', fontsize=12)
    ax.set_title('Timing Distribution Across Iterations', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add statistics text
    simple_cv = (simple['std_time_us'] / simple['mean_time_us']) * 100
    optimized_cv = (optimized['std_time_us'] / optimized['mean_time_us']) * 100

    stats_text = (
        f"Simple: {simple['mean_time_us']:.2f} µs ± {simple['std_time_us']:.2f} (CV: {simple_cv:.2f}%)\n"
        f"Optimized: {optimized['mean_time_us']:.2f} µs ± {optimized['std_time_us']:.2f} (CV: {optimized_cv:.2f}%)"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')

    plt.tight_layout()
    output_file = output_dir / 'timing_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Created: {output_file}")


def create_throughput_chart(results, output_dir: Path):
    """Create throughput comparison chart

    Args:
        results: Benchmark results dictionary
        output_dir: Output directory
    """
    simple = results['simple']
    optimized = results['optimized']

    fig, ax = plt.subplots(figsize=(10, 6))

    kernels = ['Simple', 'Optimized']
    throughputs = [simple['frames_per_second'], optimized['frames_per_second']]

    colors = ['#4CAF50', '#2196F3']
    bars = ax.bar(kernels, throughputs, color=colors, alpha=0.8)

    # Add value labels
    for bar, throughput in zip(bars, throughputs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{throughput:,.0f} fps',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Throughput (frames/second)', fontsize=12)
    ax.set_title('NPU Kernel Throughput Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add realtime factor as secondary information
    rtf_text = (
        f"Simple: {simple['realtime_factor']:.1f}x realtime\n"
        f"Optimized: {optimized['realtime_factor']:.1f}x realtime"
    )
    ax.text(0.98, 0.98, rtf_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            fontsize=9, family='monospace')

    plt.tight_layout()
    output_file = output_dir / 'throughput_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Created: {output_file}")


def create_overhead_analysis_chart(results, output_dir: Path):
    """Create overhead analysis chart

    Args:
        results: Benchmark results dictionary
        output_dir: Output directory
    """
    simple = results['simple']
    optimized = results['optimized']

    # Calculate overhead
    time_overhead_pct = ((optimized['mean_time_us'] - simple['mean_time_us']) / simple['mean_time_us']) * 100
    size_ratio = optimized['xclbin_size_bytes'] / simple['xclbin_size_bytes']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Processing time overhead
    ax1.bar(['Overhead'], [time_overhead_pct], color='#FF9800' if time_overhead_pct > 0 else '#4CAF50', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_ylabel('Overhead (%)', fontsize=12)
    ax1.set_title('Processing Time Overhead', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value label
    ax1.text(0, time_overhead_pct, f'{time_overhead_pct:+.1f}%',
             ha='center', va='bottom' if time_overhead_pct > 0 else 'top', fontsize=12, fontweight='bold')

    # Add interpretation
    if abs(time_overhead_pct) < 5:
        interpretation = "Negligible difference"
        color = '#4CAF50'
    elif abs(time_overhead_pct) < 15:
        interpretation = "Acceptable overhead" if time_overhead_pct > 0 else "Minor speedup"
        color = '#FFC107'
    else:
        interpretation = "Significant overhead" if time_overhead_pct > 0 else "Major speedup"
        color = '#FF5722' if time_overhead_pct > 0 else '#4CAF50'

    ax1.text(0.5, 0.02, interpretation, transform=ax1.transAxes,
             ha='center', fontsize=10, color=color, fontweight='bold')

    # XCLBIN size comparison
    sizes = [simple['xclbin_size_bytes'] / 1024, optimized['xclbin_size_bytes'] / 1024]
    kernels = ['Simple', 'Optimized']
    colors = ['#4CAF50', '#2196F3']

    bars = ax2.bar(kernels, sizes, color=colors, alpha=0.8)

    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.2f} KB',
                ha='center', va='bottom', fontsize=10)

    ax2.set_ylabel('XCLBIN Size (KB)', fontsize=12)
    ax2.set_title('XCLBIN Size Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add size ratio
    ax2.text(0.5, 0.98, f'Ratio: {size_ratio:.2f}x', transform=ax2.transAxes,
             ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_file = output_dir / 'overhead_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Created: {output_file}")


def create_timing_trace_chart(results, output_dir: Path):
    """Create timing trace over iterations

    Args:
        results: Benchmark results dictionary
        output_dir: Output directory
    """
    simple = results['simple']
    optimized = results['optimized']

    fig, ax = plt.subplots(figsize=(12, 6))

    iterations_simple = range(1, len(simple['execution_times']) + 1)
    iterations_optimized = range(1, len(optimized['execution_times']) + 1)

    times_simple = np.array(simple['execution_times']) * 1e6  # Convert to µs
    times_optimized = np.array(optimized['execution_times']) * 1e6

    ax.plot(iterations_simple, times_simple, label='Simple', color='#4CAF50', alpha=0.6, linewidth=0.8)
    ax.plot(iterations_optimized, times_optimized, label='Optimized', color='#2196F3', alpha=0.6, linewidth=0.8)

    # Add mean lines
    ax.axhline(y=simple['mean_time_us'], color='#4CAF50', linestyle='--', linewidth=2, label='Simple Mean')
    ax.axhline(y=optimized['mean_time_us'], color='#2196F3', linestyle='--', linewidth=2, label='Optimized Mean')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Processing Time (microseconds)', fontsize=12)
    ax.set_title('Timing Stability Over Iterations', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'timing_trace.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Created: {output_file}")


def create_summary_dashboard(results, output_dir: Path):
    """Create comprehensive summary dashboard

    Args:
        results: Benchmark results dictionary
        output_dir: Output directory
    """
    simple = results['simple']
    optimized = results['optimized']

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Processing Time Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    kernels = ['Simple', 'Optimized']
    mean_times = [simple['mean_time_us'], optimized['mean_time_us']]
    colors = ['#4CAF50', '#2196F3']
    ax1.bar(kernels, mean_times, color=colors, alpha=0.8)
    ax1.set_ylabel('Time (µs)')
    ax1.set_title('Processing Time', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add labels
    for i, (kernel, time) in enumerate(zip(kernels, mean_times)):
        ax1.text(i, time, f'{time:.1f}', ha='center', va='bottom', fontsize=9)

    # 2. Throughput
    ax2 = fig.add_subplot(gs[0, 1])
    throughputs = [simple['frames_per_second'], optimized['frames_per_second']]
    ax2.bar(kernels, throughputs, color=colors, alpha=0.8)
    ax2.set_ylabel('Frames/sec')
    ax2.set_title('Throughput', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for i, (kernel, tp) in enumerate(zip(kernels, throughputs)):
        ax2.text(i, tp, f'{tp:.0f}', ha='center', va='bottom', fontsize=9)

    # 3. Realtime Factor
    ax3 = fig.add_subplot(gs[0, 2])
    rtfs = [simple['realtime_factor'], optimized['realtime_factor']]
    ax3.bar(kernels, rtfs, color=colors, alpha=0.8)
    ax3.set_ylabel('Realtime Factor')
    ax3.set_title('Realtime Performance', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    for i, (kernel, rtf) in enumerate(zip(kernels, rtfs)):
        ax3.text(i, rtf, f'{rtf:.1f}x', ha='center', va='bottom', fontsize=9)

    # 4. Timing Distribution
    ax4 = fig.add_subplot(gs[1, :])
    data = [
        np.array(simple['execution_times']) * 1e6,
        np.array(optimized['execution_times']) * 1e6
    ]
    parts = ax4.violinplot(data, positions=[1, 2], widths=0.5,
                           showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax4.set_xticks([1, 2])
    ax4.set_xticklabels(kernels)
    ax4.set_ylabel('Time (µs)')
    ax4.set_title('Timing Distribution', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # 5. Overhead Analysis
    ax5 = fig.add_subplot(gs[2, 0])
    time_overhead = ((optimized['mean_time_us'] - simple['mean_time_us']) / simple['mean_time_us']) * 100
    overhead_color = '#FF9800' if time_overhead > 0 else '#4CAF50'
    ax5.bar(['Time'], [time_overhead], color=overhead_color, alpha=0.8)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax5.set_ylabel('Overhead (%)')
    ax5.set_title('Optimized Overhead', fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    ax5.text(0, time_overhead, f'{time_overhead:+.1f}%', ha='center',
             va='bottom' if time_overhead > 0 else 'top', fontsize=10, fontweight='bold')

    # 6. XCLBIN Size
    ax6 = fig.add_subplot(gs[2, 1])
    sizes = [simple['xclbin_size_bytes'] / 1024, optimized['xclbin_size_bytes'] / 1024]
    ax6.bar(kernels, sizes, color=colors, alpha=0.8)
    ax6.set_ylabel('Size (KB)')
    ax6.set_title('XCLBIN Size', fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)

    for i, (kernel, size) in enumerate(zip(kernels, sizes)):
        ax6.text(i, size, f'{size:.1f}', ha='center', va='bottom', fontsize=9)

    # 7. Statistics Table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    stats_text = (
        "PERFORMANCE SUMMARY\n"
        "─" * 30 + "\n"
        f"Simple Kernel:\n"
        f"  Mean: {simple['mean_time_us']:.2f} µs\n"
        f"  StdDev: {simple['std_time_us']:.2f} µs\n"
        f"  Throughput: {simple['frames_per_second']:,.0f} fps\n"
        f"  RTF: {simple['realtime_factor']:.1f}x\n"
        f"\n"
        f"Optimized Kernel:\n"
        f"  Mean: {optimized['mean_time_us']:.2f} µs\n"
        f"  StdDev: {optimized['std_time_us']:.2f} µs\n"
        f"  Throughput: {optimized['frames_per_second']:,.0f} fps\n"
        f"  RTF: {optimized['realtime_factor']:.1f}x\n"
        f"\n"
        f"Overhead: {time_overhead:+.1f}%\n"
    )

    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes,
             verticalalignment='top', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Overall title
    fig.suptitle('NPU Mel Kernel Performance Dashboard', fontsize=16, fontweight='bold')

    plt.savefig(output_dir / 'performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Created: performance_dashboard.png")


def main():
    """Main chart generation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate performance comparison charts"
    )
    parser.add_argument(
        '--results',
        default='benchmark_results/performance_benchmarks.json',
        help='Path to benchmark results JSON'
    )
    parser.add_argument(
        '--output-dir',
        default='benchmark_results/charts',
        help='Output directory for charts'
    )

    args = parser.parse_args()

    results_file = Path(args.results)
    output_dir = Path(args.output_dir)

    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("Run benchmark first: python3 benchmark_performance.py")
        sys.exit(1)

    print("=" * 70)
    print("CREATING PERFORMANCE CHARTS")
    print("=" * 70)
    print()

    # Load results
    results = load_results(results_file)
    print(f"✅ Loaded results from: {results_file}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all charts
    print("Generating charts...")
    create_processing_time_chart(results, output_dir)
    create_timing_distribution_chart(results, output_dir)
    create_throughput_chart(results, output_dir)
    create_overhead_analysis_chart(results, output_dir)
    create_timing_trace_chart(results, output_dir)
    create_summary_dashboard(results, output_dir)

    print()
    print("=" * 70)
    print("CHARTS COMPLETE!")
    print("=" * 70)
    print()
    print(f"Charts saved to: {output_dir}")
    print()
    print("Generated files:")
    for chart_file in sorted(output_dir.glob("*.png")):
        print(f"  - {chart_file.name}")
    print()


if __name__ == '__main__':
    main()
