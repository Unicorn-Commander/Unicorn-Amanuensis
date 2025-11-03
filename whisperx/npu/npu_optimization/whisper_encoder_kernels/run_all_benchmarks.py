#!/usr/bin/env python3
"""
Master Benchmark Script - Run All Benchmarks and Generate Report

Orchestrates the complete benchmark suite:
1. Individual kernel benchmarks
2. End-to-end pipeline benchmarks
3. Accuracy validation
4. Optimization comparison
5. Comprehensive report generation

Usage:
    python3 run_all_benchmarks.py [--quick] [--skip-accuracy] [--output-dir DIR]

Options:
    --quick          Run quick benchmarks (fewer iterations)
    --skip-accuracy  Skip accuracy validation
    --output-dir     Output directory for reports (default: benchmark_results/)
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime
import json

# Add benchmark suite to path
sys.path.insert(0, str(Path(__file__).parent / "benchmark_suite"))

from benchmark_suite import (
    KernelBenchmark,
    PipelineBenchmark,
    AccuracyBenchmark,
    BenchmarkComparison,
    BenchmarkReport
)


def print_banner(text: str):
    """Print a formatted banner"""
    print()
    print("=" * 70)
    print(text.center(70))
    print("=" * 70)
    print()


def main():
    """Main benchmark orchestration"""

    # Parse arguments
    parser = argparse.ArgumentParser(description='NPU Whisper Comprehensive Benchmark Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmarks')
    parser.add_argument('--skip-accuracy', action='store_true', help='Skip accuracy validation')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print_banner("NPU WHISPER COMPREHENSIVE BENCHMARK SUITE")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print(f"Quick mode: {args.quick}")
    print(f"Skip accuracy: {args.skip_accuracy}")
    print()

    # Initialize results dictionary
    all_results = {}

    # ========================================================================
    # Phase 1: Kernel Benchmarks
    # ========================================================================
    print_banner("PHASE 1: KERNEL BENCHMARKS")

    try:
        num_iterations = 20 if args.quick else 100
        kernel_bench = KernelBenchmark(num_iterations=num_iterations)
        kernel_results = kernel_bench.benchmark_all_kernels()
        all_results['kernels'] = kernel_results

        # Save kernel results
        kernel_file = output_dir / f"kernel_results_{timestamp}.json"
        kernel_bench.save_results(str(kernel_file))

        print("✅ Kernel benchmarks complete")
        print()

    except Exception as e:
        print(f"❌ Error in kernel benchmarks: {e}")
        import traceback
        traceback.print_exc()
        kernel_results = {}
        all_results['kernels'] = {}

    # ========================================================================
    # Phase 2: Pipeline Benchmarks
    # ========================================================================
    print_banner("PHASE 2: PIPELINE BENCHMARKS")

    try:
        pipeline_bench = PipelineBenchmark()

        if args.quick:
            # Quick mode: test only 30s audio
            lengths = [30]
        else:
            # Full mode: test multiple lengths
            lengths = [10, 30, 60, 120, 300]

        pipeline_results = pipeline_bench.benchmark_multiple_lengths(lengths)
        all_results['pipeline'] = pipeline_results

        # Save pipeline results
        pipeline_file = output_dir / f"pipeline_results_{timestamp}.json"
        pipeline_bench.save_results(str(pipeline_file))

        # Also benchmark single encoder block
        print()
        print("Benchmarking single encoder block...")
        block_result = pipeline_bench.benchmark_encoder_block(num_iterations=10)
        all_results['encoder_block'] = block_result

        print("✅ Pipeline benchmarks complete")
        print()

    except Exception as e:
        print(f"❌ Error in pipeline benchmarks: {e}")
        import traceback
        traceback.print_exc()
        pipeline_results = []
        all_results['pipeline'] = []

    # ========================================================================
    # Phase 3: Accuracy Validation
    # ========================================================================
    if not args.skip_accuracy:
        print_banner("PHASE 3: ACCURACY VALIDATION")

        try:
            accuracy_bench = AccuracyBenchmark()
            accuracy_results = accuracy_bench.validate_all_kernels()
            all_results['accuracy'] = accuracy_results

            # Save accuracy results
            accuracy_file = output_dir / f"accuracy_results_{timestamp}.json"
            accuracy_bench.save_results(str(accuracy_file))

            print("✅ Accuracy validation complete")
            print()

        except Exception as e:
            print(f"❌ Error in accuracy validation: {e}")
            import traceback
            traceback.print_exc()
            accuracy_results = {}
            all_results['accuracy'] = {}
    else:
        print("⏭️  Skipping accuracy validation")
        print()
        all_results['accuracy'] = {}

    # ========================================================================
    # Phase 4: Optimization Comparison
    # ========================================================================
    print_banner("PHASE 4: OPTIMIZATION COMPARISON")

    try:
        comparison_bench = BenchmarkComparison()
        comparison_results = comparison_bench.compare_optimizations()
        all_results['comparisons'] = comparison_results

        # Save comparison results
        comparison_file = output_dir / f"comparison_results_{timestamp}.json"
        comparison_bench.save_results(str(comparison_file))

        # Also compare tile sizes
        if not args.quick:
            print()
            tile_results = comparison_bench.compare_tile_sizes()
            all_results['tile_comparison'] = tile_results

        print("✅ Optimization comparison complete")
        print()

    except Exception as e:
        print(f"❌ Error in optimization comparison: {e}")
        import traceback
        traceback.print_exc()
        comparison_results = {}
        all_results['comparisons'] = {}

    # ========================================================================
    # Phase 5: Generate Reports
    # ========================================================================
    print_banner("PHASE 5: REPORT GENERATION")

    try:
        report_gen = BenchmarkReport()

        # Generate markdown report
        markdown_file = output_dir / f"BENCHMARK_REPORT_{timestamp}.md"
        report_gen.generate_markdown_report(all_results, str(markdown_file))
        print(f"✅ Markdown report: {markdown_file}")

        # Generate JSON report
        json_file = output_dir / f"benchmark_report_{timestamp}.json"
        report_gen.generate_json_report(all_results, str(json_file))
        print(f"✅ JSON report: {json_file}")

        # Create latest symlinks
        latest_md = output_dir / "BENCHMARK_REPORT_LATEST.md"
        latest_json = output_dir / "benchmark_report_latest.json"

        if latest_md.exists():
            latest_md.unlink()
        if latest_json.exists():
            latest_json.unlink()

        latest_md.symlink_to(markdown_file.name)
        latest_json.symlink_to(json_file.name)

        print(f"✅ Latest report symlink: {latest_md}")
        print()

    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Summary
    # ========================================================================
    print_banner("BENCHMARK SUITE COMPLETE")

    print("📊 Results Summary:")
    print()

    if 'kernels' in all_results and all_results['kernels']:
        print("✅ Kernel Benchmarks:")
        for kernel_name, data in all_results['kernels'].items():
            print(f"   - {data.get('kernel', kernel_name)}: {data.get('mean', 0):.3f}ms (mean)")

    print()

    if 'pipeline' in all_results and all_results['pipeline']:
        print("✅ Pipeline Benchmarks:")
        for result in all_results['pipeline']:
            print(f"   - {result['audio_length']}s audio: {result['realtime_factor']:.2f}x realtime")

    print()

    if 'accuracy' in all_results and all_results['accuracy']:
        print("✅ Accuracy Validation:")
        for kernel_name, data in all_results['accuracy'].items():
            status = "PASS" if data.get('pass', False) else "FAIL"
            print(f"   - {data.get('kernel', kernel_name)}: {status} (corr: {data.get('correlation', 0):.4f})")

    print()

    if 'comparisons' in all_results and all_results['comparisons']:
        print("✅ Optimization Comparison:")
        for config_name, data in all_results['comparisons'].items():
            print(f"   - {config_name}: {data.get('realtime_factor', 0):.2f}x realtime")

    print()
    print("=" * 70)
    print()
    print(f"📁 All results saved to: {output_dir}")
    print(f"📄 Main report: {output_dir / 'BENCHMARK_REPORT_LATEST.md'}")
    print()
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print("❌ Benchmarks interrupted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
