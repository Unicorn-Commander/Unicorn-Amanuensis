#!/usr/bin/env python3
"""
Compare Whisper Weight Formats (FP32 vs FP16 vs INT8)

Comprehensive analysis of different weight quantization strategies:
- FP32: Full precision (baseline)
- FP16: Half precision (2x compression)
- INT8: 8-bit quantization (4x compression)

Evaluates:
- Memory usage
- Quantization error
- Value distribution
- Overflow risks
- Recommendations
"""

import numpy as np
from pathlib import Path
import sys

class WeightFormatComparison:
    """Compare different weight quantization formats."""

    def __init__(self, fp32_dir, fp16_dir, int8_dir):
        self.fp32_dir = Path(fp32_dir)
        self.fp16_dir = Path(fp16_dir)
        self.int8_dir = Path(int8_dir)

        self.fp32_files = sorted(self.fp32_dir.glob("*.npy"))
        self.fp16_files = sorted(self.fp16_dir.glob("*_fp16.npy"))
        self.int8_files = sorted(self.int8_dir.glob("*_int8.npy"))

    def compare_single_weight(self, weight_name):
        """
        Compare a single weight across all formats.

        Args:
            weight_name: Base name of weight (e.g., "embed_positions_weight")

        Returns:
            dict: Comparison statistics
        """
        fp32_path = self.fp32_dir / f"{weight_name}.npy"
        fp16_path = self.fp16_dir / f"{weight_name}_fp16.npy"
        int8_path = self.int8_dir / f"{weight_name}_int8.npy"
        scale_path = self.int8_dir / f"{weight_name}_scale.npy"

        if not fp32_path.exists():
            return {'error': f"FP32 not found: {fp32_path}"}

        # Load FP32 (ground truth)
        w_fp32 = np.load(fp32_path)

        # Load FP16 if available
        if fp16_path.exists():
            w_fp16 = np.load(fp16_path).astype(np.float32)
            fp16_error = np.abs(w_fp32 - w_fp16)
            fp16_max_error = fp16_error.max()
            fp16_avg_error = fp16_error.mean()
            fp16_rel_error = np.mean(np.abs(fp16_error / (np.abs(w_fp32) + 1e-10)))
            fp16_available = True
        else:
            fp16_max_error = None
            fp16_avg_error = None
            fp16_rel_error = None
            fp16_available = False

        # Load INT8 if available
        if int8_path.exists() and scale_path.exists():
            w_int8 = np.load(int8_path)
            scale = np.load(scale_path)[0]
            w_int8_dequant = w_int8.astype(np.float32) * scale

            int8_error = np.abs(w_fp32 - w_int8_dequant)
            int8_max_error = int8_error.max()
            int8_avg_error = int8_error.mean()
            int8_rel_error = np.mean(np.abs(int8_error / (np.abs(w_fp32) + 1e-10)))
            int8_available = True
        else:
            int8_max_error = None
            int8_avg_error = None
            int8_rel_error = None
            int8_available = False

        # Calculate statistics
        stats = {
            'name': weight_name,
            'shape': w_fp32.shape,
            'num_elements': w_fp32.size,

            # FP32 statistics
            'fp32_min': w_fp32.min(),
            'fp32_max': w_fp32.max(),
            'fp32_mean': w_fp32.mean(),
            'fp32_std': w_fp32.std(),
            'fp32_size_mb': w_fp32.nbytes / 1024 / 1024,

            # FP16 statistics
            'fp16_available': fp16_available,
            'fp16_max_error': fp16_max_error,
            'fp16_avg_error': fp16_avg_error,
            'fp16_rel_error': fp16_rel_error,
            'fp16_size_mb': w_fp32.nbytes / 2 / 1024 / 1024 if fp16_available else None,

            # INT8 statistics
            'int8_available': int8_available,
            'int8_max_error': int8_max_error,
            'int8_avg_error': int8_avg_error,
            'int8_rel_error': int8_rel_error,
            'int8_size_mb': w_fp32.nbytes / 4 / 1024 / 1024 if int8_available else None,
        }

        return stats

    def compare_all_weights(self):
        """
        Compare all weights across formats.

        Returns:
            dict: Overall comparison statistics
        """
        print("=" * 80)
        print("  WEIGHT FORMAT COMPARISON")
        print("=" * 80)

        print(f"\nDirectories:")
        print(f"  FP32:  {self.fp32_dir} ({len(self.fp32_files)} files)")
        print(f"  FP16:  {self.fp16_dir} ({len(self.fp16_files)} files)")
        print(f"  INT8:  {self.int8_dir} ({len(self.int8_files)} files)")

        # Compare each weight
        results = []

        print("\n" + "=" * 80)
        print("  COMPARING WEIGHTS")
        print("=" * 80 + "\n")

        for fp32_file in self.fp32_files:
            weight_name = fp32_file.stem

            stats = self.compare_single_weight(weight_name)

            if 'error' in stats:
                print(f"❌ {weight_name:50s} {stats['error']}")
            else:
                results.append(stats)

                # Build status string
                fp16_status = "✅" if stats['fp16_available'] else "❌"
                int8_status = "✅" if stats['int8_available'] else "❌"

                print(f"{weight_name:50s} FP16:{fp16_status} INT8:{int8_status}")

        # Calculate overall statistics
        overall = self._calculate_overall_stats(results)

        # Print detailed comparison
        self._print_comparison_tables(results, overall)

        return overall

    def _calculate_overall_stats(self, results):
        """Calculate overall statistics from individual results."""
        fp16_results = [r for r in results if r['fp16_available']]
        int8_results = [r for r in results if r['int8_available']]

        return {
            'total_weights': len(results),
            'fp16_available': len(fp16_results),
            'int8_available': len(int8_results),

            # FP32 totals
            'fp32_total_mb': sum(r['fp32_size_mb'] for r in results),

            # FP16 statistics
            'fp16_total_mb': sum(r['fp16_size_mb'] for r in fp16_results) if fp16_results else 0,
            'fp16_max_error': max((r['fp16_max_error'] for r in fp16_results), default=0),
            'fp16_avg_error': np.mean([r['fp16_avg_error'] for r in fp16_results]) if fp16_results else 0,
            'fp16_rel_error': np.mean([r['fp16_rel_error'] for r in fp16_results]) if fp16_results else 0,
            'fp16_compression': 100.0 * 0.5,  # Always 50% of FP32

            # INT8 statistics
            'int8_total_mb': sum(r['int8_size_mb'] for r in int8_results) if int8_results else 0,
            'int8_max_error': max((r['int8_max_error'] for r in int8_results), default=0),
            'int8_avg_error': np.mean([r['int8_avg_error'] for r in int8_results]) if int8_results else 0,
            'int8_rel_error': np.mean([r['int8_rel_error'] for r in int8_results]) if int8_results else 0,
            'int8_compression': 100.0 * 0.25,  # Always 25% of FP32

            'results': results
        }

    def _print_comparison_tables(self, results, overall):
        """Print formatted comparison tables."""

        print("\n" + "=" * 80)
        print("  MEMORY USAGE COMPARISON")
        print("=" * 80)

        print(f"\n{'Format':<10s} {'Size (MB)':<12s} {'vs FP32':<12s} {'Compression':<15s}")
        print("-" * 80)

        fp32_size = overall['fp32_total_mb']
        print(f"{'FP32':<10s} {fp32_size:>10.1f} MB {'baseline':<12s} {'0%':<15s}")

        if overall['fp16_available'] > 0:
            fp16_size = overall['fp16_total_mb']
            fp16_savings = fp32_size - fp16_size
            fp16_pct = overall['fp16_compression']
            print(f"{'FP16':<10s} {fp16_size:>10.1f} MB "
                  f"{-fp16_savings:>+10.1f} MB {100-fp16_pct:>6.1f}% smaller")

        if overall['int8_available'] > 0:
            int8_size = overall['int8_total_mb']
            int8_savings = fp32_size - int8_size
            int8_pct = overall['int8_compression']
            print(f"{'INT8':<10s} {int8_size:>10.1f} MB "
                  f"{-int8_savings:>+10.1f} MB {100-int8_pct:>6.1f}% smaller")

        print("\n" + "=" * 80)
        print("  QUANTIZATION ERROR COMPARISON")
        print("=" * 80)

        print(f"\n{'Format':<10s} {'Max Error':<15s} {'Avg Error':<15s} {'Rel Error':<15s}")
        print("-" * 80)

        print(f"{'FP32':<10s} {'0.0 (baseline)':<15s} {'0.0':<15s} {'0.0%':<15s}")

        if overall['fp16_available'] > 0:
            print(f"{'FP16':<10s} "
                  f"{overall['fp16_max_error']:<15.2e} "
                  f"{overall['fp16_avg_error']:<15.2e} "
                  f"{100*overall['fp16_rel_error']:<14.4f}%")

        if overall['int8_available'] > 0:
            print(f"{'INT8':<10s} "
                  f"{overall['int8_max_error']:<15.2e} "
                  f"{overall['int8_avg_error']:<15.2e} "
                  f"{100*overall['int8_rel_error']:<14.4f}%")

        # Top 10 weights by FP16 error
        if overall['fp16_available'] > 0:
            print("\n" + "=" * 80)
            print("  TOP 10 WEIGHTS BY FP16 ERROR")
            print("=" * 80 + "\n")

            fp16_results = [r for r in results if r['fp16_available']]
            sorted_fp16 = sorted(fp16_results, key=lambda r: r['fp16_max_error'], reverse=True)[:10]

            for i, r in enumerate(sorted_fp16, 1):
                print(f"{i:2d}. {r['name']:50s} "
                      f"max_err={r['fp16_max_error']:.2e} "
                      f"rel_err={100*r['fp16_rel_error']:.4f}%")

        # Top 10 weights by INT8 error
        if overall['int8_available'] > 0:
            print("\n" + "=" * 80)
            print("  TOP 10 WEIGHTS BY INT8 ERROR")
            print("=" * 80 + "\n")

            int8_results = [r for r in results if r['int8_available']]
            sorted_int8 = sorted(int8_results, key=lambda r: r['int8_max_error'], reverse=True)[:10]

            for i, r in enumerate(sorted_int8, 1):
                print(f"{i:2d}. {r['name']:50s} "
                      f"max_err={r['int8_max_error']:.2e} "
                      f"rel_err={100*r['int8_rel_error']:.4f}%")

        # Recommendations
        print("\n" + "=" * 80)
        print("  RECOMMENDATIONS")
        print("=" * 80)

        self._print_recommendations(overall)

    def _print_recommendations(self, overall):
        """Print recommendations based on comparison."""

        print("\n**Memory Optimization:**")

        if overall['fp32_total_mb'] > 500:
            print(f"  ⚠️  Large model ({overall['fp32_total_mb']:.0f} MB) - "
                  "quantization recommended")
        else:
            print(f"  ✅ Moderate model size ({overall['fp32_total_mb']:.0f} MB)")

        print("\n**FP16 Assessment:**")
        if overall['fp16_available'] > 0:
            if overall['fp16_max_error'] < 1e-3:
                print("  ✅ EXCELLENT: FP16 is highly accurate (max error < 0.001)")
                print("     → RECOMMENDED for production use")
            elif overall['fp16_max_error'] < 1e-2:
                print("  ✅ GOOD: FP16 has acceptable accuracy (max error < 0.01)")
                print("     → Safe for most use cases")
            else:
                print("  ⚠️  MODERATE: FP16 has noticeable error")
                print("     → Test inference quality before deployment")
        else:
            print("  ❌ FP16 weights not available")

        print("\n**INT8 Assessment:**")
        if overall['int8_available'] > 0:
            if overall['int8_max_error'] < 1e-2:
                print("  ✅ GOOD: INT8 has acceptable accuracy (max error < 0.01)")
                print("     → Consider for extreme memory constraints")
            elif overall['int8_max_error'] < 5e-2:
                print("  ⚠️  MODERATE: INT8 has noticeable error (max error < 0.05)")
                print("     → Test WER carefully before deployment")
            else:
                print("  ❌ POOR: INT8 has high error (max error >= 0.05)")
                print("     → Likely to degrade inference quality")
        else:
            print("  ❌ INT8 weights not available")

        print("\n**Recommended Strategy:**")
        if overall['fp16_available'] and overall['fp16_max_error'] < 1e-2:
            print("  🎯 Use FP16 for balanced accuracy and memory")
            print(f"     - Memory: {overall['fp16_total_mb']:.1f} MB (50% savings)")
            print(f"     - Accuracy: Excellent (max error {overall['fp16_max_error']:.2e})")
        elif overall['int8_available'] and overall['int8_max_error'] < 1e-2:
            print("  🎯 Use INT8 for maximum compression")
            print(f"     - Memory: {overall['int8_total_mb']:.1f} MB (75% savings)")
            print(f"     - Accuracy: Acceptable (max error {overall['int8_max_error']:.2e})")
        else:
            print("  🎯 Use FP32 for maximum accuracy")
            print(f"     - Memory: {overall['fp32_total_mb']:.1f} MB (baseline)")
            print(f"     - Accuracy: Perfect (no quantization)")

        print("\n**Next Steps:**")
        print("  1. Test inference quality with each format")
        print("  2. Measure WER (Word Error Rate) on validation set")
        print("  3. Profile memory usage and latency")
        print("  4. Choose format based on accuracy/memory trade-off")

if __name__ == "__main__":
    # Configuration
    fp32_dir = "./weights/whisper_base_fp32"
    fp16_dir = "./weights/whisper_base_fp16"
    int8_dir = "./weights/whisper_base_int8"

    # Check directories exist
    if not Path(fp32_dir).exists():
        print(f"❌ FP32 directory not found: {fp32_dir}")
        print("   Run extract_whisper_weights.py first")
        sys.exit(1)

    # Create comparison
    comparison = WeightFormatComparison(fp32_dir, fp16_dir, int8_dir)

    # Run comparison
    overall = comparison.compare_all_weights()

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    print(f"\nTotal weights:     {overall['total_weights']}")
    print(f"FP16 available:    {overall['fp16_available']}")
    print(f"INT8 available:    {overall['int8_available']}")

    print(f"\nMemory usage:")
    print(f"  FP32:  {overall['fp32_total_mb']:.1f} MB (baseline)")
    if overall['fp16_available'] > 0:
        print(f"  FP16:  {overall['fp16_total_mb']:.1f} MB "
              f"(saves {overall['fp32_total_mb'] - overall['fp16_total_mb']:.1f} MB)")
    if overall['int8_available'] > 0:
        print(f"  INT8:  {overall['int8_total_mb']:.1f} MB "
              f"(saves {overall['fp32_total_mb'] - overall['int8_total_mb']:.1f} MB)")

    print("\n✅ Weight format comparison complete!")
