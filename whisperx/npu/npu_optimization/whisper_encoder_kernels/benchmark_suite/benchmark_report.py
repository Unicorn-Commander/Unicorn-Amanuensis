#!/usr/bin/env python3
"""
Benchmark Report Generation

Generates comprehensive markdown reports with:
- Performance summaries
- Kernel-level statistics
- Accuracy validation results
- Optimization comparison
- Progress tracking to 220x target
- Recommendations for next optimizations
"""

from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class BenchmarkReport:
    """Generate comprehensive benchmark reports in markdown format"""

    def __init__(self):
        """Initialize report generator"""
        self.current_rtf = 14.0  # Current realtime factor
        self.target_rtf = 220.0  # Target realtime factor

    def generate_markdown_report(self, results: Dict, output_file: str):
        """
        Generate comprehensive markdown report

        Args:
            results: Dictionary containing all benchmark results
            output_file: Path to output markdown file
        """
        report = self._build_report(results)

        # Write to file
        with open(output_file, 'w') as f:
            f.write(report)

        print(f"Report generated: {output_file}")

    def _build_report(self, results: Dict) -> str:
        """Build complete markdown report"""

        # Extract result sections
        kernels = results.get('kernels', {})
        pipeline = results.get('pipeline', [])
        accuracy = results.get('accuracy', {})
        comparisons = results.get('comparisons', {})

        # Calculate current RTF from pipeline if available
        if pipeline:
            self.current_rtf = pipeline[0].get('realtime_factor', self.current_rtf)
        elif comparisons:
            # Use best comparison result
            best_rtf = max([c.get('realtime_factor', 0) for c in comparisons.values()])
            if best_rtf > 0:
                self.current_rtf = best_rtf

        report = f"""# NPU Whisper Benchmark Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Hardware**: AMD Phoenix NPU (XDNA1) - 4×6 tile array
**XRT Version**: 2.20.0
**Firmware**: 1.5.5.391

---

## Executive Summary

{self._generate_executive_summary()}

---

## Performance Summary

{self._generate_performance_summary(kernels, pipeline)}

---

## Kernel Performance

{self._generate_kernel_performance(kernels)}

---

## Accuracy Validation

{self._generate_accuracy_section(accuracy)}

---

## Optimization Comparison

{self._generate_optimization_comparison(comparisons)}

---

## Progress to Target

{self._generate_progress_section()}

---

## Next Optimizations

{self._generate_recommendations()}

---

## Detailed Metrics

{self._generate_detailed_metrics(kernels, pipeline)}

---

**Report End**
"""
        return report

    def _generate_executive_summary(self) -> str:
        """Generate executive summary section"""
        progress_pct = (self.current_rtf / self.target_rtf) * 100
        gap = self.target_rtf / self.current_rtf

        return f"""| Metric | Value |
|--------|-------|
| **Current Realtime Factor** | **{self.current_rtf:.1f}x** |
| **Target Realtime Factor** | **{self.target_rtf:.0f}x** |
| **Progress** | **{progress_pct:.1f}%** |
| **Gap** | **{gap:.1f}x improvement needed** |
| **Status** | {'On track' if progress_pct > 5 else 'Behind schedule'} |"""

    def _generate_performance_summary(self, kernels: Dict, pipeline: List) -> str:
        """Generate performance summary section"""
        if not kernels and not pipeline:
            return "*No performance data available*"

        summary = "### Single Tile Performance\n\n"

        if kernels:
            total_time = sum([k.get('mean', 0) for k in kernels.values()])
            summary += f"**Total tile time**: {total_time:.3f}ms\n\n"

            summary += "| Kernel | Mean (ms) | Percentage |\n"
            summary += "|--------|-----------|------------|\n"

            for kernel_name, data in kernels.items():
                mean_time = data.get('mean', 0)
                percentage = (mean_time / total_time * 100) if total_time > 0 else 0
                summary += f"| {data.get('kernel', kernel_name)} | {mean_time:.3f} | {percentage:.1f}% |\n"

        if pipeline:
            summary += "\n### End-to-End Pipeline Performance\n\n"
            summary += "| Audio Length | Total Time | Realtime Factor | Throughput |\n"
            summary += "|--------------|------------|-----------------|------------|\n"

            for result in pipeline[:5]:  # Show first 5 results
                length = result.get('audio_length', 0)
                total = result.get('total_time', 0)
                rtf = result.get('realtime_factor', 0)
                throughput = length / (total / 1000) if total > 0 else 0

                summary += f"| {length:.0f}s | {total:.2f}ms | {rtf:.2f}x | {throughput:.2f}s/s |\n"

        return summary

    def _generate_kernel_performance(self, kernels: Dict) -> str:
        """Generate kernel performance section"""
        if not kernels:
            return "*No kernel benchmark data available*"

        section = "### Detailed Kernel Statistics\n\n"
        section += "| Kernel | Mean (ms) | Std (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) |\n"
        section += "|--------|-----------|----------|----------|----------|----------|----------|----------|\n"

        for kernel_name, data in kernels.items():
            name = data.get('kernel', kernel_name)
            mean = data.get('mean', 0)
            std = data.get('std', 0)
            p50 = data.get('p50', 0)
            p95 = data.get('p95', 0)
            p99 = data.get('p99', 0)
            min_time = data.get('min', 0)
            max_time = data.get('max', 0)

            section += f"| {name} | {mean:.3f} | {std:.3f} | {p50:.3f} | {p95:.3f} | {p99:.3f} | {min_time:.3f} | {max_time:.3f} |\n"

        return section

    def _generate_accuracy_section(self, accuracy: Dict) -> str:
        """Generate accuracy validation section"""
        if not accuracy:
            return "*No accuracy validation data available*"

        section = "### Kernel Accuracy vs CPU Reference\n\n"
        section += "| Kernel | Correlation | MSE | Max Diff | MAE | Status |\n"
        section += "|--------|-------------|-----|----------|-----|--------|\n"

        for kernel_name, data in accuracy.items():
            name = data.get('kernel', kernel_name)
            corr = data.get('correlation', 0)
            mse = data.get('mse', 0)
            max_diff = data.get('max_diff', 0)
            mae = data.get('mae', 0)
            passed = data.get('pass', False)
            status = 'PASS' if passed else 'FAIL'

            section += f"| {name} | {corr:.4f} | {mse:.4f} | {max_diff} | {mae:.2f} | {status} |\n"

        # Overall assessment
        all_passed = all([data.get('pass', False) for data in accuracy.values()])
        section += f"\n**Overall Accuracy**: {'PASS' if all_passed else 'FAIL'}\n"

        return section

    def _generate_optimization_comparison(self, comparisons: Dict) -> str:
        """Generate optimization comparison section"""
        if not comparisons:
            return "*No optimization comparison data available*"

        section = "### Optimization Impact\n\n"
        section += "| Configuration | Tile Time (ms) | RTF | Speedup vs Baseline |\n"
        section += "|---------------|----------------|-----|---------------------|\n"

        baseline_rtf = comparisons.get('baseline', {}).get('realtime_factor', 1.0)

        for config_name, data in comparisons.items():
            tile_time = data.get('tile_time_ms', 0)
            rtf = data.get('realtime_factor', 0)
            speedup = rtf / baseline_rtf if baseline_rtf > 0 else 1.0

            section += f"| {config_name} | {tile_time:.3f} | {rtf:.2f}x | {speedup:.2f}x |\n"

        return section

    def _generate_progress_section(self) -> str:
        """Generate progress tracking section"""
        progress_pct = (self.current_rtf / self.target_rtf) * 100
        filled = int(progress_pct / 5)  # 20 blocks total
        empty = 20 - filled

        progress_bar = '█' * filled + '░' * empty

        section = f"""### Current vs Target

```
Current:  {self.current_rtf:6.1f}x {progress_bar} {progress_pct:5.1f}%
Target:   {self.target_rtf:6.0f}x {'█' * 20} 100.0%
```

### Milestones

| Milestone | Target RTF | Status |
|-----------|------------|--------|
| Phase 1: Baseline | 10-15x | {'✅ COMPLETE' if self.current_rtf >= 10 else '⏳ In Progress'} |
| Phase 2: Buffer Optimization | 15-20x | {'✅ COMPLETE' if self.current_rtf >= 15 else '⏳ In Progress'} |
| Phase 3: Larger Tiles (64x64) | 40-60x | {'✅ COMPLETE' if self.current_rtf >= 40 else '📋 Planned'} |
| Phase 4: Batch Processing | 80-120x | {'✅ COMPLETE' if self.current_rtf >= 80 else '📋 Planned'} |
| Phase 5: Multi-core NPU | 150-180x | {'✅ COMPLETE' if self.current_rtf >= 150 else '📋 Planned'} |
| Phase 6: Full Optimization | 220x+ | {'✅ COMPLETE' if self.current_rtf >= 220 else '🎯 Target'} |
"""
        return section

    def _generate_recommendations(self) -> str:
        """Generate recommendations for next optimizations"""
        recommendations = []

        if self.current_rtf < 20:
            recommendations.append("1. **Implement Larger Matmul Tiles** (16×16 → 64×64)")
            recommendations.append("   - Expected: 4-6x speedup")
            recommendations.append("   - Priority: HIGH")
            recommendations.append("")
            recommendations.append("2. **Optimize Buffer Management**")
            recommendations.append("   - Minimize DMA sync operations")
            recommendations.append("   - Reuse buffers across kernel calls")
            recommendations.append("   - Expected: 1.5-2x speedup")
            recommendations.append("")

        if 20 <= self.current_rtf < 80:
            recommendations.append("1. **Implement Batch Processing**")
            recommendations.append("   - Process multiple tiles in parallel")
            recommendations.append("   - Reduce per-tile overhead")
            recommendations.append("   - Expected: 2-3x speedup")
            recommendations.append("")
            recommendations.append("2. **Optimize Attention Kernel**")
            recommendations.append("   - Currently 54% of total time")
            recommendations.append("   - Vectorize operations")
            recommendations.append("   - Expected: 1.5-2x speedup")
            recommendations.append("")

        if 80 <= self.current_rtf < 180:
            recommendations.append("1. **Enable Multi-core NPU**")
            recommendations.append("   - Utilize all 24 cores (4×6 array)")
            recommendations.append("   - Parallel tile processing")
            recommendations.append("   - Expected: 4-8x speedup")
            recommendations.append("")
            recommendations.append("2. **Pipeline Optimization**")
            recommendations.append("   - Overlap DMA with compute")
            recommendations.append("   - Prefetch next tile")
            recommendations.append("   - Expected: 1.5-2x speedup")
            recommendations.append("")

        if self.current_rtf >= 180:
            recommendations.append("1. **Fine-tune Parameters**")
            recommendations.append("   - Optimize tile sizes")
            recommendations.append("   - Tune DMA settings")
            recommendations.append("   - Expected: 1.2-1.5x speedup")
            recommendations.append("")
            recommendations.append("2. **Profile and Optimize Hotspots**")
            recommendations.append("   - Identify remaining bottlenecks")
            recommendations.append("   - Optimize critical paths")
            recommendations.append("")

        if not recommendations:
            recommendations.append("**Target achieved!** 🎉")
            recommendations.append("")
            recommendations.append("Consider:")
            recommendations.append("- Optimizing for lower power consumption")
            recommendations.append("- Supporting larger models")
            recommendations.append("- Improving accuracy")

        return "\n".join(recommendations)

    def _generate_detailed_metrics(self, kernels: Dict, pipeline: List) -> str:
        """Generate detailed metrics section"""
        section = "### System Information\n\n"
        section += "- **Hardware**: AMD Ryzen 9 8945HS with Phoenix NPU\n"
        section += "- **NPU**: 4×6 tile array, 16 TOPS INT8\n"
        section += "- **Memory**: DDR5, shared with system\n"
        section += "- **XRT**: 2.20.0\n"
        section += "- **Firmware**: 1.5.5.391\n\n"

        if kernels:
            section += "### Kernel Overhead Analysis\n\n"
            total_time = sum([k.get('mean', 0) for k in kernels.values()])

            section += "| Component | Time (ms) | Overhead (%) |\n"
            section += "|-----------|-----------|---------------|\n"

            for kernel_name, data in kernels.items():
                mean_time = data.get('mean', 0)
                overhead = ((mean_time / total_time) * 100) if total_time > 0 else 0
                section += f"| {data.get('kernel', kernel_name)} | {mean_time:.3f} | {overhead:.1f}% |\n"

        return section

    def generate_json_report(self, results: Dict, output_file: str):
        """Generate JSON report for programmatic access"""
        import json

        report_data = {
            'timestamp': datetime.now().isoformat(),
            'current_rtf': self.current_rtf,
            'target_rtf': self.target_rtf,
            'progress_pct': (self.current_rtf / self.target_rtf) * 100,
            'gap': self.target_rtf / self.current_rtf,
            'results': results
        }

        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"JSON report generated: {output_file}")


if __name__ == "__main__":
    # Example usage
    report = BenchmarkReport()

    example_results = {
        'kernels': {
            'attention': {'kernel': 'Attention', 'mean': 3.12, 'std': 0.05},
            'layernorm': {'kernel': 'LayerNorm', 'mean': 1.02, 'std': 0.03},
            'matmul': {'kernel': 'MatMul', 'mean': 0.90, 'std': 0.02},
            'gelu': {'kernel': 'GELU', 'mean': 0.47, 'std': 0.01}
        }
    }

    report.generate_markdown_report(example_results, "example_report.md")
