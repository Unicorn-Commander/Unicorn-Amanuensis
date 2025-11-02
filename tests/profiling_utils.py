#!/usr/bin/env python3
"""
Week 18: Performance Profiling Utilities
CC-1L Performance Engineering Team

Hierarchical performance profiling framework with:
- Context managers for automatic timing
- Nested measurement support
- Statistical analysis (mean, median, p95, p99)
- Export to JSON for analysis
- Memory tracking
- CPU/GPU/NPU profiling
"""

import time
import json
import statistics
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimingMeasurement:
    """Single timing measurement"""
    name: str
    parent: Optional[str]
    start_time: float
    end_time: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'parent': self.parent,
            'duration_ms': self.duration_ms,
            'metadata': self.metadata
        }


@dataclass
class TimingStatistics:
    """Statistical analysis of timing measurements"""
    name: str
    count: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float
    p95_ms: float
    p99_ms: float
    total_ms: float

    def to_dict(self) -> Dict:
        return asdict(self)


class PerformanceProfiler:
    """
    Hierarchical performance profiler with statistical analysis

    Example usage:
        profiler = PerformanceProfiler()

        with profiler.measure("total_processing"):
            # Do some work
            with profiler.measure("mel_spectrogram", parent="total_processing"):
                # Generate mel spectrogram
                with profiler.measure("fft", parent="mel_spectrogram"):
                    # FFT computation
                    pass
                with profiler.measure("log_mel", parent="mel_spectrogram"):
                    # Log-mel transformation
                    pass

            with profiler.measure("npu_encoder", parent="total_processing"):
                # NPU execution
                pass

        # Get statistics
        stats = profiler.get_statistics()
        profiler.print_report()
        profiler.save_json("results.json")
    """

    def __init__(self, name: str = "profiler"):
        """Initialize profiler"""
        self.name = name
        self.measurements: List[TimingMeasurement] = []
        self.active_measurements: Dict[str, float] = {}
        self.measurement_stack: List[str] = []

    @contextmanager
    def measure(self, name: str, parent: Optional[str] = None, **metadata):
        """
        Context manager for timing a code block

        Args:
            name: Name of the measurement
            parent: Optional parent measurement name for hierarchy
            **metadata: Additional metadata to store with measurement
        """
        # Use parent from stack if not explicitly provided
        if parent is None and self.measurement_stack:
            parent = self.measurement_stack[-1]

        # Create full name with parent
        full_name = f"{parent}.{name}" if parent else name

        # Start timing
        start_time = time.perf_counter()
        self.active_measurements[full_name] = start_time
        self.measurement_stack.append(full_name)

        try:
            yield
        finally:
            # End timing
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Record measurement
            measurement = TimingMeasurement(
                name=name,
                parent=parent,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                metadata=metadata
            )
            self.measurements.append(measurement)

            # Cleanup
            self.active_measurements.pop(full_name, None)
            self.measurement_stack.pop()

    def get_timings(self, name: Optional[str] = None) -> List[float]:
        """Get all timing measurements for a specific name"""
        if name is None:
            return [m.duration_ms for m in self.measurements]
        return [m.duration_ms for m in self.measurements if m.name == name]

    def get_statistics(self, name: Optional[str] = None) -> Dict[str, TimingStatistics]:
        """
        Calculate statistics for all measurements or specific name

        Args:
            name: Optional name to filter by

        Returns:
            Dictionary of statistics by measurement name
        """
        # Group measurements by name
        grouped: Dict[str, List[float]] = defaultdict(list)
        for m in self.measurements:
            if name is None or m.name == name:
                key = m.name
                grouped[key].append(m.duration_ms)

        # Calculate statistics for each group
        stats = {}
        for key, timings in grouped.items():
            if not timings:
                continue

            stats[key] = TimingStatistics(
                name=key,
                count=len(timings),
                mean_ms=statistics.mean(timings),
                median_ms=statistics.median(timings),
                min_ms=min(timings),
                max_ms=max(timings),
                std_dev_ms=statistics.stdev(timings) if len(timings) > 1 else 0.0,
                p95_ms=self._percentile(timings, 0.95),
                p99_ms=self._percentile(timings, 0.99),
                total_ms=sum(timings)
            )

        return stats

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]

    def get_hierarchy(self) -> Dict[str, Any]:
        """
        Build hierarchical tree of measurements

        Returns:
            Nested dictionary representing timing hierarchy
        """
        tree = {}

        # Build tree from measurements
        for m in self.measurements:
            if m.parent is None:
                # Root level
                if m.name not in tree:
                    tree[m.name] = {'measurements': [], 'children': {}}
                tree[m.name]['measurements'].append(m.duration_ms)
            else:
                # Nested level - find parent
                parent_parts = m.parent.split('.')
                current = tree

                # Navigate to parent
                for part in parent_parts:
                    if part not in current:
                        current[part] = {'measurements': [], 'children': {}}
                    current = current[part]['children'] if 'children' in current[part] else current[part]

                # Add measurement to parent
                if m.name not in current:
                    current[m.name] = {'measurements': [], 'children': {}}
                current[m.name]['measurements'].append(m.duration_ms)

        return tree

    def print_report(self, detailed: bool = True):
        """
        Print formatted performance report

        Args:
            detailed: Include detailed statistics
        """
        logger.info("\n" + "="*80)
        logger.info(f"  PERFORMANCE PROFILING REPORT: {self.name}")
        logger.info("="*80)

        if not self.measurements:
            logger.info("  No measurements recorded")
            return

        stats = self.get_statistics()

        # Print summary table
        logger.info("\n  TIMING SUMMARY")
        logger.info("  " + "-"*76)
        logger.info(f"  {'Operation':<30} {'Count':>8} {'Mean':>10} {'Median':>10} {'P95':>10}")
        logger.info("  " + "-"*76)

        for name, stat in sorted(stats.items(), key=lambda x: x[1].mean_ms, reverse=True):
            logger.info(f"  {name:<30} {stat.count:>8} {stat.mean_ms:>9.2f}ms {stat.median_ms:>9.2f}ms {stat.p95_ms:>9.2f}ms")

        logger.info("  " + "-"*76)

        if detailed:
            logger.info("\n  DETAILED STATISTICS")
            logger.info("  " + "-"*76)

            for name, stat in sorted(stats.items()):
                logger.info(f"\n  {name}:")
                logger.info(f"    Count:     {stat.count}")
                logger.info(f"    Mean:      {stat.mean_ms:.2f} ms")
                logger.info(f"    Median:    {stat.median_ms:.2f} ms")
                logger.info(f"    Std Dev:   {stat.std_dev_ms:.2f} ms")
                logger.info(f"    Min:       {stat.min_ms:.2f} ms")
                logger.info(f"    Max:       {stat.max_ms:.2f} ms")
                logger.info(f"    P95:       {stat.p95_ms:.2f} ms")
                logger.info(f"    P99:       {stat.p99_ms:.2f} ms")
                logger.info(f"    Total:     {stat.total_ms:.2f} ms")

        # Print hierarchy
        logger.info("\n  TIMING HIERARCHY")
        logger.info("  " + "-"*76)
        self._print_hierarchy_tree(self.get_hierarchy())

        logger.info("\n" + "="*80)

    def _print_hierarchy_tree(self, tree: Dict, indent: int = 2):
        """Recursively print hierarchy tree"""
        for name, data in tree.items():
            measurements = data.get('measurements', [])
            children = data.get('children', {})

            if measurements:
                avg = statistics.mean(measurements)
                count = len(measurements)
                logger.info(f"  {' '*indent}{name}: {avg:.2f}ms (n={count})")

            if children:
                self._print_hierarchy_tree(children, indent + 2)

    def save_json(self, output_path: Path):
        """
        Save profiling results to JSON file

        Args:
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.get_statistics()
        hierarchy = self.get_hierarchy()

        data = {
            'profiler_name': self.name,
            'total_measurements': len(self.measurements),
            'statistics': {name: stat.to_dict() for name, stat in stats.items()},
            'hierarchy': hierarchy,
            'raw_measurements': [m.to_dict() for m in self.measurements]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"\n  📄 Profiling results saved to: {output_path}")

    def reset(self):
        """Reset all measurements"""
        self.measurements.clear()
        self.active_measurements.clear()
        self.measurement_stack.clear()


class MultiRunProfiler:
    """
    Profiler for running multiple iterations and aggregating results

    Example:
        profiler = MultiRunProfiler(num_runs=10, warmup_runs=2)

        for i in range(profiler.total_runs):
            with profiler.run(i):
                # Do work
                with profiler.measure("operation"):
                    # Timed operation
                    pass

        profiler.print_summary()
    """

    def __init__(self, num_runs: int = 10, warmup_runs: int = 2):
        """
        Initialize multi-run profiler

        Args:
            num_runs: Number of measured runs (after warmup)
            warmup_runs: Number of warmup runs to discard
        """
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.total_runs = num_runs + warmup_runs
        self.profilers: List[PerformanceProfiler] = []
        self.current_run = 0
        self.current_profiler: Optional[PerformanceProfiler] = None

    @contextmanager
    def run(self, run_id: int):
        """
        Context manager for a single run

        Args:
            run_id: Run identifier (0-indexed)
        """
        is_warmup = run_id < self.warmup_runs
        name = f"warmup_{run_id}" if is_warmup else f"run_{run_id - self.warmup_runs}"

        self.current_profiler = PerformanceProfiler(name=name)
        self.current_run = run_id

        try:
            yield self.current_profiler
        finally:
            if not is_warmup:
                self.profilers.append(self.current_profiler)
            self.current_profiler = None

    @contextmanager
    def measure(self, name: str, parent: Optional[str] = None, **metadata):
        """Measure within current run"""
        if self.current_profiler is None:
            raise RuntimeError("No active run - use within 'with profiler.run()' block")

        with self.current_profiler.measure(name, parent, **metadata):
            yield

    def get_aggregated_statistics(self) -> Dict[str, TimingStatistics]:
        """Get aggregated statistics across all runs"""
        if not self.profilers:
            return {}

        # Collect all timings by name
        all_timings: Dict[str, List[float]] = defaultdict(list)

        for profiler in self.profilers:
            stats = profiler.get_statistics()
            for name, stat in stats.items():
                all_timings[name].extend(profiler.get_timings(name))

        # Calculate aggregated statistics
        aggregated = {}
        for name, timings in all_timings.items():
            if not timings:
                continue

            aggregated[name] = TimingStatistics(
                name=name,
                count=len(timings),
                mean_ms=statistics.mean(timings),
                median_ms=statistics.median(timings),
                min_ms=min(timings),
                max_ms=max(timings),
                std_dev_ms=statistics.stdev(timings) if len(timings) > 1 else 0.0,
                p95_ms=PerformanceProfiler._percentile(timings, 0.95),
                p99_ms=PerformanceProfiler._percentile(timings, 0.99),
                total_ms=sum(timings)
            )

        return aggregated

    def print_summary(self):
        """Print summary of all runs"""
        logger.info("\n" + "="*80)
        logger.info(f"  MULTI-RUN PERFORMANCE SUMMARY")
        logger.info("="*80)
        logger.info(f"  Warmup runs: {self.warmup_runs}")
        logger.info(f"  Measured runs: {self.num_runs}")
        logger.info(f"  Total measurements: {len(self.profilers)}")

        stats = self.get_aggregated_statistics()

        if not stats:
            logger.info("  No measurements recorded")
            return

        # Print aggregated statistics
        logger.info("\n  AGGREGATED STATISTICS (across all runs)")
        logger.info("  " + "-"*76)
        logger.info(f"  {'Operation':<30} {'Count':>8} {'Mean':>10} {'Median':>10} {'P95':>10} {'P99':>10}")
        logger.info("  " + "-"*76)

        for name, stat in sorted(stats.items(), key=lambda x: x[1].mean_ms, reverse=True):
            logger.info(f"  {name:<30} {stat.count:>8} {stat.mean_ms:>9.2f}ms {stat.median_ms:>9.2f}ms {stat.p95_ms:>9.2f}ms {stat.p99_ms:>9.2f}ms")

        logger.info("  " + "-"*76)
        logger.info("\n" + "="*80)

    def save_json(self, output_path: Path):
        """Save multi-run results to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        aggregated_stats = self.get_aggregated_statistics()

        data = {
            'num_runs': self.num_runs,
            'warmup_runs': self.warmup_runs,
            'aggregated_statistics': {name: stat.to_dict() for name, stat in aggregated_stats.items()},
            'individual_runs': []
        }

        for profiler in self.profilers:
            stats = profiler.get_statistics()
            data['individual_runs'].append({
                'name': profiler.name,
                'statistics': {name: stat.to_dict() for name, stat in stats.items()}
            })

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"\n  📄 Multi-run results saved to: {output_path}")


def create_ascii_bar_chart(data: Dict[str, float], title: str = "Performance Breakdown", max_width: int = 60) -> str:
    """
    Create ASCII bar chart

    Args:
        data: Dictionary of name -> value (in ms)
        title: Chart title
        max_width: Maximum bar width in characters

    Returns:
        ASCII bar chart as string
    """
    if not data:
        return "No data"

    max_value = max(data.values())

    lines = []
    lines.append("\n" + "="*80)
    lines.append(f"  {title}")
    lines.append("="*80)

    for name, value in sorted(data.items(), key=lambda x: x[1], reverse=True):
        bar_width = int((value / max_value) * max_width) if max_value > 0 else 0
        bar = "█" * bar_width
        percentage = (value / sum(data.values())) * 100 if sum(data.values()) > 0 else 0
        lines.append(f"  {name:<30} {bar:<{max_width}} {value:>8.2f}ms ({percentage:>5.1f}%)")

    lines.append("="*80)

    return "\n".join(lines)


def create_waterfall_diagram(measurements: List[TimingMeasurement], title: str = "Pipeline Waterfall") -> str:
    """
    Create ASCII waterfall diagram showing pipeline stages

    Args:
        measurements: List of timing measurements
        title: Diagram title

    Returns:
        ASCII waterfall diagram as string
    """
    if not measurements:
        return "No measurements"

    # Sort by start time
    sorted_measurements = sorted(measurements, key=lambda m: m.start_time)

    # Calculate relative positions
    min_time = sorted_measurements[0].start_time
    max_time = max(m.end_time for m in sorted_measurements)
    total_duration = max_time - min_time

    max_width = 60

    lines = []
    lines.append("\n" + "="*80)
    lines.append(f"  {title}")
    lines.append("="*80)
    lines.append(f"  Total Duration: {total_duration*1000:.2f}ms")
    lines.append("")

    for m in sorted_measurements:
        rel_start = m.start_time - min_time
        rel_end = m.end_time - min_time

        start_pos = int((rel_start / total_duration) * max_width) if total_duration > 0 else 0
        bar_width = int((m.duration_ms / 1000 / total_duration) * max_width) if total_duration > 0 else 1
        bar_width = max(1, bar_width)  # Minimum 1 character

        prefix = " " * start_pos
        bar = "█" * bar_width

        lines.append(f"  {m.name:<30} {prefix}{bar} {m.duration_ms:.2f}ms")

    lines.append("="*80)

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)

    # Single profiler example
    profiler = PerformanceProfiler("example")

    with profiler.measure("total"):
        time.sleep(0.1)
        with profiler.measure("step1"):
            time.sleep(0.05)
        with profiler.measure("step2"):
            time.sleep(0.03)

    profiler.print_report()

    # Multi-run example
    multi_profiler = MultiRunProfiler(num_runs=5, warmup_runs=2)

    for i in range(multi_profiler.total_runs):
        with multi_profiler.run(i):
            with multi_profiler.measure("operation"):
                time.sleep(0.01)

    multi_profiler.print_summary()
