#!/usr/bin/env python3
"""
NPU Utilization Monitoring Script

Monitors NPU utilization during pipeline load testing to validate:
- NPU utilization improvement: 0.12% → 15% (+1775% target)
- Sustained NPU activity during concurrent requests
- Correlation between throughput and NPU usage

Usage:
    # Monitor NPU during load test (run in separate terminal)
    python monitor_npu_utilization.py --duration 60

    # Monitor with custom sampling rate
    python monitor_npu_utilization.py --duration 60 --interval 0.5

    # Export to CSV
    python monitor_npu_utilization.py --duration 60 --output npu_stats.csv

Note:
    This script requires access to NPU monitoring tools:
    - xrt-smi (AMD XRT tools)
    - /sys/class/accel/accel* (sysfs interface)
    - Custom NPU profiling if available

Performance Target:
    - Sequential: 0.12% NPU utilization (single request)
    - Pipeline: 15% NPU utilization (10-15 concurrent requests)
    - Target improvement: +1775%

Author: CC-1L Multi-Stream Integration Team
Date: November 1, 2025
"""

import subprocess
import time
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class NPUSample:
    """Single NPU utilization sample"""
    timestamp: float
    utilization_percent: float
    power_watts: Optional[float] = None
    temperature_c: Optional[float] = None
    frequency_mhz: Optional[float] = None

    def __str__(self):
        return (f"[{self.timestamp:.2f}s] "
                f"Util: {self.utilization_percent:.1f}%, "
                f"Power: {self.power_watts:.1f}W, "
                f"Temp: {self.temperature_c:.1f}°C")


@dataclass
class NPUMonitoringResult:
    """Results from NPU monitoring session"""
    duration: float
    samples: List[NPUSample] = field(default_factory=list)

    @property
    def mean_utilization(self) -> float:
        """Mean utilization percentage"""
        if not self.samples:
            return 0.0
        return np.mean([s.utilization_percent for s in self.samples])

    @property
    def max_utilization(self) -> float:
        """Maximum utilization percentage"""
        if not self.samples:
            return 0.0
        return np.max([s.utilization_percent for s in self.samples])

    @property
    def min_utilization(self) -> float:
        """Minimum utilization percentage"""
        if not self.samples:
            return 0.0
        return np.min([s.utilization_percent for s in self.samples])

    @property
    def mean_power(self) -> float:
        """Mean power consumption in watts"""
        power_samples = [s.power_watts for s in self.samples if s.power_watts is not None]
        if not power_samples:
            return 0.0
        return np.mean(power_samples)

    def print_summary(self):
        """Print monitoring summary"""
        print(f"\n{'='*70}")
        print(f"  NPU Utilization Monitoring Results")
        print(f"{'='*70}")
        print(f"  Duration:           {self.duration:.1f}s")
        print(f"  Samples:            {len(self.samples)}")
        print(f"  ")
        print(f"  Utilization:")
        print(f"    Mean:             {self.mean_utilization:.2f}%")
        print(f"    Max:              {self.max_utilization:.2f}%")
        print(f"    Min:              {self.min_utilization:.2f}%")
        print(f"  ")

        if self.mean_power > 0:
            print(f"  Power:")
            print(f"    Mean:             {self.mean_power:.2f}W")
            print(f"  ")

        # Performance targets
        print(f"  Performance Targets:")
        print(f"    Sequential:       0.12% utilization")
        print(f"    Pipeline:         15% utilization (+1775%)")
        print(f"  ")

        if self.mean_utilization >= 15.0:
            print(f"  ✅ Target achieved! ({self.mean_utilization:.2f}% >= 15%)")
        elif self.mean_utilization >= 10.0:
            print(f"  ⚠️  Close to target ({self.mean_utilization:.2f}% / 15%)")
        else:
            print(f"  ❌ Below target ({self.mean_utilization:.2f}% / 15%)")

        print(f"{'='*70}\n")


class NPUMonitor:
    """NPU utilization monitor"""

    def __init__(self, sampling_interval: float = 0.5):
        """
        Initialize NPU monitor.

        Args:
            sampling_interval: Time between samples in seconds
        """
        self.sampling_interval = sampling_interval
        self.monitoring_method = self._detect_monitoring_method()

        print(f"[NPUMonitor] Initialized")
        print(f"  Sampling interval: {sampling_interval}s")
        print(f"  Monitoring method: {self.monitoring_method}")

    def _detect_monitoring_method(self) -> str:
        """
        Detect available NPU monitoring method.

        Returns:
            Monitoring method name
        """
        # Try xrt-smi
        try:
            result = subprocess.run(
                ["xrt-smi", "examine"],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            if result.returncode == 0:
                return "xrt-smi"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try sysfs
        sysfs_paths = list(Path("/sys/class/accel").glob("accel*"))
        if sysfs_paths:
            return "sysfs"

        # Fallback to simulation
        print("  Warning: No NPU monitoring tools detected. Using simulation mode.")
        return "simulation"

    def _sample_xrt_smi(self) -> Optional[NPUSample]:
        """
        Sample NPU utilization using xrt-smi.

        Returns:
            NPUSample or None if failed
        """
        try:
            result = subprocess.run(
                ["xrt-smi", "examine", "-r", "all"],
                capture_output=True,
                text=True,
                timeout=2.0
            )

            if result.returncode != 0:
                return None

            # Parse output (example format, adjust based on actual xrt-smi output)
            output = result.stdout

            # Extract utilization (example: look for "NPU Utilization: XX%")
            utilization = 0.0
            for line in output.split('\n'):
                if "utilization" in line.lower():
                    # Try to extract percentage
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            try:
                                utilization = float(part.replace('%', ''))
                                break
                            except ValueError:
                                pass

            return NPUSample(
                timestamp=time.time(),
                utilization_percent=utilization
            )

        except Exception as e:
            print(f"  Warning: xrt-smi sampling failed: {e}")
            return None

    def _sample_sysfs(self) -> Optional[NPUSample]:
        """
        Sample NPU utilization using sysfs.

        Returns:
            NPUSample or None if failed
        """
        try:
            # Find NPU device
            accel_paths = list(Path("/sys/class/accel").glob("accel*"))
            if not accel_paths:
                return None

            accel_path = accel_paths[0]

            # Try to read utilization (path varies by driver)
            util_paths = [
                accel_path / "device" / "utilization",
                accel_path / "device" / "npu_utilization",
                accel_path / "utilization"
            ]

            utilization = 0.0
            for util_path in util_paths:
                if util_path.exists():
                    try:
                        utilization = float(util_path.read_text().strip())
                        break
                    except (ValueError, IOError):
                        continue

            return NPUSample(
                timestamp=time.time(),
                utilization_percent=utilization
            )

        except Exception as e:
            print(f"  Warning: sysfs sampling failed: {e}")
            return None

    def _sample_simulation(self) -> NPUSample:
        """
        Simulate NPU utilization (for testing when NPU not available).

        Returns:
            Simulated NPUSample
        """
        # Simulate realistic utilization pattern
        # Base: 5-20% with some variation
        base_util = 10.0 + np.random.normal(0, 3)
        utilization = max(0.0, min(100.0, base_util))

        return NPUSample(
            timestamp=time.time(),
            utilization_percent=utilization,
            power_watts=5.0 + utilization * 0.1,  # Simulate power scaling
            temperature_c=45.0 + utilization * 0.3  # Simulate temp scaling
        )

    def sample(self) -> Optional[NPUSample]:
        """
        Take a single NPU utilization sample.

        Returns:
            NPUSample or None if sampling failed
        """
        if self.monitoring_method == "xrt-smi":
            return self._sample_xrt_smi()
        elif self.monitoring_method == "sysfs":
            return self._sample_sysfs()
        elif self.monitoring_method == "simulation":
            return self._sample_simulation()
        else:
            return None

    def monitor(self, duration: float) -> NPUMonitoringResult:
        """
        Monitor NPU utilization for specified duration.

        Args:
            duration: Monitoring duration in seconds

        Returns:
            NPUMonitoringResult with samples
        """
        print(f"\n[NPUMonitor] Starting monitoring for {duration}s...")
        print(f"  Method: {self.monitoring_method}")
        print(f"  Sampling every {self.sampling_interval}s")
        print(f"{'='*70}\n")

        samples = []
        start_time = time.time()

        while time.time() - start_time < duration:
            sample = self.sample()
            if sample:
                samples.append(sample)

                # Print progress every 5 seconds
                elapsed = time.time() - start_time
                if len(samples) % int(5 / self.sampling_interval) == 0:
                    recent = samples[-10:] if len(samples) >= 10 else samples
                    avg_util = np.mean([s.utilization_percent for s in recent])
                    print(f"  [{elapsed:.0f}s] Avg utilization (last 10 samples): {avg_util:.2f}%")

            # Sleep until next sample
            next_sample = start_time + (len(samples) + 1) * self.sampling_interval
            sleep_time = next_sample - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        total_duration = time.time() - start_time

        print(f"\n[NPUMonitor] Monitoring complete")
        print(f"  Collected {len(samples)} samples in {total_duration:.1f}s")

        return NPUMonitoringResult(
            duration=total_duration,
            samples=samples
        )

    def export_csv(self, result: NPUMonitoringResult, output_path: Path):
        """
        Export monitoring results to CSV.

        Args:
            result: Monitoring results
            output_path: Output CSV file path
        """
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Header
            writer.writerow(['timestamp', 'utilization_percent', 'power_watts', 'temperature_c'])

            # Data
            for sample in result.samples:
                writer.writerow([
                    sample.timestamp,
                    sample.utilization_percent,
                    sample.power_watts or '',
                    sample.temperature_c or ''
                ])

        print(f"\n  Exported to: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Monitor NPU utilization")
    parser.add_argument("--duration", type=float, default=30.0, help="Monitoring duration (seconds)")
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling interval (seconds)")
    parser.add_argument("--output", type=Path, help="Export to CSV file")

    args = parser.parse_args()

    # Create monitor
    monitor = NPUMonitor(sampling_interval=args.interval)

    # Run monitoring
    result = monitor.monitor(args.duration)

    # Print summary
    result.print_summary()

    # Export if requested
    if args.output:
        monitor.export_csv(result, args.output)

    # Exit with status based on target achievement
    if result.mean_utilization >= 15.0:
        print("✅ NPU utilization target achieved!")
        sys.exit(0)
    else:
        print(f"⚠️  NPU utilization below target: {result.mean_utilization:.2f}% / 15%")
        sys.exit(1)


if __name__ == "__main__":
    main()
