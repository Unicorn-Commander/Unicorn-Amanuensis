#!/usr/bin/env python3
"""
Week 17: NPU Performance Measurement Suite
CC-1L NPU Performance Measurement Team

Mission: Measure ACTUAL NPU performance with real audio and validate 400-500× realtime target

This script performs comprehensive performance measurements:
1. Audio processing with different durations (1s, 5s, 30s)
2. Total processing time measurement
3. Realtime factor calculation (audio_duration / processing_time)
4. NPU utilization estimation
5. Bottleneck identification
6. Comparison against 400-500× target

Performance Targets (Week 15 validation):
- 400-500× realtime for Whisper Base
- NPU utilization: ~2.3% (97% headroom)
- Latency: 60-75ms for 30s audio
"""

import numpy as np
import requests
import json
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceResult:
    """Performance measurement result"""
    test_name: str
    audio_duration_s: float
    audio_file: str
    
    # Timing metrics
    processing_time_ms: float = 0.0
    realtime_factor: float = 0.0  # audio_duration / processing_time
    
    # Component breakdown (if available)
    mel_time_ms: float = 0.0
    encoder_time_ms: float = 0.0
    decoder_time_ms: float = 0.0
    
    # NPU metrics
    npu_utilization_percent: float = 0.0
    
    # Status
    success: bool = True
    error: str = ""
    
    # Transcription result (for validation)
    transcription: str = ""


class Week17PerformanceMeasurement:
    """
    Week 17 Performance Measurement Suite
    
    Measures actual NPU performance with real audio and validates against
    the 400-500× realtime target.
    """
    
    def __init__(self, service_url: str = "http://127.0.0.1:9050"):
        """Initialize performance measurement suite"""
        self.service_url = service_url
        self.results: List[PerformanceResult] = []
        
        # Performance targets (from Week 15 validation)
        self.target_realtime_min = 400.0
        self.target_realtime_max = 500.0
        self.target_npu_utilization = 2.3  # percent
        
        logger.info("="*80)
        logger.info("  WEEK 17: NPU PERFORMANCE MEASUREMENT")
        logger.info("="*80)
        logger.info(f"  Service URL: {service_url}")
        logger.info(f"  Target: {self.target_realtime_min}-{self.target_realtime_max}× realtime")
        logger.info(f"  Expected NPU utilization: ~{self.target_npu_utilization}%")
        logger.info("="*80)
        
    def check_service_health(self) -> bool:
        """Check if service is running and healthy"""
        try:
            logger.info("\n" + "="*80)
            logger.info("  CHECKING SERVICE HEALTH")
            logger.info("="*80)
            
            response = requests.get(f"{self.service_url}/health", timeout=5)
            response.raise_for_status()
            
            health = response.json()
            logger.info(f"  Service: {health.get('service', 'Unknown')}")
            logger.info(f"  Status: {health.get('status', 'Unknown')}")
            logger.info(f"  NPU Enabled: {health.get('npu_enabled', False)}")
            
            if health.get('status') != 'healthy':
                logger.error(f"  ❌ Service unhealthy: {health.get('status')}")
                return False
                
            if not health.get('npu_enabled', False):
                logger.warning("  ⚠️  NPU not enabled - performance will be degraded")
                
            logger.info("  ✅ Service is healthy and ready")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"  ❌ Service not responding: {e}")
            logger.error("\n  To start service:")
            logger.error("    cd /home/ccadmin/CC-1L/npu-services/unicorn-amanuensis")
            logger.error("    source ~/mlir-aie/ironenv/bin/activate")
            logger.error("    source /opt/xilinx/xrt/setup.sh 2>/dev/null")
            logger.error("    ENABLE_PIPELINE=true python -m uvicorn xdna2.server:app --host 127.0.0.1 --port 9050")
            return False
            
    def measure_audio_performance(
        self,
        audio_path: Path,
        test_name: str
    ) -> PerformanceResult:
        """
        Measure performance for a single audio file
        
        Args:
            audio_path: Path to audio file
            test_name: Name for this test
            
        Returns:
            PerformanceResult with timing metrics
        """
        logger.info("\n" + "="*80)
        logger.info(f"  TEST: {test_name}")
        logger.info("="*80)
        logger.info(f"  Audio file: {audio_path.name}")
        
        # Get audio duration (approximate from file size for 16kHz WAV)
        file_size_bytes = audio_path.stat().st_size
        # WAV header is 44 bytes, data is 2 bytes per sample @ 16kHz
        audio_duration_s = (file_size_bytes - 44) / (2 * 16000)
        logger.info(f"  Audio duration: {audio_duration_s:.2f}s")
        
        result = PerformanceResult(
            test_name=test_name,
            audio_duration_s=audio_duration_s,
            audio_file=str(audio_path.name)
        )
        
        try:
            # Prepare request
            with open(audio_path, 'rb') as f:
                files = {'file': (audio_path.name, f, 'audio/wav')}
                
                # Measure end-to-end time
                start_time = time.perf_counter()
                
                response = requests.post(
                    f"{self.service_url}/v1/audio/transcriptions",
                    files=files,
                    timeout=60  # 60 second timeout
                )
                
                end_time = time.perf_counter()
                
            # Calculate timing
            processing_time_s = end_time - start_time
            result.processing_time_ms = processing_time_s * 1000
            
            if audio_duration_s > 0:
                result.realtime_factor = audio_duration_s / processing_time_s
            
            # Check response
            response.raise_for_status()
            data = response.json()
            
            # Extract transcription
            result.transcription = data.get('text', '')
            
            # Extract component timing if available
            if 'timing' in data:
                timing = data['timing']
                result.mel_time_ms = timing.get('mel_ms', 0.0)
                result.encoder_time_ms = timing.get('encoder_ms', 0.0)
                result.decoder_time_ms = timing.get('decoder_ms', 0.0)
            
            # Estimate NPU utilization
            # Week 15 analysis: 2.3% utilization needed for 400-500× realtime
            # If we achieve target, utilization should be ~2-3%
            if result.realtime_factor > 0:
                # Simple estimate: (target_realtime / actual_realtime) * target_utilization
                result.npu_utilization_percent = (self.target_realtime_min / result.realtime_factor) * self.target_npu_utilization
            
            # Log results
            logger.info(f"\n  ✅ Transcription successful!")
            logger.info(f"  Processing time: {result.processing_time_ms:.2f} ms")
            logger.info(f"  Realtime factor: {result.realtime_factor:.1f}×")
            logger.info(f"  NPU utilization (est): {result.npu_utilization_percent:.2f}%")
            
            if result.realtime_factor >= self.target_realtime_min:
                logger.info(f"  🎯 TARGET MET! ({self.target_realtime_min}-{self.target_realtime_max}×)")
            else:
                gap = self.target_realtime_min - result.realtime_factor
                logger.warning(f"  ⚠️  Below target by {gap:.1f}×")
            
            # Component breakdown
            if result.encoder_time_ms > 0:
                logger.info(f"\n  Component Breakdown:")
                logger.info(f"    Mel: {result.mel_time_ms:.2f} ms")
                logger.info(f"    Encoder: {result.encoder_time_ms:.2f} ms")
                logger.info(f"    Decoder: {result.decoder_time_ms:.2f} ms")
            
            logger.info(f"\n  Transcription: \"{result.transcription[:100]}...\"")
            
        except requests.exceptions.Timeout:
            result.success = False
            result.error = "Request timeout (>60s)"
            logger.error(f"  ❌ {result.error}")
            
        except requests.exceptions.RequestException as e:
            result.success = False
            result.error = str(e)
            logger.error(f"  ❌ Request failed: {e}")
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"  ❌ Unexpected error: {e}")
            
        self.results.append(result)
        return result
        
    def run_full_benchmark_suite(self, test_dir: Path) -> Dict:
        """
        Run full performance benchmark suite
        
        Tests:
        1. 1s audio - Quick validation
        2. 5s audio - Medium duration
        3. 30s audio - Target scenario (Whisper Base)
        4. Silence - Edge case
        
        Args:
            test_dir: Directory containing test audio files
            
        Returns:
            Complete benchmark results
        """
        logger.info("\n" + "="*80)
        logger.info("  RUNNING FULL PERFORMANCE BENCHMARK SUITE")
        logger.info("="*80)
        
        # Check service health first
        if not self.check_service_health():
            return {
                'status': 'FAILED',
                'error': 'Service not available',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            }
        
        # Test files
        test_files = [
            ('test_1s.wav', '1 Second Audio'),
            ('test_5s.wav', '5 Second Audio'),
            ('test_30s.wav', '30 Second Audio (Target)'),
            ('test_silence.wav', 'Silence (Edge Case)'),
        ]
        
        # Run tests
        for wav_file, test_name in test_files:
            audio_path = test_dir / wav_file
            
            if not audio_path.exists():
                logger.warning(f"\n⚠️  Skipping {test_name}: {wav_file} not found")
                continue
                
            self.measure_audio_performance(audio_path, test_name)
            
            # Brief pause between tests
            time.sleep(0.5)
        
        # Generate summary
        return self.generate_summary()
        
    def generate_summary(self) -> Dict:
        """Generate comprehensive performance summary"""
        logger.info("\n" + "="*80)
        logger.info("  PERFORMANCE SUMMARY")
        logger.info("="*80)
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'total_tests': len(self.results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'target': f"{self.target_realtime_min}-{self.target_realtime_max}× realtime",
            'results': [asdict(r) for r in self.results]
        }
        
        if successful_results:
            # Overall metrics
            realtime_factors = [r.realtime_factor for r in successful_results]
            avg_realtime = np.mean(realtime_factors)
            min_realtime = np.min(realtime_factors)
            max_realtime = np.max(realtime_factors)
            
            target_met = min_realtime >= self.target_realtime_min
            
            summary['metrics'] = {
                'avg_realtime_factor': float(avg_realtime),
                'min_realtime_factor': float(min_realtime),
                'max_realtime_factor': float(max_realtime),
                'target_met': target_met,
                'avg_npu_utilization_percent': float(np.mean([r.npu_utilization_percent for r in successful_results]))
            }
            
            logger.info(f"\n  Tests: {len(successful_results)}/{len(self.results)} successful")
            logger.info(f"\n  Realtime Factors:")
            logger.info(f"    Average: {avg_realtime:.1f}×")
            logger.info(f"    Min: {min_realtime:.1f}×")
            logger.info(f"    Max: {max_realtime:.1f}×")
            logger.info(f"    Target: {self.target_realtime_min}-{self.target_realtime_max}×")
            
            if target_met:
                logger.info(f"\n  🎉 TARGET ACHIEVED! All tests meet {self.target_realtime_min}× minimum")
                summary['status'] = 'TARGET_MET'
            else:
                gap = self.target_realtime_min - min_realtime
                logger.warning(f"\n  ⚠️  Target not met. Gap: {gap:.1f}×")
                summary['status'] = 'BELOW_TARGET'
                
            # Bottleneck analysis
            logger.info(f"\n  Estimated NPU Utilization: {summary['metrics']['avg_npu_utilization_percent']:.2f}%")
            
            if summary['metrics']['avg_npu_utilization_percent'] > 5.0:
                logger.info("  💡 High NPU utilization - consider optimization")
            else:
                logger.info("  ✅ Low NPU utilization - excellent headroom")
                
        else:
            logger.error("\n  ❌ No successful tests")
            summary['status'] = 'ALL_FAILED'
            
        if failed_results:
            logger.error(f"\n  Failed Tests:")
            for r in failed_results:
                logger.error(f"    - {r.test_name}: {r.error}")
                
        return summary
        
    def save_results(self, output_path: Path):
        """Save results to JSON file"""
        summary = self.generate_summary() if not hasattr(self, '_summary') else self._summary
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"\n  📄 Results saved to: {output_path}")


def main():
    """Main performance measurement"""
    import sys
    
    try:
        # Initialize measurement suite
        suite = Week17PerformanceMeasurement()
        
        # Test directory
        test_dir = Path(__file__).parent
        
        # Run full benchmark suite
        results = suite.run_full_benchmark_suite(test_dir)
        
        # Save results
        output_dir = Path("/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/tests/results")
        suite.save_results(output_dir / "week17_performance_results.json")
        
        # Exit code
        if results.get('status') == 'TARGET_MET':
            logger.info("\n✅ SUCCESS: Performance target achieved!")
            return 0
        elif results.get('status') == 'BELOW_TARGET':
            logger.warning("\n⚠️  PARTIAL: Performance below target but functional")
            return 1
        else:
            logger.error("\n❌ FAILED: Performance measurement failed")
            return 2
            
    except Exception as e:
        logger.error(f"\n✗ FATAL: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
