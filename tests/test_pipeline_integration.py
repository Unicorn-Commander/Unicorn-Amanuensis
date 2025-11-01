#!/usr/bin/env python3
"""
Pipeline Integration Tests

Comprehensive test suite for multi-stream pipeline integration:
- Single request processing
- Concurrent request handling
- Pipeline statistics endpoints
- Health check endpoints
- Error handling and recovery
- Timeout behavior

Usage:
    # Run all tests
    pytest test_pipeline_integration.py -v

    # Run specific test
    pytest test_pipeline_integration.py::test_pipeline_single_request -v

    # Run with coverage
    pytest test_pipeline_integration.py --cov=transcription_pipeline

Requirements:
    - Service must be running on localhost:9050
    - ENABLE_PIPELINE=true environment variable
    - Test audio files in tests/audio/ directory

Author: CC-1L Multi-Stream Integration Team
Date: November 1, 2025
"""

import pytest
import aiohttp
import asyncio
import time
import os
from pathlib import Path

# Test configuration
BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:9050")
TEST_AUDIO_DIR = Path(__file__).parent / "audio"
TIMEOUT = 30.0  # seconds


@pytest.fixture
async def http_session():
    """Create aiohttp session for tests"""
    async with aiohttp.ClientSession() as session:
        yield session


@pytest.fixture
def test_audio_file():
    """Get path to test audio file"""
    audio_path = TEST_AUDIO_DIR / "test_audio.wav"
    if not audio_path.exists():
        pytest.skip(f"Test audio not found: {audio_path}")
    return audio_path


@pytest.mark.asyncio
async def test_service_health(http_session):
    """Test 1: Verify service is running and healthy"""
    async with http_session.get(f"{BASE_URL}/health") as resp:
        assert resp.status == 200, "Service not healthy"
        data = await resp.json()
        assert data.get("status") == "healthy", f"Service unhealthy: {data}"
        print(f"  Service health: {data.get('status')}")
        print(f"  Backend: {data.get('backend')}")


@pytest.mark.asyncio
async def test_pipeline_enabled(http_session):
    """Test 2: Verify pipeline is enabled"""
    async with http_session.get(f"{BASE_URL}/health/pipeline") as resp:
        assert resp.status == 200, "Pipeline health endpoint failed"
        data = await resp.json()
        assert data.get("mode") == "pipeline", "Pipeline not in pipeline mode"
        assert data.get("healthy") == True, f"Pipeline unhealthy: {data}"
        print(f"  Pipeline mode: {data.get('mode')}")
        print(f"  Pipeline healthy: {data.get('healthy')}")


@pytest.mark.asyncio
async def test_pipeline_single_request(http_session, test_audio_file):
    """Test 3: Process single request through pipeline"""
    print(f"\n  Testing single request with: {test_audio_file}")

    with open(test_audio_file, "rb") as f:
        data = aiohttp.FormData()
        data.add_field("file", f, filename="test.wav", content_type="audio/wav")

        start_time = time.perf_counter()

        async with http_session.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            data=data,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        ) as resp:
            elapsed = time.perf_counter() - start_time

            assert resp.status == 200, f"Request failed with status {resp.status}"

            result = await resp.json()

            # Validate response structure
            assert "text" in result, "Missing 'text' in response"
            assert "segments" in result, "Missing 'segments' in response"
            assert "performance" in result, "Missing 'performance' in response"

            # Validate performance data
            perf = result["performance"]
            assert perf.get("mode") == "pipeline", "Not using pipeline mode"

            print(f"  Response time: {elapsed*1000:.1f}ms")
            print(f"  Audio duration: {perf.get('audio_duration_s', 0):.2f}s")
            print(f"  Processing time: {perf.get('processing_time_s', 0)*1000:.1f}ms")
            print(f"  Realtime factor: {perf.get('realtime_factor', 0):.1f}x")
            print(f"  Text length: {len(result.get('text', ''))} chars")


@pytest.mark.asyncio
async def test_pipeline_concurrent_requests(http_session, test_audio_file):
    """Test 4: Process 5 concurrent requests"""
    num_concurrent = 5
    print(f"\n  Testing {num_concurrent} concurrent requests")

    async def send_request(request_id):
        """Send a single transcription request"""
        with open(test_audio_file, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename=f"test_{request_id}.wav", content_type="audio/wav")

            start = time.perf_counter()

            async with http_session.post(
                f"{BASE_URL}/v1/audio/transcriptions",
                data=data,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT)
            ) as resp:
                latency = time.perf_counter() - start

                assert resp.status == 200, f"Request {request_id} failed with status {resp.status}"

                result = await resp.json()
                assert "text" in result, f"Request {request_id} missing 'text'"

                return {
                    "request_id": request_id,
                    "latency": latency,
                    "text_length": len(result.get("text", "")),
                    "performance": result.get("performance", {})
                }

    # Launch all requests concurrently
    start_time = time.perf_counter()
    tasks = [send_request(i) for i in range(num_concurrent)]
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time

    # Validate all requests succeeded
    assert len(results) == num_concurrent, f"Expected {num_concurrent} results, got {len(results)}"

    # Calculate statistics
    latencies = [r["latency"] for r in results]
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)

    print(f"  Total time: {total_time*1000:.1f}ms")
    print(f"  Throughput: {num_concurrent/total_time:.2f} req/s")
    print(f"  Avg latency: {avg_latency*1000:.1f}ms")
    print(f"  Min latency: {min_latency*1000:.1f}ms")
    print(f"  Max latency: {max_latency*1000:.1f}ms")

    # All requests should complete successfully
    for i, result in enumerate(results):
        assert result["text_length"] > 0, f"Request {i} returned empty text"


@pytest.mark.asyncio
async def test_pipeline_stats_endpoint(http_session):
    """Test 5: Verify pipeline statistics endpoint"""
    async with http_session.get(f"{BASE_URL}/stats/pipeline") as resp:
        assert resp.status == 200, "Pipeline stats endpoint failed"

        stats = await resp.json()

        # Validate structure
        assert stats.get("enabled") == True, "Pipeline not enabled"
        assert "throughput_rps" in stats, "Missing throughput_rps"
        assert "queue" in stats, "Missing queue stats"
        assert "stages" in stats, "Missing stage stats"

        # Validate stage stats
        stages = stats["stages"]
        assert "stage1_load_mel" in stages, "Missing stage1 stats"
        assert "stage2_encoder" in stages, "Missing stage2 stats"
        assert "stage3_decoder_align" in stages, "Missing stage3 stats"

        print(f"\n  Pipeline Statistics:")
        print(f"  Throughput: {stats.get('throughput_rps', 0):.2f} req/s")
        print(f"  Queue depth: {stats.get('queue', {}).get('depth', 0)}")
        print(f"  Active requests: {stats.get('active_requests', 0)}")

        for stage_name, stage_stats in stages.items():
            print(f"  {stage_name}:")
            print(f"    Processed: {stage_stats.get('total_processed', 0)}")
            print(f"    Avg time: {stage_stats.get('avg_time_ms', 0):.2f}ms")
            print(f"    Queue depth: {stage_stats.get('queue_depth', 0)}")


@pytest.mark.asyncio
async def test_pipeline_health_endpoint(http_session):
    """Test 6: Verify pipeline health endpoint"""
    async with http_session.get(f"{BASE_URL}/health/pipeline") as resp:
        assert resp.status == 200, "Pipeline health endpoint failed"

        health = await resp.json()

        # Validate structure
        assert "healthy" in health, "Missing 'healthy' field"
        assert "stages" in health, "Missing 'stages' field"

        # Check each stage
        stages = health["stages"]
        for stage_name in ["stage1", "stage2", "stage3"]:
            assert stage_name in stages, f"Missing {stage_name} health"
            stage_health = stages[stage_name]
            assert stage_health.get("healthy") == True, f"{stage_name} unhealthy: {stage_health}"
            assert stage_health.get("running") == True, f"{stage_name} not running"
            assert stage_health.get("workers_active", 0) > 0, f"{stage_name} has no active workers"

        print(f"\n  Pipeline Health:")
        print(f"  Overall: {health.get('healthy')}")
        print(f"  Message: {health.get('message')}")

        for stage_name, stage_health in stages.items():
            print(f"  {stage_name}: healthy={stage_health.get('healthy')}, "
                  f"workers={stage_health.get('workers_active')}/{stage_health.get('workers_total')}")


@pytest.mark.asyncio
async def test_pipeline_high_concurrency(http_session, test_audio_file):
    """Test 7: Process 15 concurrent requests (stress test)"""
    num_concurrent = 15
    print(f"\n  Testing {num_concurrent} concurrent requests (stress test)")

    async def send_request(request_id):
        """Send a single transcription request"""
        with open(test_audio_file, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename=f"stress_{request_id}.wav", content_type="audio/wav")

            start = time.perf_counter()

            try:
                async with http_session.post(
                    f"{BASE_URL}/v1/audio/transcriptions",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=TIMEOUT)
                ) as resp:
                    latency = time.perf_counter() - start

                    if resp.status != 200:
                        return {
                            "request_id": request_id,
                            "success": False,
                            "status": resp.status,
                            "latency": latency
                        }

                    result = await resp.json()

                    return {
                        "request_id": request_id,
                        "success": True,
                        "latency": latency,
                        "text_length": len(result.get("text", ""))
                    }

            except asyncio.TimeoutError:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": "timeout",
                    "latency": TIMEOUT
                }

            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "latency": time.perf_counter() - start
                }

    # Launch all requests concurrently
    start_time = time.perf_counter()
    tasks = [send_request(i) for i in range(num_concurrent)]
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time

    # Count successes and failures
    successes = [r for r in results if r.get("success", False)]
    failures = [r for r in results if not r.get("success", False)]

    success_rate = len(successes) / len(results) * 100

    print(f"  Total time: {total_time*1000:.1f}ms")
    print(f"  Throughput: {len(successes)/total_time:.2f} req/s")
    print(f"  Success rate: {success_rate:.1f}% ({len(successes)}/{len(results)})")

    if successes:
        latencies = [r["latency"] for r in successes]
        print(f"  Avg latency: {sum(latencies)/len(latencies)*1000:.1f}ms")
        print(f"  Max latency: {max(latencies)*1000:.1f}ms")

    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for failure in failures[:5]:  # Show first 5 failures
            print(f"    Request {failure['request_id']}: {failure.get('error', failure.get('status'))}")

    # Minimum success requirement: 80% of requests should succeed
    assert success_rate >= 80.0, f"Success rate too low: {success_rate:.1f}% (expected >= 80%)"


@pytest.mark.asyncio
async def test_pipeline_sequential_consistency(http_session, test_audio_file):
    """Test 8: Verify sequential requests produce consistent results"""
    num_requests = 3
    print(f"\n  Testing {num_requests} sequential requests for consistency")

    results = []

    for i in range(num_requests):
        with open(test_audio_file, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename=f"consistency_{i}.wav", content_type="audio/wav")

            async with http_session.post(
                f"{BASE_URL}/v1/audio/transcriptions",
                data=data,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT)
            ) as resp:
                assert resp.status == 200, f"Request {i} failed"

                result = await resp.json()
                results.append({
                    "text": result.get("text", ""),
                    "num_segments": len(result.get("segments", []))
                })

    # All results should be identical (same audio file)
    reference_text = results[0]["text"]
    reference_segments = results[0]["num_segments"]

    for i, result in enumerate(results[1:], 1):
        # Text should be identical or very similar
        assert result["text"] == reference_text, \
            f"Request {i} produced different text: '{result['text']}' vs '{reference_text}'"

        # Segment count should match
        assert result["num_segments"] == reference_segments, \
            f"Request {i} produced different segment count: {result['num_segments']} vs {reference_segments}"

    print(f"  All {num_requests} requests produced consistent results")
    print(f"  Text length: {len(reference_text)} chars")
    print(f"  Segments: {reference_segments}")


if __name__ == "__main__":
    # Run tests with pytest
    import sys
    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
