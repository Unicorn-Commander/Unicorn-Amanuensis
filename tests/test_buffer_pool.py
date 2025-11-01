#!/usr/bin/env python3
"""
Buffer Pool Unit Tests

Comprehensive test suite for buffer pool implementation, covering:
- Single-threaded allocation/release
- Concurrent access (10+ threads)
- Pool exhaustion and error handling
- Memory leak verification (1000+ iterations)
- Statistics accuracy
- Configuration edge cases

Author: CC-1L Buffer Pool Implementation Team
Date: November 1, 2025
Status: Week 8 Days 1-2 Implementation
"""

import pytest
import numpy as np
import threading
import time
from typing import List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffer_pool import BufferPool, GlobalBufferManager, BufferMetadata


class TestBufferPool:
    """Test suite for BufferPool class"""

    def test_initialization(self):
        """Test buffer pool initialization"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=3,
            max_size=5
        )

        assert pool.buffer_type == 'test'
        assert pool.buffer_size == 1024
        assert pool.initial_size == 3
        assert pool.max_size == 5

        stats = pool.get_stats()
        assert stats['buffers_available'] == 3
        assert stats['buffers_in_use'] == 0
        assert stats['total_buffers'] == 3

    def test_invalid_initialization(self):
        """Test that invalid configurations raise errors"""
        # initial_size > max_size
        with pytest.raises(ValueError):
            BufferPool(
                buffer_type='test',
                buffer_size=1024,
                initial_size=10,
                max_size=5
            )

        # Negative sizes
        with pytest.raises(ValueError):
            BufferPool(
                buffer_type='test',
                buffer_size=1024,
                initial_size=-1,
                max_size=5
            )

        # Invalid buffer size
        with pytest.raises(ValueError):
            BufferPool(
                buffer_type='test',
                buffer_size=0,
                initial_size=1,
                max_size=5
            )

    def test_acquire_release_single_thread(self):
        """Test basic acquire/release in single thread"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=2,
            max_size=5
        )

        # Acquire buffer
        buffer1 = pool.acquire()
        assert buffer1 is not None
        assert buffer1.shape == (256,)  # 1024 / 4 (float32)
        assert buffer1.dtype == np.float32

        stats = pool.get_stats()
        assert stats['buffers_in_use'] == 1
        assert stats['buffers_available'] == 1
        assert stats['pool_hits'] == 1

        # Acquire second buffer
        buffer2 = pool.acquire()
        assert buffer2 is not None
        assert buffer2 is not buffer1

        stats = pool.get_stats()
        assert stats['buffers_in_use'] == 2
        assert stats['buffers_available'] == 0

        # Release first buffer
        pool.release(buffer1)

        stats = pool.get_stats()
        assert stats['buffers_in_use'] == 1
        assert stats['buffers_available'] == 1
        assert stats['total_releases'] == 1

        # Release second buffer
        pool.release(buffer2)

        stats = pool.get_stats()
        assert stats['buffers_in_use'] == 0
        assert stats['buffers_available'] == 2
        assert stats['total_releases'] == 2

    def test_buffer_reuse(self):
        """Test that released buffers are reused"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=1,
            max_size=5
        )

        # Acquire and release buffer
        buffer1 = pool.acquire()
        buffer1_id = id(buffer1)
        pool.release(buffer1)

        # Acquire again - should get same buffer
        buffer2 = pool.acquire()
        buffer2_id = id(buffer2)

        assert buffer1_id == buffer2_id, "Released buffer should be reused"

        stats = pool.get_stats()
        assert stats['pool_hits'] == 2  # Both acquires hit the pool
        assert stats['pool_misses'] == 0

        pool.release(buffer2)

    def test_pool_growth(self):
        """Test pool grows when exhausted (up to max_size)"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=1,
            max_size=3
        )

        buffers = []

        # Acquire all buffers (should trigger growth)
        for i in range(3):
            buffer = pool.acquire()
            buffers.append(buffer)

        stats = pool.get_stats()
        assert stats['total_buffers'] == 3
        assert stats['buffers_in_use'] == 3
        assert stats['pool_hits'] == 1  # First one was from pool
        assert stats['pool_misses'] == 2  # Next two were allocated

        # Release all
        for buffer in buffers:
            pool.release(buffer)

        stats = pool.get_stats()
        assert stats['buffers_available'] == 3

    def test_pool_exhaustion(self):
        """Test that pool raises error when max_size exceeded"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=1,
            max_size=2
        )

        # Acquire max buffers
        buffer1 = pool.acquire()
        buffer2 = pool.acquire()

        # Try to acquire one more - should raise error
        with pytest.raises(RuntimeError) as exc_info:
            pool.acquire()

        assert "exhausted" in str(exc_info.value).lower()

        stats = pool.get_stats()
        assert stats['errors'] == 1

        # Release and try again
        pool.release(buffer1)

        buffer3 = pool.acquire()  # Should succeed now
        assert buffer3 is not None

        pool.release(buffer2)
        pool.release(buffer3)

    def test_double_release_protection(self):
        """Test that double release is detected"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=1,
            max_size=5
        )

        buffer = pool.acquire()
        pool.release(buffer)

        # Try to release again
        pool.release(buffer)  # Should log error but not crash

        stats = pool.get_stats()
        assert stats['errors'] == 1

    def test_custom_shape(self):
        """Test buffer pool with custom shape"""
        pool = BufferPool(
            buffer_type='mel',
            buffer_size=960 * 1024,
            dtype=np.float32,
            shape=(80, 3000),
            initial_size=2,
            max_size=5
        )

        buffer = pool.acquire()
        assert buffer.shape == (80, 3000)
        assert buffer.dtype == np.float32

        pool.release(buffer)

    def test_zero_on_release(self):
        """Test that buffers are zeroed when zero_on_release=True"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=1,
            max_size=5,
            zero_on_release=True
        )

        buffer = pool.acquire()
        buffer.fill(42.0)  # Fill with non-zero values

        pool.release(buffer)

        # Acquire again - should be zeroed
        buffer2 = pool.acquire()
        assert np.all(buffer2 == 0.0), "Buffer should be zeroed on release"

        pool.release(buffer2)

    def test_statistics_accuracy(self):
        """Test that statistics are accurately tracked"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=2,
            max_size=5
        )

        # Acquire and release 10 times
        for _ in range(10):
            buffer = pool.acquire()
            pool.release(buffer)

        stats = pool.get_stats()
        assert stats['total_acquires'] == 10
        assert stats['total_releases'] == 10
        assert stats['pool_hits'] == 10  # All from initial pool
        assert stats['pool_misses'] == 0
        assert stats['hit_rate'] == 1.0
        assert stats['miss_rate'] == 0.0
        assert stats['leaked_buffers'] == 0
        assert not stats['has_leaks']

    def test_leak_detection(self):
        """Test that memory leaks are detected"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=2,
            max_size=5
        )

        # Acquire buffers but don't release
        buffer1 = pool.acquire()
        buffer2 = pool.acquire()

        stats = pool.get_stats()
        assert stats['total_acquires'] == 2
        assert stats['total_releases'] == 0
        assert stats['leaked_buffers'] == 2
        assert stats['has_leaks']

        # Now release
        pool.release(buffer1)
        pool.release(buffer2)

        stats = pool.get_stats()
        assert stats['leaked_buffers'] == 0
        assert not stats['has_leaks']


class TestConcurrentAccess:
    """Test suite for concurrent buffer pool access"""

    def test_concurrent_acquire_release(self):
        """Test concurrent buffer acquisition from multiple threads"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=5,
            max_size=20
        )

        num_threads = 10
        iterations_per_thread = 100
        errors = []

        def worker():
            """Worker thread that acquires and releases buffers"""
            try:
                for _ in range(iterations_per_thread):
                    buffer = pool.acquire()
                    # Simulate work
                    buffer.fill(threading.current_thread().ident)
                    time.sleep(0.0001)  # 0.1ms
                    pool.release(buffer)
            except Exception as e:
                errors.append(e)

        # Launch threads
        threads = [threading.Thread(target=worker) for _ in range(num_threads)]

        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        elapsed = time.time() - start_time

        # Check for errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify statistics
        stats = pool.get_stats()
        total_operations = num_threads * iterations_per_thread

        assert stats['total_acquires'] == total_operations
        assert stats['total_releases'] == total_operations
        assert stats['leaked_buffers'] == 0
        assert not stats['has_leaks']

        print(f"\n  Concurrent test: {num_threads} threads × {iterations_per_thread} ops = {total_operations} ops in {elapsed:.2f}s")
        print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")
        print(f"  Pool size: {stats['total_buffers']} buffers")

    def test_concurrent_pool_exhaustion(self):
        """Test behavior when pool is exhausted under concurrent load"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=2,
            max_size=5  # Small pool
        )

        num_threads = 10
        success_count = [0]
        error_count = [0]
        lock = threading.Lock()

        def worker():
            """Worker that tries to hold buffer for a while"""
            try:
                buffer = pool.acquire()
                with lock:
                    success_count[0] += 1
                time.sleep(0.01)  # Hold buffer for 10ms
                pool.release(buffer)
            except RuntimeError:
                # Pool exhausted - expected
                with lock:
                    error_count[0] += 1

        # Launch threads
        threads = [threading.Thread(target=worker) for _ in range(num_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Some should succeed, some should fail due to exhaustion
        assert success_count[0] + error_count[0] == num_threads
        assert success_count[0] <= pool.max_size  # Can't exceed max
        assert error_count[0] > 0  # Some should have failed

        print(f"\n  Pool exhaustion test: {success_count[0]} succeeded, {error_count[0]} failed")

    def test_high_concurrency_stress(self):
        """Stress test with high concurrency"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=4096,
            initial_size=10,
            max_size=50
        )

        num_threads = 20
        iterations_per_thread = 50
        errors = []

        def worker():
            """Aggressive worker that rapidly acquires/releases"""
            try:
                for _ in range(iterations_per_thread):
                    buffer = pool.acquire()
                    # Minimal work
                    buffer[0] = 1.0
                    pool.release(buffer)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]

        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        elapsed = time.time() - start_time

        assert len(errors) == 0, f"Errors: {errors}"

        stats = pool.get_stats()
        total_ops = num_threads * iterations_per_thread

        assert stats['total_acquires'] == total_ops
        assert stats['total_releases'] == total_ops
        assert stats['leaked_buffers'] == 0

        ops_per_sec = total_ops / elapsed
        print(f"\n  Stress test: {total_ops} ops in {elapsed:.2f}s ({ops_per_sec:.0f} ops/sec)")


class TestMemoryLeakVerification:
    """Test suite for memory leak detection"""

    def test_1000_iterations_no_leaks(self):
        """Verify no memory leaks over 1000 iterations"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=5,
            max_size=10
        )

        iterations = 1000

        for i in range(iterations):
            buffer = pool.acquire()
            buffer.fill(float(i))
            pool.release(buffer)

        stats = pool.get_stats()
        assert stats['total_acquires'] == iterations
        assert stats['total_releases'] == iterations
        assert stats['leaked_buffers'] == 0
        assert not stats['has_leaks']

        print(f"\n  Memory leak test: {iterations} iterations, 0 leaks")

    def test_alternating_acquire_release(self):
        """Test alternating patterns of acquire/release"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=2048,
            initial_size=3,
            max_size=6
        )

        buffers: List[np.ndarray] = []

        # Acquire all
        for _ in range(3):
            buffers.append(pool.acquire())

        # Release all
        for buffer in buffers:
            pool.release(buffer)

        buffers.clear()

        # Repeat pattern 100 times
        for _ in range(100):
            for _ in range(3):
                buffers.append(pool.acquire())

            for buffer in buffers:
                pool.release(buffer)

            buffers.clear()

        stats = pool.get_stats()
        assert stats['leaked_buffers'] == 0
        assert stats['hit_rate'] > 0.99  # Should be nearly 100% hits


class TestGlobalBufferManager:
    """Test suite for GlobalBufferManager"""

    def setup_method(self):
        """Reset singleton before each test"""
        GlobalBufferManager._instance = None

    def test_singleton_pattern(self):
        """Test that GlobalBufferManager is a singleton"""
        manager1 = GlobalBufferManager.instance()
        manager2 = GlobalBufferManager.instance()

        assert manager1 is manager2, "Should return same instance"

        # Direct construction should raise error
        with pytest.raises(RuntimeError):
            GlobalBufferManager()

    def test_configure_pools(self):
        """Test pool configuration"""
        manager = GlobalBufferManager.instance()

        manager.configure({
            'mel': {
                'size': 960 * 1024,
                'count': 5,
                'max_count': 10
            },
            'audio': {
                'size': 480 * 1024,
                'count': 3,
                'max_count': 6
            }
        })

        stats = manager.get_stats()
        assert 'mel' in stats
        assert 'audio' in stats
        assert stats['mel']['buffers_available'] == 5
        assert stats['audio']['buffers_available'] == 3

    def test_acquire_release_multiple_pools(self):
        """Test acquiring from multiple pools"""
        manager = GlobalBufferManager.instance()

        manager.configure({
            'mel': {'size': 1024, 'count': 2, 'max_count': 5},
            'audio': {'size': 2048, 'count': 2, 'max_count': 5}
        })

        # Acquire from different pools
        mel_buf = manager.acquire('mel')
        audio_buf = manager.acquire('audio')

        assert mel_buf.shape != audio_buf.shape

        # Release
        manager.release('mel', mel_buf)
        manager.release('audio', audio_buf)

        stats = manager.get_stats()
        assert stats['mel']['leaked_buffers'] == 0
        assert stats['audio']['leaked_buffers'] == 0

    def test_unknown_pool_error(self):
        """Test that accessing unknown pool raises error"""
        manager = GlobalBufferManager.instance()

        with pytest.raises(ValueError):
            manager.acquire('nonexistent')

        # Configure a pool
        manager.configure({'test': {'size': 1024, 'count': 1, 'max_count': 2}})

        buffer = manager.acquire('test')

        # Try to release to wrong pool
        with pytest.raises(ValueError):
            manager.release('nonexistent', buffer)

        manager.release('test', buffer)

    def test_concurrent_multi_pool_access(self):
        """Test concurrent access to multiple pools"""
        manager = GlobalBufferManager.instance()

        manager.configure({
            'pool1': {'size': 1024, 'count': 5, 'max_count': 10},
            'pool2': {'size': 2048, 'count': 5, 'max_count': 10},
            'pool3': {'size': 4096, 'count': 5, 'max_count': 10}
        })

        num_threads = 15
        iterations = 50
        errors = []

        def worker(pool_name):
            """Worker for specific pool"""
            try:
                for _ in range(iterations):
                    buffer = manager.acquire(pool_name)
                    buffer.fill(threading.current_thread().ident)
                    time.sleep(0.0001)
                    manager.release(pool_name, buffer)
            except Exception as e:
                errors.append(e)

        # Create threads for different pools
        threads = []
        for i in range(num_threads):
            pool_name = f'pool{(i % 3) + 1}'
            threads.append(threading.Thread(target=worker, args=(pool_name,)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0

        stats = manager.get_stats()
        for pool_name in ['pool1', 'pool2', 'pool3']:
            assert stats[pool_name]['leaked_buffers'] == 0

    def test_shutdown(self):
        """Test graceful shutdown"""
        manager = GlobalBufferManager.instance()

        manager.configure({
            'test': {'size': 1024, 'count': 2, 'max_count': 5}
        })

        # Acquire but don't release (simulate leak)
        buffer = manager.acquire('test')

        # Shutdown should warn about leaks
        manager.shutdown()

        stats = manager.get_stats()
        # Pool should be cleared despite leak
        assert stats['test']['buffers_available'] == 0
        assert stats['test']['buffers_in_use'] == 0


class TestEdgeCases:
    """Test suite for edge cases and error conditions"""

    def test_zero_initial_size(self):
        """Test pool with zero initial size"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=0,
            max_size=5
        )

        stats = pool.get_stats()
        assert stats['buffers_available'] == 0
        assert stats['total_buffers'] == 0

        # First acquire should allocate
        buffer = pool.acquire()
        assert buffer is not None

        stats = pool.get_stats()
        assert stats['pool_misses'] == 1
        assert stats['total_buffers'] == 1

        pool.release(buffer)

    def test_dtype_variations(self):
        """Test different dtypes"""
        for dtype in [np.float32, np.float64, np.int32, np.int8]:
            pool = BufferPool(
                buffer_type='test',
                buffer_size=1024,
                dtype=dtype,
                initial_size=1,
                max_size=5
            )

            buffer = pool.acquire()
            assert buffer.dtype == dtype

            pool.release(buffer)

    def test_large_buffers(self):
        """Test pool with large buffers"""
        pool = BufferPool(
            buffer_type='large',
            buffer_size=10 * 1024 * 1024,  # 10MB
            initial_size=2,
            max_size=4
        )

        buffer = pool.acquire()
        assert buffer.nbytes == 10 * 1024 * 1024

        pool.release(buffer)

    def test_clear_with_buffers_in_use(self):
        """Test clearing pool with buffers still in use"""
        pool = BufferPool(
            buffer_type='test',
            buffer_size=1024,
            initial_size=2,
            max_size=5
        )

        buffer = pool.acquire()

        # Clear should warn but not crash
        pool.clear()

        stats = pool.get_stats()
        assert stats['buffers_available'] == 0
        assert stats['buffers_in_use'] == 0  # Cleared despite in-use


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
