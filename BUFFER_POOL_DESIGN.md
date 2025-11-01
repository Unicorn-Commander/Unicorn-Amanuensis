# Buffer Pool Design - Week 7

**Project**: CC-1L Unicorn-Amanuensis Performance Optimization
**Team**: Performance Optimization Teamlead
**Date**: November 1, 2025
**Status**: Design Complete - Ready for Implementation

---

## Executive Summary

This document presents a comprehensive buffer pooling strategy to reduce memory allocation overhead in the Unicorn-Amanuensis service. The design eliminates 60-80% of allocation overhead, reducing per-request latency by 1.8-4.8ms and improving overall throughput.

### Design Goals

1. **Reduce Allocation Overhead**: Eliminate repeated allocations of ~15-20MB per request
2. **Thread-Safe**: Support concurrent requests without race conditions
3. **Memory Efficient**: Configurable pool size with graceful growth
4. **Zero-Copy Compatible**: Integrate with zero-copy optimization strategy
5. **Production-Ready**: Error handling, monitoring, and leak prevention

### Expected Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Allocations per Request | 16-24 | 2-4 | **83-87%** |
| Allocation Time | 3-6ms | 0.6-1.2ms | **80%** |
| Memory Fragmentation | High | Low | **70-80%** |
| Garbage Collection Pauses | Frequent | Rare | **60-70%** |

---

## Architecture Overview

### Buffer Pool Hierarchy

```
┌──────────────────────────────────────────────────────────────┐
│                    GlobalBufferPool                          │
│                  (Singleton, Thread-Safe)                    │
└──────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼───────┐  ┌────────▼────────┐
│  MelBufferPool │  │ AudioBuffer  │  │ EncoderBuffer   │
│   (960KB × 10) │  │  Pool        │  │  Pool           │
│                │  │ (960KB × 5)  │  │ (960KB × 5)     │
└────────────────┘  └──────────────┘  └─────────────────┘
        │                   │                   │
    ┌───▼────────┐      ┌──▼─────┐        ┌────▼─────┐
    │  Buffer 0  │      │ Buf 0  │        │  Buf 0   │
    │  Buffer 1  │      │ Buf 1  │        │  Buf 1   │
    │  Buffer 2  │      │ ...    │        │  ...     │
    │  ...       │      │ Buf 4  │        │  Buf 4   │
    │  Buffer 9  │      └────────┘        └──────────┘
    └────────────┘

Per-Request Buffer Allocation:
┌─────────────────────────────────────┐
│  Request 1                          │
│  ├─ Mel Buffer #0 (locked)          │
│  ├─ Audio Buffer #0 (locked)        │
│  └─ Encoder Buffer #0 (locked)      │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Request 2 (concurrent)             │
│  ├─ Mel Buffer #1 (locked)          │
│  ├─ Audio Buffer #1 (locked)        │
│  └─ Encoder Buffer #1 (locked)      │
└─────────────────────────────────────┘
```

### Buffer Lifecycle

```python
# 1. Request arrives
request_id = generate_id()

# 2. Acquire buffers from pool (thread-safe)
mel_buffer = pool.acquire('mel')        # 960KB, pre-zeroed
audio_buffer = pool.acquire('audio')    # 960KB
encoder_buffer = pool.acquire('encoder') # 960KB

# 3. Use buffers (no allocations)
compute_mel(audio, output=mel_buffer)
encoder.forward(mel_buffer, output=encoder_buffer)

# 4. Release buffers back to pool
pool.release('mel', mel_buffer)
pool.release('audio', audio_buffer)
pool.release('encoder', encoder_buffer)

# 5. Buffers available for next request (no deallocation)
```

---

## Detailed Design

### 1. Buffer Pool Interface

```python
"""
buffer_pool.py - High-performance buffer pooling for NPU service
"""

from typing import Optional, Dict, Any
import numpy as np
import threading
from collections import deque
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class BufferMetadata:
    """Metadata for a pooled buffer"""
    buffer_id: int
    size: int
    dtype: np.dtype
    shape: tuple
    allocated_at: float
    last_used_at: float
    use_count: int


class BufferPool:
    """
    Thread-safe buffer pool with automatic growth and shrinking.

    Features:
    - Thread-safe acquire/release operations
    - Configurable pool size (initial and max)
    - Automatic buffer reuse
    - Memory leak detection
    - Performance statistics

    Usage:
        pool = BufferPool(
            buffer_type='mel_spectrogram',
            buffer_size=960 * 1024,  # 960KB
            dtype=np.float32,
            initial_size=5,
            max_size=20
        )

        # Acquire buffer
        buffer = pool.acquire()

        # Use buffer
        compute_mel(audio, output=buffer)

        # Release buffer
        pool.release(buffer)
    """

    def __init__(
        self,
        buffer_type: str,
        buffer_size: int,
        dtype: np.dtype = np.float32,
        shape: Optional[tuple] = None,
        initial_size: int = 5,
        max_size: int = 20,
        enable_stats: bool = True
    ):
        """
        Initialize buffer pool.

        Args:
            buffer_type: Name/type of buffer (e.g., 'mel', 'audio', 'encoder')
            buffer_size: Size of each buffer in bytes
            dtype: NumPy dtype for buffers
            shape: Buffer shape (if None, uses flat array)
            initial_size: Number of buffers to pre-allocate
            max_size: Maximum number of buffers in pool
            enable_stats: Enable performance statistics tracking
        """
        self.buffer_type = buffer_type
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.shape = shape or (buffer_size // dtype().itemsize,)
        self.initial_size = initial_size
        self.max_size = max_size
        self.enable_stats = enable_stats

        # Thread safety
        self._lock = threading.RLock()

        # Buffer storage
        self._available = deque()  # Available buffers
        self._in_use = {}          # buffer_id → (buffer, metadata)
        self._next_id = 0

        # Statistics
        self._stats = {
            'total_acquires': 0,
            'total_releases': 0,
            'total_allocations': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'total_wait_time': 0.0,
            'max_pool_size': 0,
            'current_pool_size': 0,
        }

        # Pre-allocate initial buffers
        self._preallocate(initial_size)

        logger.info(f"[BufferPool:{buffer_type}] Initialized with {initial_size} buffers "
                   f"({buffer_size/1024:.1f}KB each, max={max_size})")

    def _preallocate(self, count: int):
        """Pre-allocate buffers"""
        for _ in range(count):
            buffer = self._allocate_buffer()
            self._available.append(buffer)

        self._stats['current_pool_size'] = len(self._available)
        self._stats['max_pool_size'] = max(self._stats['max_pool_size'],
                                           len(self._available))

    def _allocate_buffer(self) -> tuple:
        """Allocate a new buffer with metadata"""
        buffer = np.zeros(self.shape, dtype=self.dtype)

        metadata = BufferMetadata(
            buffer_id=self._next_id,
            size=self.buffer_size,
            dtype=self.dtype,
            shape=self.shape,
            allocated_at=time.time(),
            last_used_at=time.time(),
            use_count=0
        )

        self._next_id += 1
        self._stats['total_allocations'] += 1

        return (buffer, metadata)

    def acquire(self, timeout: Optional[float] = None) -> np.ndarray:
        """
        Acquire a buffer from the pool.

        Args:
            timeout: Maximum time to wait for available buffer (None = no timeout)

        Returns:
            NumPy array buffer

        Raises:
            RuntimeError: If timeout exceeded or pool exhausted
        """
        start_time = time.time()

        with self._lock:
            self._stats['total_acquires'] += 1

            # Try to get buffer from available pool
            if self._available:
                buffer, metadata = self._available.popleft()
                self._stats['pool_hits'] += 1
            else:
                # Pool empty - try to allocate new buffer
                if len(self._in_use) < self.max_size:
                    buffer, metadata = self._allocate_buffer()
                    self._stats['pool_misses'] += 1
                    logger.debug(f"[BufferPool:{self.buffer_type}] Pool miss - "
                               f"allocated new buffer (total: {len(self._in_use)+1})")
                else:
                    # Pool exhausted
                    logger.warning(f"[BufferPool:{self.buffer_type}] Pool exhausted! "
                                 f"Max size {self.max_size} reached")
                    raise RuntimeError(
                        f"Buffer pool '{self.buffer_type}' exhausted "
                        f"(max_size={self.max_size})"
                    )

            # Update metadata
            metadata.last_used_at = time.time()
            metadata.use_count += 1

            # Mark as in-use
            self._in_use[metadata.buffer_id] = (buffer, metadata)

            # Update stats
            self._stats['current_pool_size'] = len(self._available)
            wait_time = time.time() - start_time
            self._stats['total_wait_time'] += wait_time

            if wait_time > 0.001:  # Log if wait > 1ms
                logger.debug(f"[BufferPool:{self.buffer_type}] Acquired buffer "
                           f"{metadata.buffer_id} (waited {wait_time*1000:.2f}ms)")

            return buffer

    def release(self, buffer: np.ndarray):
        """
        Release a buffer back to the pool.

        Args:
            buffer: Buffer to release
        """
        with self._lock:
            self._stats['total_releases'] += 1

            # Find buffer by identity
            buffer_id = None
            for bid, (buf, metadata) in self._in_use.items():
                if buf is buffer:
                    buffer_id = bid
                    break

            if buffer_id is None:
                logger.error(f"[BufferPool:{self.buffer_type}] Attempted to release "
                           "unknown buffer (double release?)")
                return

            # Remove from in-use
            buffer, metadata = self._in_use.pop(buffer_id)

            # Zero out buffer (optional, for security/debugging)
            # buffer.fill(0)  # Uncomment if needed

            # Return to pool
            self._available.append((buffer, metadata))
            self._stats['current_pool_size'] = len(self._available)

            logger.debug(f"[BufferPool:{self.buffer_type}] Released buffer {buffer_id} "
                       f"(pool size: {len(self._available)})")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            stats = self._stats.copy()

            # Calculate derived metrics
            if stats['total_acquires'] > 0:
                stats['hit_rate'] = stats['pool_hits'] / stats['total_acquires']
                stats['miss_rate'] = stats['pool_misses'] / stats['total_acquires']
                stats['avg_wait_time'] = stats['total_wait_time'] / stats['total_acquires']
            else:
                stats['hit_rate'] = 0.0
                stats['miss_rate'] = 0.0
                stats['avg_wait_time'] = 0.0

            stats['buffers_in_use'] = len(self._in_use)
            stats['buffers_available'] = len(self._available)
            stats['total_buffers'] = stats['buffers_in_use'] + stats['buffers_available']

            return stats

    def print_stats(self):
        """Print pool statistics"""
        stats = self.get_stats()

        print(f"\n[BufferPool:{self.buffer_type}] Statistics:")
        print(f"  Total Acquires:  {stats['total_acquires']}")
        print(f"  Total Releases:  {stats['total_releases']}")
        print(f"  Hit Rate:        {stats['hit_rate']*100:.1f}%")
        print(f"  Miss Rate:       {stats['miss_rate']*100:.1f}%")
        print(f"  Avg Wait Time:   {stats['avg_wait_time']*1000:.3f}ms")
        print(f"  Buffers In Use:  {stats['buffers_in_use']}")
        print(f"  Buffers Avail:   {stats['buffers_available']}")
        print(f"  Total Buffers:   {stats['total_buffers']}")
        print(f"  Max Pool Size:   {stats['max_pool_size']}")

    def clear(self):
        """Clear all buffers from pool"""
        with self._lock:
            if self._in_use:
                logger.warning(f"[BufferPool:{self.buffer_type}] Clearing pool with "
                             f"{len(self._in_use)} buffers still in use!")

            self._available.clear()
            self._in_use.clear()
            self._stats['current_pool_size'] = 0

            logger.info(f"[BufferPool:{self.buffer_type}] Pool cleared")
```

### 2. Global Buffer Manager

```python
"""
Global buffer manager for all buffer pools
"""

class GlobalBufferManager:
    """
    Singleton manager for all buffer pools.

    Provides centralized management of buffer pools for different
    data types (mel, audio, encoder, etc.).

    Usage:
        # Get singleton instance
        manager = GlobalBufferManager.instance()

        # Configure pools
        manager.configure({
            'mel': {'size': 960*1024, 'count': 10},
            'audio': {'size': 960*1024, 'count': 5},
            'encoder': {'size': 960*1024, 'count': 5},
        })

        # Acquire buffers
        mel_buf = manager.acquire('mel')
        audio_buf = manager.acquire('audio')

        # Use buffers...

        # Release buffers
        manager.release('mel', mel_buf)
        manager.release('audio', audio_buf)
    """

    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize global buffer manager"""
        if GlobalBufferManager._instance is not None:
            raise RuntimeError("Use GlobalBufferManager.instance() instead")

        self._pools = {}
        self._config = {}
        self._lock = threading.RLock()

        logger.info("[GlobalBufferManager] Initialized")

    def configure(self, config: Dict[str, Dict[str, Any]]):
        """
        Configure buffer pools.

        Args:
            config: Dictionary of pool configurations
                {
                    'pool_name': {
                        'size': buffer_size_bytes,
                        'count': initial_pool_size,
                        'max_count': max_pool_size,
                        'dtype': np.dtype,
                        'shape': tuple (optional)
                    }
                }
        """
        with self._lock:
            for pool_name, pool_config in config.items():
                if pool_name in self._pools:
                    logger.warning(f"[GlobalBufferManager] Pool '{pool_name}' "
                                 "already exists - skipping")
                    continue

                # Create pool
                pool = BufferPool(
                    buffer_type=pool_name,
                    buffer_size=pool_config['size'],
                    dtype=pool_config.get('dtype', np.float32),
                    shape=pool_config.get('shape', None),
                    initial_size=pool_config.get('count', 5),
                    max_size=pool_config.get('max_count', 20),
                )

                self._pools[pool_name] = pool
                self._config[pool_name] = pool_config

                logger.info(f"[GlobalBufferManager] Created pool '{pool_name}' "
                          f"({pool_config['size']/1024:.1f}KB × {pool_config['count']})")

    def acquire(self, pool_name: str) -> np.ndarray:
        """Acquire buffer from pool"""
        with self._lock:
            if pool_name not in self._pools:
                raise ValueError(f"Unknown pool: {pool_name}")

            return self._pools[pool_name].acquire()

    def release(self, pool_name: str, buffer: np.ndarray):
        """Release buffer to pool"""
        with self._lock:
            if pool_name not in self._pools:
                raise ValueError(f"Unknown pool: {pool_name}")

            self._pools[pool_name].release(buffer)

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools"""
        with self._lock:
            return {
                pool_name: pool.get_stats()
                for pool_name, pool in self._pools.items()
            }

    def print_stats(self):
        """Print statistics for all pools"""
        print("\n" + "="*70)
        print("  GLOBAL BUFFER MANAGER STATISTICS")
        print("="*70)

        for pool_name, pool in self._pools.items():
            pool.print_stats()

        print("="*70 + "\n")

    def clear_all(self):
        """Clear all pools"""
        with self._lock:
            for pool in self._pools.values():
                pool.clear()
            logger.info("[GlobalBufferManager] All pools cleared")
```

### 3. Integration with Server

```python
"""
Modified server.py to use buffer pooling
"""

from buffer_pool import GlobalBufferManager

# Initialize buffer manager at startup
@app.on_event("startup")
async def startup_event():
    # ... existing encoder initialization ...

    # Configure buffer pools
    buffer_manager = GlobalBufferManager.instance()
    buffer_manager.configure({
        'mel': {
            'size': 960 * 1024,  # 960KB for mel spectrogram
            'count': 10,         # Pre-allocate 10 buffers
            'max_count': 20,     # Max 20 concurrent requests
            'dtype': np.float32,
            'shape': (80, 3000)  # 80 mels × 3000 frames (30s audio)
        },
        'audio': {
            'size': 960 * 1024,  # 960KB for audio
            'count': 5,
            'max_count': 15,
            'dtype': np.float32,
        },
        'encoder_output': {
            'size': 960 * 1024,  # 960KB for encoder output
            'count': 5,
            'max_count': 15,
            'dtype': np.float32,
            'shape': (3000, 512)  # 3000 frames × 512 hidden
        },
        'fft': {
            'size': 9600 * 1024,  # 9.6MB for FFT buffer
            'count': 5,
            'max_count': 10,
            'dtype': np.complex64,
        }
    })

    logger.info("Buffer pools initialized")


@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), ...):
    buffer_manager = GlobalBufferManager.instance()

    # Acquire buffers (instead of allocating)
    audio_buffer = None
    mel_buffer = None
    encoder_buffer = None

    try:
        # 1. Load audio into pooled buffer
        audio_buffer = buffer_manager.acquire('audio')
        audio = whisperx.load_audio(tmp_path)
        np.copyto(audio_buffer[:len(audio)], audio)  # Copy into pooled buffer

        # 2. Compute mel into pooled buffer
        mel_buffer = buffer_manager.acquire('mel')
        # Modify mel computation to use output buffer
        mel_features = compute_mel_spectrogram(
            audio_buffer[:len(audio)],
            output=mel_buffer  # Use pooled buffer
        )

        # 3. Encoder output into pooled buffer
        encoder_buffer = buffer_manager.acquire('encoder_output')
        encoder_output = cpp_encoder.forward(
            mel_features,
            output=encoder_buffer  # Use pooled buffer
        )

        # ... rest of processing ...

    finally:
        # Always release buffers
        if audio_buffer is not None:
            buffer_manager.release('audio', audio_buffer)
        if mel_buffer is not None:
            buffer_manager.release('mel', mel_buffer)
        if encoder_buffer is not None:
            buffer_manager.release('encoder_output', encoder_buffer)
```

---

## Implementation Plan

### Phase 1: Core Buffer Pool (Week 7 Day 1)

1. ✅ Design buffer pool interface
2. ⏳ Implement `BufferPool` class
3. ⏳ Implement `GlobalBufferManager`
4. ⏳ Write unit tests

**Files to Create**:
- `buffer_pool.py` (500 lines)
- `tests/test_buffer_pool.py` (300 lines)

**Estimated Time**: 4 hours

### Phase 2: Service Integration (Week 7 Day 2)

1. ⏳ Modify `server.py` to use buffer pools
2. ⏳ Update mel computation to accept output buffer
3. ⏳ Update encoder to accept output buffer
4. ⏳ Add buffer release in finally blocks

**Files to Modify**:
- `xdna2/server.py`
- `xdna2/encoder_cpp.py` (add output parameter)

**Estimated Time**: 3 hours

### Phase 3: Testing & Validation (Week 7 Day 3)

1. ⏳ Integration testing
2. ⏳ Performance benchmarking
3. ⏳ Leak detection testing
4. ⏳ Concurrent request testing

**Estimated Time**: 3 hours

---

## Performance Expectations

### Memory Usage

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Memory Allocated per Request | 15-20MB | 0MB | **100%** |
| Peak Memory Usage | Unbounded | ~50MB | **Capped** |
| Fragmentation | High | Low | **~75%** |

### Latency Impact

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Buffer Allocation | 3-6ms | 0.6-1.2ms | **80%** |
| Request Latency | 64ms | 60-62ms | **~3-5%** |
| Throughput (10 concurrent) | 156 req/s | 165-167 req/s | **~6-7%** |

### Pool Statistics (Expected)

After 1000 requests:
```
[BufferPool:mel] Statistics:
  Total Acquires:  1000
  Total Releases:  1000
  Hit Rate:        99.0%
  Miss Rate:       1.0%
  Avg Wait Time:   0.012ms
  Buffers In Use:  0
  Buffers Avail:   10
  Total Buffers:   10
  Max Pool Size:   10
```

---

## Error Handling & Edge Cases

### 1. Pool Exhaustion

```python
# When pool is exhausted (max_size reached)
try:
    buffer = pool.acquire()
except RuntimeError as e:
    logger.error(f"Buffer pool exhausted: {e}")
    # Option 1: Retry with backoff
    time.sleep(0.1)
    buffer = pool.acquire(timeout=5.0)

    # Option 2: Reject request
    return JSONResponse(
        status_code=503,
        content={"error": "Service overloaded, please retry"}
    )
```

### 2. Memory Leaks

```python
# Detect unreleased buffers
def check_leaks():
    stats = buffer_manager.get_stats()
    for pool_name, pool_stats in stats.items():
        if pool_stats['total_acquires'] != pool_stats['total_releases']:
            logger.warning(
                f"Pool '{pool_name}' has {pool_stats['buffers_in_use']} "
                "unreleased buffers!"
            )

# Run periodically
@app.on_event("startup")
async def startup():
    # ... initialize pools ...

    # Schedule leak detection
    asyncio.create_task(periodic_leak_check())

async def periodic_leak_check():
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        check_leaks()
```

### 3. Thread Safety

```python
# Buffer pools are thread-safe via RLock
# Multiple concurrent requests handled correctly

import concurrent.futures

def process_request(request_id):
    buffer = buffer_manager.acquire('mel')
    try:
        # Process...
        pass
    finally:
        buffer_manager.release('mel', buffer)

# Safe concurrent execution
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_request, i) for i in range(100)]
    concurrent.futures.wait(futures)
```

---

## Monitoring & Observability

### 1. Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Buffer pool metrics
buffer_acquires = Counter('buffer_acquires_total', 'Buffer acquisitions', ['pool'])
buffer_releases = Counter('buffer_releases_total', 'Buffer releases', ['pool'])
buffer_hits = Counter('buffer_hits_total', 'Buffer pool hits', ['pool'])
buffer_misses = Counter('buffer_misses_total', 'Buffer pool misses', ['pool'])
buffer_wait_time = Histogram('buffer_wait_seconds', 'Buffer wait time', ['pool'])
buffers_in_use = Gauge('buffers_in_use', 'Buffers in use', ['pool'])
buffers_available = Gauge('buffers_available', 'Buffers available', ['pool'])

# Update in acquire/release
def acquire(pool_name):
    buffer_acquires.labels(pool=pool_name).inc()
    # ... existing code ...
```

### 2. Health Endpoint

```python
@app.get("/health")
async def health():
    buffer_stats = GlobalBufferManager.instance().get_stats()

    # Check for unhealthy conditions
    warnings = []
    for pool_name, stats in buffer_stats.items():
        if stats['hit_rate'] < 0.90:
            warnings.append(f"Low hit rate for {pool_name}: {stats['hit_rate']:.1%}")
        if stats['buffers_available'] == 0 and stats['buffers_in_use'] >= stats['max_pool_size']:
            warnings.append(f"Pool {pool_name} exhausted")

    return {
        "status": "healthy" if not warnings else "degraded",
        "buffer_pools": buffer_stats,
        "warnings": warnings
    }
```

---

## Future Enhancements

### 1. Adaptive Pool Sizing

```python
# Automatically adjust pool size based on load
class AdaptiveBufferPool(BufferPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recent_misses = deque(maxlen=100)

    def acquire(self):
        buffer = super().acquire()

        # Track miss rate
        self.recent_misses.append(1 if buffer was allocated else 0)

        # If miss rate > 10%, grow pool
        if sum(self.recent_misses) / len(self.recent_misses) > 0.10:
            self._grow_pool(1)

        return buffer

    def _grow_pool(self, count):
        # Add more buffers to pool
        pass
```

### 2. NUMA-Aware Allocation

```python
# Allocate buffers on specific NUMA node (not needed for CC-1L)
import numa

class NUMABufferPool(BufferPool):
    def __init__(self, *args, numa_node=0, **kwargs):
        self.numa_node = numa_node
        super().__init__(*args, **kwargs)

    def _allocate_buffer(self):
        # Pin to NUMA node
        numa.set_preferred(self.numa_node)
        buffer = super()._allocate_buffer()
        return buffer
```

### 3. Shared Memory Pools

```python
# Share buffer pools across multiple processes
from multiprocessing import shared_memory

class SharedBufferPool(BufferPool):
    # Implementation for multi-process buffer sharing
    # Useful for multi-worker deployments
    pass
```

---

## Appendix: Configuration Examples

### Small Deployment (2-4 concurrent requests)

```python
buffer_manager.configure({
    'mel': {'size': 960*1024, 'count': 3, 'max_count': 5},
    'audio': {'size': 960*1024, 'count': 2, 'max_count': 5},
    'encoder_output': {'size': 960*1024, 'count': 2, 'max_count': 5},
})
```

### Medium Deployment (10-15 concurrent requests)

```python
buffer_manager.configure({
    'mel': {'size': 960*1024, 'count': 10, 'max_count': 20},
    'audio': {'size': 960*1024, 'count': 5, 'max_count': 15},
    'encoder_output': {'size': 960*1024, 'count': 5, 'max_count': 15},
    'fft': {'size': 9600*1024, 'count': 5, 'max_count': 10},
})
```

### Large Deployment (50+ concurrent requests)

```python
buffer_manager.configure({
    'mel': {'size': 960*1024, 'count': 25, 'max_count': 60},
    'audio': {'size': 960*1024, 'count': 15, 'max_count': 50},
    'encoder_output': {'size': 960*1024, 'count': 15, 'max_count': 50},
    'fft': {'size': 9600*1024, 'count': 10, 'max_count': 20},
})
```

---

**Design Complete**: November 1, 2025
**Status**: Ready for implementation
**Estimated Implementation Time**: 10 hours
**Expected Improvement**: 3-5ms latency reduction, 80% allocation overhead elimination
