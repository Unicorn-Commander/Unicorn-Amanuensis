#!/usr/bin/env python3
"""
NPU Buffer Pool - Pre-allocated buffer management for zero-copy DMA

This module provides efficient buffer pooling to reduce DMA overhead:
- Pre-allocates buffers at initialization
- Reuses buffers across multiple kernel invocations
- Aligns buffers to cache lines (64 bytes)
- Supports buffer pinning for NPU access
- Minimizes allocate/deallocate overhead

Usage:
    pool = NPUBufferPool(device, num_buffers=8)
    buf = pool.allocate_buffer("attn_input", 12288, kernel.group_id(3))
    # ... use buffer multiple times ...
    pool.reset()  # Optionally reset for new workload
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
from typing import Dict, Tuple


class NPUBufferPool:
    """Pre-allocated buffer pool for efficient NPU DMA operations"""

    def __init__(self, device: xrt.device, num_buffers: int = 8, verbose: bool = False):
        """
        Initialize buffer pool

        Args:
            device: XRT device handle
            num_buffers: Maximum number of concurrent buffers to support
            verbose: Print allocation details
        """
        self.device = device
        self.num_buffers = num_buffers
        self.verbose = verbose

        # Buffer storage: {name: (xrt.bo, size, group_id)}
        self.buffers: Dict[str, Tuple[xrt.bo, int, int]] = {}

        # Statistics
        self.stats = {
            'allocations': 0,
            'reuses': 0,
            'total_bytes_allocated': 0,
            'peak_buffers': 0
        }

        if self.verbose:
            print(f"NPU Buffer Pool initialized (capacity: {num_buffers} buffers)")

    def _align_size(self, size: int, alignment: int = 64) -> int:
        """
        Align buffer size to cache line boundary

        Args:
            size: Requested size in bytes
            alignment: Alignment boundary (default 64 bytes for cache lines)

        Returns:
            Aligned size
        """
        return ((size + alignment - 1) // alignment) * alignment

    def allocate_buffer(self, name: str, size: int, group_id: int,
                       flags: xrt.bo.flags = xrt.bo.flags.host_only,
                       force_recreate: bool = False) -> xrt.bo:
        """
        Allocate or reuse buffer

        Args:
            name: Buffer identifier (used for reuse)
            size: Buffer size in bytes
            group_id: Kernel group ID for buffer placement
            flags: XRT buffer flags (default: host_only)
            force_recreate: Force buffer recreation even if exists

        Returns:
            XRT buffer object
        """
        # Align size to cache line
        aligned_size = self._align_size(size)

        # Check if buffer exists and can be reused
        if name in self.buffers and not force_recreate:
            existing_bo, existing_size, existing_gid = self.buffers[name]

            # Reuse if size and group_id match
            if existing_size >= aligned_size and existing_gid == group_id:
                self.stats['reuses'] += 1
                if self.verbose:
                    print(f"  Buffer '{name}' reused ({existing_size} bytes)")
                return existing_bo
            else:
                # Size mismatch - need to recreate
                if self.verbose:
                    print(f"  Buffer '{name}' resizing: {existing_size} -> {aligned_size} bytes")
                del self.buffers[name]

        # Allocate new buffer
        bo = xrt.bo(self.device, aligned_size, flags, group_id)
        self.buffers[name] = (bo, aligned_size, group_id)

        # Update statistics
        self.stats['allocations'] += 1
        self.stats['total_bytes_allocated'] += aligned_size
        self.stats['peak_buffers'] = max(self.stats['peak_buffers'], len(self.buffers))

        if self.verbose:
            print(f"  Buffer '{name}' allocated ({aligned_size} bytes, group {group_id})")

        return bo

    def get_buffer(self, name: str) -> xrt.bo:
        """
        Get existing buffer by name

        Args:
            name: Buffer identifier

        Returns:
            XRT buffer object

        Raises:
            KeyError: If buffer doesn't exist
        """
        if name not in self.buffers:
            raise KeyError(f"Buffer '{name}' not found in pool")

        bo, _, _ = self.buffers[name]
        return bo

    def has_buffer(self, name: str) -> bool:
        """Check if buffer exists in pool"""
        return name in self.buffers

    def remove_buffer(self, name: str):
        """Remove buffer from pool"""
        if name in self.buffers:
            bo, size, _ = self.buffers[name]
            del self.buffers[name]
            if self.verbose:
                print(f"  Buffer '{name}' removed ({size} bytes)")

    def reset(self):
        """Clear all buffers (useful between workloads)"""
        num_buffers = len(self.buffers)
        total_bytes = sum(size for _, size, _ in self.buffers.values())

        self.buffers.clear()

        if self.verbose:
            print(f"  Buffer pool reset ({num_buffers} buffers, {total_bytes} bytes freed)")

    def get_statistics(self) -> Dict[str, int]:
        """
        Get buffer pool statistics

        Returns:
            Dictionary with allocation statistics
        """
        stats = self.stats.copy()
        stats['current_buffers'] = len(self.buffers)
        stats['current_bytes'] = sum(size for _, size, _ in self.buffers.values())

        if stats['allocations'] > 0:
            stats['reuse_ratio'] = stats['reuses'] / stats['allocations']
        else:
            stats['reuse_ratio'] = 0.0

        return stats

    def print_statistics(self):
        """Print buffer pool statistics"""
        stats = self.get_statistics()

        print("\nNPU Buffer Pool Statistics:")
        print("=" * 50)
        print(f"  Total allocations:    {stats['allocations']}")
        print(f"  Buffer reuses:        {stats['reuses']}")
        print(f"  Reuse ratio:          {stats['reuse_ratio']:.2%}")
        print(f"  Current buffers:      {stats['current_buffers']}")
        print(f"  Current bytes:        {stats['current_bytes']:,}")
        print(f"  Peak buffers:         {stats['peak_buffers']}")
        print(f"  Total allocated:      {stats['total_bytes_allocated']:,} bytes")
        print("=" * 50)


class BufferCache:
    """
    High-level buffer cache for reusing buffers across frames

    This extends NPUBufferPool with frame-level caching semantics.
    Useful for processing multiple audio frames with the same buffer layout.
    """

    def __init__(self, pool: NPUBufferPool):
        """
        Initialize buffer cache

        Args:
            pool: Underlying NPU buffer pool
        """
        self.pool = pool
        self.cache_hits = 0
        self.cache_misses = 0

    def get_or_create(self, key: str, size: int, group_id: int,
                     flags: xrt.bo.flags = xrt.bo.flags.host_only) -> xrt.bo:
        """
        Get cached buffer or create new one

        Args:
            key: Cache key (unique identifier)
            size: Buffer size in bytes
            group_id: Kernel group ID
            flags: XRT buffer flags

        Returns:
            XRT buffer object
        """
        if self.pool.has_buffer(key):
            self.cache_hits += 1
            return self.pool.get_buffer(key)
        else:
            self.cache_misses += 1
            return self.pool.allocate_buffer(key, size, group_id, flags)

    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache hit/miss statistics"""
        total = self.cache_hits + self.cache_misses
        if total > 0:
            hit_rate = self.cache_hits / total
        else:
            hit_rate = 0.0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': hit_rate
        }

    def print_cache_stats(self):
        """Print cache statistics"""
        stats = self.get_cache_stats()

        print("\nBuffer Cache Statistics:")
        print("=" * 50)
        print(f"  Cache hits:           {stats['cache_hits']}")
        print(f"  Cache misses:         {stats['cache_misses']}")
        print(f"  Cache hit rate:       {stats['cache_hit_rate']:.2%}")
        print("=" * 50)


# Example usage
if __name__ == "__main__":
    print("NPU Buffer Pool Test")
    print("=" * 70)
    print()

    # Initialize XRT device
    device = xrt.device(0)
    print("Device initialized: /dev/accel/accel0")
    print()

    # Create buffer pool
    pool = NPUBufferPool(device, num_buffers=8, verbose=True)
    print()

    # Simulate buffer allocation pattern
    print("Simulating buffer allocations...")
    print()

    # First allocation
    buf1 = pool.allocate_buffer("input", 12288, 3)
    print()

    # Reuse same buffer
    buf1_reused = pool.allocate_buffer("input", 12288, 3)
    assert buf1 is buf1_reused, "Buffer should be reused"
    print()

    # Allocate different buffer
    buf2 = pool.allocate_buffer("output", 4096, 4)
    print()

    # Get buffer by name
    buf1_get = pool.get_buffer("input")
    assert buf1 is buf1_get, "Should get same buffer"
    print()

    # Print statistics
    pool.print_statistics()
    print()

    # Test buffer cache
    print("\nTesting Buffer Cache...")
    print()
    cache = BufferCache(pool)

    # Cache operations
    b1 = cache.get_or_create("frame_0_input", 12288, 3)
    b2 = cache.get_or_create("frame_0_input", 12288, 3)  # Should hit cache
    b3 = cache.get_or_create("frame_1_input", 12288, 3)  # Should miss

    cache.print_cache_stats()
    print()

    print("Buffer Pool test complete!")
