#!/usr/bin/env python3
"""
Request Queue - Priority Queue for Transcription Requests

Provides priority-based request queueing with timeout handling,
backpressure management, and statistics tracking for the
multi-stream transcription pipeline.

Features:
- Priority-based FIFO scheduling
- Timeout handling for queue operations
- Queue size limits and backpressure
- Comprehensive statistics tracking
- Thread-safe async operations

Author: CC-1L Multi-Stream Pipeline Team
Date: November 1, 2025
Status: Week 9 Implementation
"""

import asyncio
import time
import logging
import uuid
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QueuedRequest:
    """
    A queued transcription request with metadata.

    Attributes:
        request_id: Unique identifier for request tracking
        audio_data: Raw audio bytes
        options: Transcription options (diarize, min/max speakers, etc.)
        enqueued_at: Timestamp when request was queued
        priority: Request priority (higher = more urgent, default=0)
        metadata: Additional metadata for tracking
    """
    request_id: str
    audio_data: bytes
    options: Dict[str, Any]
    enqueued_at: float = field(default_factory=time.time)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def age_seconds(self) -> float:
        """Calculate how long this request has been queued"""
        return time.time() - self.enqueued_at

    def __lt__(self, other):
        """Compare by priority (for PriorityQueue), then by enqueue time (FIFO)"""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.enqueued_at < other.enqueued_at  # FIFO within same priority


class RequestQueue:
    """
    Thread-safe priority queue for transcription requests.

    Features:
    - Priority-based scheduling (higher priority processed first)
    - FIFO ordering within same priority level
    - Timeout handling for enqueue operations
    - Queue size limits with backpressure
    - Comprehensive statistics tracking
    - Request aging and wait time monitoring

    Usage:
        queue = RequestQueue(max_queue_size=100)

        # Enqueue request
        request = QueuedRequest(
            request_id=str(uuid.uuid4()),
            audio_data=audio_bytes,
            options={'diarize': True},
            priority=0
        )

        success = await queue.enqueue(request, timeout=5.0)
        if not success:
            raise RuntimeError("Queue full")

        # Dequeue request
        request = await queue.dequeue()

        # Get statistics
        stats = queue.get_stats()

    Thread Safety:
        All operations are thread-safe via asyncio primitives.
        Multiple coroutines can safely enqueue/dequeue concurrently.
    """

    def __init__(self, max_queue_size: int = 100, timeout_warning_ms: float = 100.0):
        """
        Initialize request queue.

        Args:
            max_queue_size: Maximum number of queued requests
            timeout_warning_ms: Log warning if wait time exceeds this (ms)

        Raises:
            ValueError: If max_queue_size < 1
        """
        if max_queue_size < 1:
            raise ValueError(f"max_queue_size must be >= 1, got {max_queue_size}")

        self.max_queue_size = max_queue_size
        self.timeout_warning_ms = timeout_warning_ms

        # asyncio PriorityQueue for request storage
        self._queue = asyncio.PriorityQueue(maxsize=max_queue_size)

        # Statistics tracking
        self._stats = {
            'total_enqueued': 0,
            'total_dequeued': 0,
            'total_dropped': 0,
            'total_timeout': 0,
            'total_wait_time': 0.0,
            'max_wait_time': 0.0,
            'min_wait_time': float('inf'),
        }

        # Lock for stats updates
        self._stats_lock = asyncio.Lock()

        logger.info(
            f"[RequestQueue] Initialized (max_size={max_queue_size}, "
            f"timeout_warning={timeout_warning_ms}ms)"
        )

    async def enqueue(
        self,
        request: QueuedRequest,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Enqueue a request with optional timeout.

        If the queue is full, this will wait up to `timeout` seconds
        for space to become available. If timeout is None, waits indefinitely.

        Args:
            request: Request to enqueue
            timeout: Maximum time to wait for queue space (seconds)

        Returns:
            True if enqueued successfully, False if dropped due to timeout

        Raises:
            ValueError: If request is invalid
        """
        if not isinstance(request, QueuedRequest):
            raise ValueError(f"Expected QueuedRequest, got {type(request)}")

        if not request.audio_data:
            raise ValueError("Request has no audio data")

        try:
            # Try to enqueue (blocks if queue full)
            if timeout is not None:
                await asyncio.wait_for(
                    self._queue.put((request.priority, request.enqueued_at, request)),
                    timeout=timeout
                )
            else:
                await self._queue.put((request.priority, request.enqueued_at, request))

            # Update stats
            async with self._stats_lock:
                self._stats['total_enqueued'] += 1

            logger.debug(
                f"[RequestQueue] Enqueued request {request.request_id} "
                f"(priority={request.priority}, queue_size={self._queue.qsize()})"
            )

            return True

        except asyncio.TimeoutError:
            # Queue full, timeout exceeded
            async with self._stats_lock:
                self._stats['total_dropped'] += 1
                self._stats['total_timeout'] += 1

            logger.warning(
                f"[RequestQueue] Dropped request {request.request_id} - "
                f"queue full (timeout={timeout}s, max_size={self.max_queue_size})"
            )

            return False

    async def dequeue(self, timeout: Optional[float] = None) -> QueuedRequest:
        """
        Dequeue the highest priority request.

        This will wait for a request to become available. Within the same
        priority level, requests are dequeued in FIFO order.

        Args:
            timeout: Maximum time to wait for a request (seconds)

        Returns:
            Next request to process

        Raises:
            asyncio.TimeoutError: If timeout exceeded with no requests
        """
        try:
            # Get next request (blocks if queue empty)
            if timeout is not None:
                _, _, request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout
                )
            else:
                _, _, request = await self._queue.get()

            # Calculate wait time
            wait_time = time.time() - request.enqueued_at

            # Update stats
            async with self._stats_lock:
                self._stats['total_dequeued'] += 1
                self._stats['total_wait_time'] += wait_time
                self._stats['max_wait_time'] = max(self._stats['max_wait_time'], wait_time)
                if wait_time < self._stats['min_wait_time']:
                    self._stats['min_wait_time'] = wait_time

            # Log if wait time exceeds warning threshold
            if wait_time * 1000 > self.timeout_warning_ms:
                logger.warning(
                    f"[RequestQueue] Request {request.request_id} waited "
                    f"{wait_time*1000:.1f}ms in queue (threshold={self.timeout_warning_ms}ms)"
                )
            else:
                logger.debug(
                    f"[RequestQueue] Dequeued request {request.request_id} "
                    f"(wait={wait_time*1000:.1f}ms, queue_size={self._queue.qsize()})"
                )

            return request

        except asyncio.TimeoutError:
            logger.debug(
                f"[RequestQueue] Dequeue timeout after {timeout}s "
                f"(queue_size={self._queue.qsize()})"
            )
            raise

    def qsize(self) -> int:
        """
        Get current queue size (number of pending requests).

        Returns:
            Number of requests currently in queue
        """
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()

    def is_full(self) -> bool:
        """Check if queue is at max capacity"""
        return self._queue.full()

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Dictionary with statistics including:
            - total_enqueued: Total requests enqueued
            - total_dequeued: Total requests dequeued
            - total_dropped: Total requests dropped (queue full)
            - current_size: Current queue depth
            - avg_wait_time: Average time requests wait in queue
            - max_wait_time: Maximum wait time seen
            - min_wait_time: Minimum wait time seen
            - utilization: Queue utilization percentage
        """
        async with self._stats_lock:
            stats = self._stats.copy()

        # Add current state
        stats['current_size'] = self.qsize()
        stats['max_queue_size'] = self.max_queue_size
        stats['is_full'] = self.is_full()
        stats['is_empty'] = self.is_empty()

        # Calculate derived metrics
        if stats['total_dequeued'] > 0:
            stats['avg_wait_time'] = stats['total_wait_time'] / stats['total_dequeued']
        else:
            stats['avg_wait_time'] = 0.0

        # Queue utilization
        if self.max_queue_size > 0:
            stats['utilization'] = stats['current_size'] / self.max_queue_size
        else:
            stats['utilization'] = 0.0

        # Drop rate
        if stats['total_enqueued'] > 0:
            stats['drop_rate'] = stats['total_dropped'] / stats['total_enqueued']
        else:
            stats['drop_rate'] = 0.0

        return stats

    async def print_stats(self):
        """Print queue statistics to console"""
        stats = await self.get_stats()

        print("\n[RequestQueue] Statistics:")
        print(f"  Total Enqueued:  {stats['total_enqueued']}")
        print(f"  Total Dequeued:  {stats['total_dequeued']}")
        print(f"  Total Dropped:   {stats['total_dropped']}")
        print(f"  Drop Rate:       {stats['drop_rate']*100:.1f}%")
        print(f"  Current Size:    {stats['current_size']}/{stats['max_queue_size']}")
        print(f"  Utilization:     {stats['utilization']*100:.1f}%")
        print(f"  Avg Wait Time:   {stats['avg_wait_time']*1000:.2f}ms")
        print(f"  Max Wait Time:   {stats['max_wait_time']*1000:.2f}ms")

        if stats['min_wait_time'] != float('inf'):
            print(f"  Min Wait Time:   {stats['min_wait_time']*1000:.2f}ms")

    async def clear(self):
        """Clear all pending requests from queue"""
        count = 0
        while not self._queue.empty():
            try:
                _ = self._queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break

        logger.info(f"[RequestQueue] Cleared {count} pending requests")
        return count


async def main():
    """Demonstration of request queue functionality"""
    print("Request Queue - Demonstration\n")

    # Create queue
    print("[Demo] Creating request queue...")
    queue = RequestQueue(max_queue_size=5, timeout_warning_ms=50.0)

    # Enqueue some requests
    print("\n[Demo] Enqueuing requests...")
    requests = []
    for i in range(3):
        request = QueuedRequest(
            request_id=f"req-{i}",
            audio_data=b"fake audio data",
            options={'diarize': False},
            priority=i % 2  # Alternate priorities
        )
        success = await queue.enqueue(request, timeout=1.0)
        print(f"  Request {i}: Enqueued={success}, Priority={request.priority}")
        requests.append(request)

    # Wait a bit to simulate queue wait time
    await asyncio.sleep(0.1)

    # Dequeue requests
    print("\n[Demo] Dequeuing requests...")
    while not queue.is_empty():
        request = await queue.dequeue()
        print(f"  Dequeued: {request.request_id} (priority={request.priority}, "
              f"wait={request.age_seconds()*1000:.1f}ms)")

    # Print statistics
    print("\n[Demo] Final statistics:")
    await queue.print_stats()

    print("\n✅ Demo complete!")
    return 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(main()))
