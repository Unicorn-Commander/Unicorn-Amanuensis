#!/usr/bin/env python3
"""
Pipeline Workers - Multi-Stage Pipeline Worker Pools

Implements worker pools for the 3-stage transcription pipeline:
- Stage 1: Audio Load + Mel Spectrogram (CPU-bound, thread pool)
- Stage 2: NPU Encoder (NPU-bound, single worker for serialization)
- Stage 3: Decoder + Alignment (CPU-bound, process pool for GIL avoidance)

Features:
- Async worker management with ThreadPoolExecutor/ProcessPoolExecutor
- Automatic work routing between stages
- Per-stage statistics and monitoring
- Graceful start/stop with queue draining
- Error handling and retry logic

Author: CC-1L Multi-Stream Pipeline Team
Date: November 1, 2025
Status: Week 9 Implementation
"""

import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, Any, Dict, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    """
    A work item flowing through the pipeline.

    Attributes:
        request_id: Unique identifier for tracking
        data: Stage-specific data payload
        metadata: Additional metadata for tracking
        stage: Current pipeline stage (1, 2, or 3)
        created_at: Timestamp when item was created
    """
    request_id: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage: int = 1
    created_at: float = field(default_factory=time.time)

    def age_seconds(self) -> float:
        """Calculate how long this item has been in the pipeline"""
        return time.time() - self.created_at


class PipelineStage:
    """
    A pipeline stage with worker pool.

    Each stage has:
    - Input queue for receiving work items
    - Worker pool (threads or processes) for parallel processing
    - Processing function to execute
    - Output queue for completed work items
    - Statistics tracking for performance monitoring

    Usage:
        # Create stage
        stage = PipelineStage(
            name="LoadMel",
            process_func=load_and_mel,
            num_workers=4,
            use_processes=False
        )

        # Start workers
        await stage.start()

        # Put work in input queue
        await stage.input_queue.put(work_item)

        # Get result from output queue
        result = await stage.output_queue.get()

        # Stop workers
        await stage.stop()

    Thread Safety:
        All operations are thread-safe via asyncio primitives.
        Multiple workers can process items concurrently.
    """

    def __init__(
        self,
        name: str,
        process_func: Callable[[WorkItem], WorkItem],
        num_workers: int = 4,
        use_processes: bool = False,
        max_queue_size: int = 50,
        worker_timeout: float = 30.0
    ):
        """
        Initialize pipeline stage.

        Args:
            name: Stage name for logging
            process_func: Function to process work items
                         Must accept WorkItem and return WorkItem
            num_workers: Number of worker threads/processes
            use_processes: Use processes instead of threads (for GIL-bound work)
            max_queue_size: Maximum items in input/output queues
            worker_timeout: Timeout for individual work items (seconds)

        Raises:
            ValueError: If num_workers < 1
        """
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")

        self.name = name
        self.process_func = process_func
        self.num_workers = num_workers
        self.use_processes = use_processes
        self.max_queue_size = max_queue_size
        self.worker_timeout = worker_timeout

        # Create executor (threads or processes)
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=num_workers)
            logger.info(f"[{name}] Using ProcessPoolExecutor with {num_workers} workers")
        else:
            self.executor = ThreadPoolExecutor(max_workers=num_workers)
            logger.info(f"[{name}] Using ThreadPoolExecutor with {num_workers} workers")

        # Input and output queues
        self.input_queue = asyncio.Queue(maxsize=max_queue_size)
        self.output_queue = asyncio.Queue(maxsize=max_queue_size)

        # Worker tasks
        self._workers: List[asyncio.Task] = []
        self._running = False

        # Statistics
        self._stats = {
            'total_processed': 0,
            'total_errors': 0,
            'total_timeouts': 0,
            'total_time': 0.0,
            'max_time': 0.0,
            'min_time': float('inf'),
        }
        self._stats_lock = asyncio.Lock()

        logger.info(
            f"[{name}] Initialized with {num_workers} workers "
            f"({'processes' if use_processes else 'threads'})"
        )

    async def start(self):
        """Start all worker tasks"""
        if self._running:
            logger.warning(f"[{self.name}] Already running")
            return

        self._running = True

        # Create worker tasks
        self._workers = [
            asyncio.create_task(self._worker(worker_id))
            for worker_id in range(self.num_workers)
        ]

        logger.info(f"[{self.name}] Started {len(self._workers)} workers")

    async def stop(self, drain_queues: bool = True, timeout: float = 10.0):
        """
        Stop all worker tasks.

        Args:
            drain_queues: Wait for input queue to drain before stopping
            timeout: Maximum time to wait for queue draining (seconds)
        """
        if not self._running:
            logger.warning(f"[{self.name}] Not running")
            return

        self._running = False

        # Optionally drain input queue
        if drain_queues and not self.input_queue.empty():
            logger.info(
                f"[{self.name}] Draining input queue "
                f"({self.input_queue.qsize()} items)..."
            )

            drain_start = time.time()
            while not self.input_queue.empty():
                if time.time() - drain_start > timeout:
                    logger.warning(
                        f"[{self.name}] Queue drain timeout after {timeout}s "
                        f"({self.input_queue.qsize()} items remaining)"
                    )
                    break

                await asyncio.sleep(0.1)

        # Cancel all workers
        logger.info(f"[{self.name}] Cancelling {len(self._workers)} workers...")
        for worker in self._workers:
            worker.cancel()

        # Wait for cancellation with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"[{self.name}] Worker cancellation timeout")

        # Shutdown executor
        self.executor.shutdown(wait=True, cancel_futures=not drain_queues)

        logger.info(f"[{self.name}] Stopped all workers")

    async def _worker(self, worker_id: int):
        """
        Worker coroutine that processes items from input queue.

        Args:
            worker_id: Worker identifier for logging
        """
        logger.info(f"[{self.name}] Worker {worker_id} started")

        while self._running:
            try:
                # Get work item from input queue (with timeout to check running flag)
                try:
                    item = await asyncio.wait_for(
                        self.input_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # No work available, check if still running
                    continue

                # Process item in executor to avoid blocking event loop
                start_time = time.perf_counter()

                try:
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            self.executor,
                            self.process_func,
                            item
                        ),
                        timeout=self.worker_timeout
                    )

                    elapsed = time.perf_counter() - start_time

                    # Put result in output queue
                    await self.output_queue.put(result)

                    # Update stats
                    async with self._stats_lock:
                        self._stats['total_processed'] += 1
                        self._stats['total_time'] += elapsed
                        self._stats['max_time'] = max(self._stats['max_time'], elapsed)
                        if elapsed < self._stats['min_time']:
                            self._stats['min_time'] = elapsed

                    logger.debug(
                        f"[{self.name}] Worker {worker_id} processed {item.request_id} "
                        f"in {elapsed*1000:.1f}ms"
                    )

                except asyncio.TimeoutError:
                    # Worker timeout
                    elapsed = time.perf_counter() - start_time

                    async with self._stats_lock:
                        self._stats['total_timeouts'] += 1

                    logger.error(
                        f"[{self.name}] Worker {worker_id} timeout after {elapsed:.1f}s "
                        f"processing {item.request_id}"
                    )

                    # Put error result in output queue
                    error_result = WorkItem(
                        request_id=item.request_id,
                        data={'error': f'Timeout after {elapsed:.1f}s'},
                        metadata={**item.metadata, 'timeout': True, 'stage': self.name},
                        stage=item.stage
                    )
                    await self.output_queue.put(error_result)

                except Exception as e:
                    # Processing error
                    elapsed = time.perf_counter() - start_time

                    async with self._stats_lock:
                        self._stats['total_errors'] += 1

                    logger.error(
                        f"[{self.name}] Worker {worker_id} error processing {item.request_id}: {e}",
                        exc_info=True
                    )

                    # Put error result in output queue
                    error_result = WorkItem(
                        request_id=item.request_id,
                        data={'error': str(e)},
                        metadata={**item.metadata, 'error': True, 'stage': self.name},
                        stage=item.stage
                    )
                    await self.output_queue.put(error_result)

            except asyncio.CancelledError:
                logger.info(f"[{self.name}] Worker {worker_id} cancelled")
                break

            except Exception as e:
                logger.error(
                    f"[{self.name}] Worker {worker_id} unexpected error: {e}",
                    exc_info=True
                )

        logger.info(f"[{self.name}] Worker {worker_id} stopped")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get stage statistics.

        Returns:
            Dictionary with performance metrics including:
            - total_processed: Total items processed
            - total_errors: Total errors encountered
            - total_timeouts: Total timeouts
            - avg_time: Average processing time per item
            - max_time: Maximum processing time seen
            - min_time: Minimum processing time seen
            - input_queue_size: Current input queue depth
            - output_queue_size: Current output queue depth
            - workers_active: Number of active workers
        """
        async with self._stats_lock:
            stats = self._stats.copy()

        # Calculate derived metrics
        if stats['total_processed'] > 0:
            stats['avg_time'] = stats['total_time'] / stats['total_processed']
        else:
            stats['avg_time'] = 0.0

        # Current state
        stats['input_queue_size'] = self.input_queue.qsize()
        stats['output_queue_size'] = self.output_queue.qsize()
        stats['workers_active'] = len([w for w in self._workers if not w.done()])
        stats['workers_total'] = len(self._workers)
        stats['is_running'] = self._running

        # Error rates
        total_items = stats['total_processed'] + stats['total_errors'] + stats['total_timeouts']
        if total_items > 0:
            stats['error_rate'] = stats['total_errors'] / total_items
            stats['timeout_rate'] = stats['total_timeouts'] / total_items
            stats['success_rate'] = stats['total_processed'] / total_items
        else:
            stats['error_rate'] = 0.0
            stats['timeout_rate'] = 0.0
            stats['success_rate'] = 0.0

        return stats

    async def print_stats(self):
        """Print stage statistics to console"""
        stats = await self.get_stats()

        print(f"\n[PipelineStage:{self.name}] Statistics:")
        print(f"  Total Processed:  {stats['total_processed']}")
        print(f"  Total Errors:     {stats['total_errors']}")
        print(f"  Total Timeouts:   {stats['total_timeouts']}")
        print(f"  Success Rate:     {stats['success_rate']*100:.1f}%")
        print(f"  Avg Time:         {stats['avg_time']*1000:.2f}ms")
        print(f"  Max Time:         {stats['max_time']*1000:.2f}ms")

        if stats['min_time'] != float('inf'):
            print(f"  Min Time:         {stats['min_time']*1000:.2f}ms")

        print(f"  Input Queue:      {stats['input_queue_size']}")
        print(f"  Output Queue:     {stats['output_queue_size']}")
        print(f"  Workers:          {stats['workers_active']}/{stats['workers_total']} active")

    def is_healthy(self) -> bool:
        """Check if stage is healthy (running with active workers)"""
        return self._running and any(not w.done() for w in self._workers)


async def main():
    """Demonstration of pipeline stage functionality"""
    print("Pipeline Workers - Demonstration\n")

    # Define a simple processing function
    def process_item(item: WorkItem) -> WorkItem:
        """Simulate work by sleeping"""
        import time as sync_time
        sync_time.sleep(0.1)  # Simulate 100ms work

        # Transform data
        result = WorkItem(
            request_id=item.request_id,
            data={'processed': item.data, 'result': 'success'},
            metadata={**item.metadata, 'processed_at': sync_time.time()},
            stage=item.stage + 1
        )
        return result

    # Create stage
    print("[Demo] Creating pipeline stage...")
    stage = PipelineStage(
        name="TestStage",
        process_func=process_item,
        num_workers=2,
        use_processes=False
    )

    # Start stage
    print("[Demo] Starting stage...")
    await stage.start()

    # Submit work items
    print("\n[Demo] Submitting work items...")
    for i in range(5):
        item = WorkItem(
            request_id=f"item-{i}",
            data={'value': i},
            stage=1
        )
        await stage.input_queue.put(item)
        print(f"  Submitted: {item.request_id}")

    # Wait for processing
    print("\n[Demo] Waiting for processing...")
    await asyncio.sleep(0.5)

    # Get results
    print("\n[Demo] Retrieving results...")
    while not stage.output_queue.empty():
        result = await stage.output_queue.get()
        print(f"  Result: {result.request_id} -> {result.data}")

    # Print statistics
    print("\n[Demo] Stage statistics:")
    await stage.print_stats()

    # Stop stage
    print("\n[Demo] Stopping stage...")
    await stage.stop()

    print("\n✅ Demo complete!")
    return 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(main()))
