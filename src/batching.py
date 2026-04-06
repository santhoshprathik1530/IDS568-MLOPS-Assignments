"""
Dynamic Request Batching Module

Implements a hybrid batching strategy that groups concurrent inference
requests together based on:
- Max batch size threshold (process when N requests accumulate)
- Timeout threshold (process after T ms regardless of batch size)
- Whichever triggers first

This amortizes model weight loading cost across multiple requests,
improving GPU utilization and overall throughput.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional

from src.config import BatchingConfig

logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """A request waiting to be included in a batch."""

    prompt: str
    model: str
    temperature: float
    max_tokens: int
    future: asyncio.Future
    arrival_time: float


@dataclass
class BatchStats:
    """Tracks batching performance metrics."""

    total_requests: int = 0
    total_batches: int = 0
    total_batch_time_ms: float = 0.0
    size_triggered: int = 0   # Batches triggered by size threshold
    timeout_triggered: int = 0  # Batches triggered by timeout

    @property
    def avg_batch_size(self) -> float:
        return self.total_requests / self.total_batches if self.total_batches > 0 else 0.0

    @property
    def avg_batch_latency_ms(self) -> float:
        return self.total_batch_time_ms / self.total_batches if self.total_batches > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": round(self.avg_batch_size, 2),
            "avg_batch_latency_ms": round(self.avg_batch_latency_ms, 2),
            "size_triggered": self.size_triggered,
            "timeout_triggered": self.timeout_triggered,
        }

    def reset(self):
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_time_ms = 0.0
        self.size_triggered = 0
        self.timeout_triggered = 0


class DynamicBatcher:
    """
    Groups concurrent inference requests into batches using a hybrid strategy:
    - Process when max_batch_size requests accumulate, OR
    - Process after max_wait_ms milliseconds, whichever comes first.

    Uses asyncio.Lock for thread-safe access and asyncio.Event for
    signaling between the timeout processor and request submission.
    """

    def __init__(
        self,
        config: BatchingConfig,
        inference_fn: Callable[..., Coroutine],
    ):
        """
        Args:
            config: Batching configuration parameters.
            inference_fn: Async callable that performs batched inference.
                          Signature: async def fn(prompts, model, temperature, max_tokens) -> List[str]
        """
        self.config = config
        self.max_batch_size = config.max_batch_size
        self.max_wait_ms = config.max_wait_ms
        self.inference_fn = inference_fn

        self._pending: List[PendingRequest] = []
        self._lock = asyncio.Lock()
        self._has_pending = asyncio.Event()
        self.stats = BatchStats()
        self._running = False
        self._timeout_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the background timeout processor."""
        self._running = True
        self._timeout_task = asyncio.create_task(self._timeout_processor())
        logger.info(
            "Batcher started: max_batch=%d, max_wait=%.0fms",
            self.max_batch_size,
            self.max_wait_ms,
        )

    async def stop(self):
        """Stop the batcher and process any remaining requests."""
        self._running = False
        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
        # Process any remaining requests
        await self._process_batch(trigger="shutdown")

    async def submit(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Submit a request for batched inference.

        The request is queued and will be processed either when the batch
        is full or the timeout expires. Returns the inference result.
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request = PendingRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            future=future,
            arrival_time=time.time(),
        )

        should_process = False
        async with self._lock:
            self._pending.append(request)
            self._has_pending.set()
            should_process = len(self._pending) >= self.max_batch_size

        if should_process:
            asyncio.create_task(self._process_batch(trigger="size"))

        return await future

    async def _process_batch(self, trigger: str = "unknown"):
        """Extract and process the current batch of pending requests."""
        async with self._lock:
            if not self._pending:
                return
            batch = self._pending[: self.max_batch_size]
            self._pending = self._pending[self.max_batch_size :]
            if not self._pending:
                self._has_pending.clear()

        batch_size = len(batch)
        self.stats.total_requests += batch_size
        self.stats.total_batches += 1
        if trigger == "size":
            self.stats.size_triggered += 1
        elif trigger == "timeout":
            self.stats.timeout_triggered += 1

        logger.info(
            "Processing batch #%d: %d requests (trigger=%s)",
            self.stats.total_batches,
            batch_size,
            trigger,
        )

        start_time = time.time()
        try:
            # All requests in a batch use the same model params (first request's)
            prompts = [r.prompt for r in batch]
            results = await self.inference_fn(
                prompts=prompts,
                model=batch[0].model,
                temperature=batch[0].temperature,
                max_tokens=batch[0].max_tokens,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats.total_batch_time_ms += elapsed_ms

            # Deliver results to waiting callers
            for i, request in enumerate(batch):
                if not request.future.done():
                    request.future.set_result(results[i])

        except Exception as e:
            logger.error("Batch inference failed: %s", e)
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)

    async def _timeout_processor(self):
        """
        Background task that checks for pending requests and triggers
        batch processing when the max wait time is exceeded.
        """
        while self._running:
            try:
                # Wait until there are pending requests
                await asyncio.wait_for(
                    self._has_pending.wait(),
                    timeout=0.5,
                )
            except asyncio.TimeoutError:
                continue

            # Check if oldest request has waited long enough
            async with self._lock:
                if self._pending:
                    oldest = self._pending[0]
                    wait_time_ms = (time.time() - oldest.arrival_time) * 1000
                    should_process = wait_time_ms >= self.max_wait_ms
                else:
                    should_process = False

            if should_process:
                await self._process_batch(trigger="timeout")
            else:
                # Small sleep to avoid busy-waiting
                await asyncio.sleep(self.max_wait_ms / 2000)

    def get_stats(self) -> Dict[str, Any]:
        """Return batching statistics as a dictionary."""
        stats = self.stats.to_dict()
        stats["pending_requests"] = len(self._pending)
        stats["max_batch_size"] = self.max_batch_size
        stats["max_wait_ms"] = self.max_wait_ms
        return stats
