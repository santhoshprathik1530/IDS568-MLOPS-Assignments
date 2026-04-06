"""
Inference Caching Module

Implements an in-memory LLM response cache with:
- SHA-256 hashed cache keys (no plaintext user data stored)
- Configurable TTL (time-to-live) based expiration
- Configurable max-entry limit with LRU eviction
- Cache bypass for non-deterministic requests (temperature > 0)
- Thread-safe async access via asyncio.Lock
- Hit/miss/bypass statistics tracking
"""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from src.config import CachingConfig


@dataclass
class CacheEntry:
    """A single cached inference response with metadata."""

    response: str
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0

    def __post_init__(self):
        if self.last_accessed == 0.0:
            self.last_accessed = self.created_at


@dataclass
class CacheStats:
    """Tracks cache performance metrics."""

    hits: int = 0
    misses: int = 0
    bypasses: int = 0
    evictions_ttl: int = 0
    evictions_lru: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses + self.bypasses

    @property
    def hit_rate(self) -> float:
        """Hit rate excludes bypasses (they never check the cache)."""
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "bypasses": self.bypasses,
            "evictions_ttl": self.evictions_ttl,
            "evictions_lru": self.evictions_lru,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
        }

    def reset(self):
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.evictions_ttl = 0
        self.evictions_lru = 0


class InferenceCache:
    """
    In-memory LLM response cache with TTL expiration, LRU eviction,
    and hashed keys for privacy preservation.

    Cache keys are SHA-256 hashes of (prompt + model + temperature + max_tokens).
    No plaintext user identifiers or prompts are stored.
    """

    def __init__(self, config: CachingConfig):
        self.config = config
        self.ttl_seconds = config.ttl_seconds
        self.max_entries = config.max_entries
        # OrderedDict for LRU tracking: most recently used at end
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self._lock = asyncio.Lock()

    @staticmethod
    def make_cache_key(
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Create a deterministic, hashed cache key from request parameters.

        The prompt is normalized (stripped, lowercased) before hashing.
        The key is a SHA-256 hex digest — the raw prompt text is never stored.
        """
        key_data = {
            "prompt": prompt.strip().lower(),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        content = json.dumps(key_data, sort_keys=True)
        return f"llm:{hashlib.sha256(content.encode()).hexdigest()}"

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has exceeded its TTL."""
        return (time.time() - entry.created_at) > self.ttl_seconds

    def _evict_expired(self) -> None:
        """Remove all expired entries from the cache."""
        now = time.time()
        expired_keys = [
            k for k, v in self._store.items()
            if (now - v.created_at) > self.ttl_seconds
        ]
        for key in expired_keys:
            del self._store[key]
            self.stats.evictions_ttl += 1

    def _evict_lru(self) -> None:
        """Evict the least recently used entry if cache exceeds max size."""
        while len(self._store) >= self.max_entries:
            self._store.popitem(last=False)  # Remove oldest (LRU)
            self.stats.evictions_lru += 1

    async def get(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Tuple[Optional[str], str]:
        """
        Look up a cached response.

        Returns (response_or_None, status_string).
        Bypasses cache for non-deterministic requests (temperature > 0).
        """
        # Bypass: non-deterministic requests should never use cache
        if temperature > 0:
            self.stats.bypasses += 1
            return None, "BYPASS"

        key = self.make_cache_key(prompt, model, temperature, max_tokens)

        async with self._lock:
            if key in self._store:
                entry = self._store[key]
                if self._is_expired(entry):
                    del self._store[key]
                    self.stats.evictions_ttl += 1
                    self.stats.misses += 1
                    return None, "MISS_EXPIRED"
                # Mark as recently used (move to end of OrderedDict)
                self._store.move_to_end(key)
                entry.access_count += 1
                entry.last_accessed = time.time()
                self.stats.hits += 1
                return entry.response, "HIT"

            self.stats.misses += 1
            return None, "MISS"

    async def put(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        response: str,
    ) -> None:
        """Store a response in the cache. Skips non-deterministic requests."""
        if temperature > 0:
            return  # Never cache non-deterministic responses

        key = self.make_cache_key(prompt, model, temperature, max_tokens)

        async with self._lock:
            # Evict expired entries first
            self._evict_expired()
            # Evict LRU if at capacity
            self._evict_lru()
            # Store the new entry
            self._store[key] = CacheEntry(
                response=response,
                created_at=time.time(),
            )

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self._store)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics as a dictionary."""
        stats = self.stats.to_dict()
        stats["current_size"] = self.size
        stats["max_entries"] = self.max_entries
        stats["ttl_seconds"] = self.ttl_seconds
        return stats
