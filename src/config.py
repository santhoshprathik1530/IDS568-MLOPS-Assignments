"""
Configuration management for the LLM Inference Optimization Server.

Centralizes all tunable parameters for batching, caching, model selection,
and server behavior. Values can be overridden via environment variables.
"""

import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the LLM model."""

    # Model identifier on Hugging Face Hub
    model_name: str = os.getenv("MODEL_NAME", "gpt2")

    # Maximum tokens the model can generate per request
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "50"))

    # Device to run inference on ("cpu", "cuda", "mps")
    device: str = os.getenv("DEVICE", "cpu")


@dataclass
class BatchingConfig:
    """Configuration for dynamic request batching."""

    # Maximum number of requests to group in a single batch
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "8"))

    # Maximum time (milliseconds) to wait before processing a partial batch
    max_wait_ms: float = float(os.getenv("MAX_WAIT_MS", "100.0"))

    # Whether batching is enabled
    enabled: bool = os.getenv("BATCHING_ENABLED", "true").lower() == "true"


@dataclass
class CachingConfig:
    """Configuration for the inference cache."""

    # Time-to-live for cache entries (seconds)
    ttl_seconds: float = float(os.getenv("CACHE_TTL_SECONDS", "300.0"))

    # Maximum number of entries in the cache
    max_entries: int = int(os.getenv("CACHE_MAX_ENTRIES", "1000"))

    # Whether caching is enabled
    enabled: bool = os.getenv("CACHING_ENABLED", "true").lower() == "true"

    # Backend: "memory" for in-process dict, "redis" for Redis
    backend: str = os.getenv("CACHE_BACKEND", "memory")

    # Redis URL (only used when backend == "redis")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")


@dataclass
class ServerConfig:
    """Top-level server configuration."""

    host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVER_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")

    model: ModelConfig = field(default_factory=ModelConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)


def get_config() -> ServerConfig:
    """Return the current server configuration."""
    return ServerConfig()


def print_config(config: ServerConfig) -> None:
    """Pretty-print the current configuration."""
    print("=" * 60)
    print("SERVER CONFIGURATION")
    print("=" * 60)
    print(f"  Host:            {config.host}:{config.port}")
    print(f"  Log Level:       {config.log_level}")
    print(f"\n  Model:")
    print(f"    Name:          {config.model.model_name}")
    print(f"    Max Tokens:    {config.model.max_new_tokens}")
    print(f"    Device:        {config.model.device}")
    print(f"\n  Batching:")
    print(f"    Enabled:       {config.batching.enabled}")
    print(f"    Max Batch:     {config.batching.max_batch_size}")
    print(f"    Max Wait (ms): {config.batching.max_wait_ms}")
    print(f"\n  Caching:")
    print(f"    Enabled:       {config.caching.enabled}")
    print(f"    Backend:       {config.caching.backend}")
    print(f"    TTL (sec):     {config.caching.ttl_seconds}")
    print(f"    Max Entries:   {config.caching.max_entries}")
    print("=" * 60)


if __name__ == "__main__":
    cfg = get_config()
    print_config(cfg)
