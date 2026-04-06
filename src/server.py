"""
LLM Inference Optimization Server

Production-ready FastAPI server implementing:
- Dynamic request batching (configurable batch size & timeout)
- Intelligent caching with TTL and LRU eviction
- Async endpoints for concurrent request handling
- Health checks and metrics endpoints

Usage:
    uvicorn src.server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.batching import DynamicBatcher
from src.caching import InferenceCache
from src.config import ServerConfig, get_config, print_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
config: ServerConfig = get_config()
cache: Optional[InferenceCache] = None
batcher: Optional[DynamicBatcher] = None
tokenizer = None
model = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(cfg: ServerConfig):
    """Load the tokenizer and model from Hugging Face Hub."""
    global tokenizer, model
    logger.info("Loading model: %s on device: %s", cfg.model.model_name, cfg.model.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name)
    device = cfg.model.device
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
    else:
        model = model.to("cpu")
    model.eval()
    logger.info("Model loaded successfully.")


# ---------------------------------------------------------------------------
# Inference function (used by batcher)
# ---------------------------------------------------------------------------
async def run_batched_inference(
    prompts: list[str],
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> list[str]:
    """
    Run batched inference on a list of prompts.
    Executed in a thread pool to avoid blocking the event loop.
    """

    def _generate():
        device = next(model.parameters()).device
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )

        results = []
        for i, output in enumerate(outputs):
            # Decode only the newly generated tokens
            input_len = inputs["input_ids"][i].shape[0]
            generated = tokenizer.decode(
                output[input_len:], skip_special_tokens=True
            )
            results.append(generated.strip())
        return results

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _generate)


# ---------------------------------------------------------------------------
# Single-request inference (no batching)
# ---------------------------------------------------------------------------
async def run_single_inference(
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Run inference for a single prompt (bypasses batching)."""
    results = await run_batched_inference(
        prompts=[prompt],
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return results[0]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The input prompt.")
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Sampling temperature."
    )
    max_tokens: int = Field(
        default=50, ge=1, le=512, description="Maximum tokens to generate."
    )
    use_cache: bool = Field(
        default=True, description="Whether to use the cache for this request."
    )
    use_batching: bool = Field(
        default=True, description="Whether to use batching for this request."
    )


class InferenceResponse(BaseModel):
    prompt: str
    response: str
    cached: bool = False
    cache_status: str = ""
    latency_ms: float = 0.0
    batch_size: int = 1


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global config, cache, batcher

    config = get_config()
    print_config(config)

    # Load model
    load_model(config)

    # Initialize cache
    if config.caching.enabled:
        cache = InferenceCache(config.caching)
        logger.info("Cache initialized (TTL=%.0fs, max=%d entries)",
                     config.caching.ttl_seconds, config.caching.max_entries)

    # Initialize batcher
    if config.batching.enabled:
        batcher = DynamicBatcher(
            config=config.batching,
            inference_fn=run_batched_inference,
        )
        await batcher.start()
        logger.info("Batcher initialized (batch=%d, wait=%.0fms)",
                     config.batching.max_batch_size, config.batching.max_wait_ms)

    yield  # Server is running

    # Shutdown
    if batcher:
        await batcher.stop()
    logger.info("Server shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM Inference Optimization Server",
    description="Production-ready inference API with dynamic batching and caching.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "LLM Inference Optimization Server",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate": "Generate text from a prompt",
            "GET /health": "Health check",
            "GET /metrics": "Cache & batching statistics",
            "POST /cache/clear": "Clear the cache",
            "POST /metrics/reset": "Reset metrics counters",
            "GET /docs": "Interactive API documentation (Swagger UI)",
        },
    }


@app.post("/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest):
    """
    Generate text from a prompt.

    The request flows through:
    1. Cache lookup (if enabled and use_cache=True)
    2. Dynamic batching (if enabled and use_batching=True)
    3. Direct inference (fallback)
    """
    start_time = time.time()
    model_name = config.model.model_name
    cache_status = "DISABLED"
    cached = False

    # Step 1: Check cache
    if config.caching.enabled and cache and request.use_cache:
        cached_response, cache_status = await cache.get(
            prompt=request.prompt,
            model=model_name,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        if cached_response is not None:
            elapsed_ms = (time.time() - start_time) * 1000
            return InferenceResponse(
                prompt=request.prompt,
                response=cached_response,
                cached=True,
                cache_status=cache_status,
                latency_ms=round(elapsed_ms, 2),
            )

    # Step 2: Run inference (batched or single)
    try:
        if config.batching.enabled and batcher and request.use_batching:
            response_text = await batcher.submit(
                prompt=request.prompt,
                model=model_name,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        else:
            response_text = await run_single_inference(
                prompt=request.prompt,
                model_name=model_name,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Step 3: Store in cache
    if config.caching.enabled and cache and request.use_cache:
        await cache.put(
            prompt=request.prompt,
            model=model_name,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response=response_text,
        )

    elapsed_ms = (time.time() - start_time) * 1000
    return InferenceResponse(
        prompt=request.prompt,
        response=response_text,
        cached=False,
        cache_status=cache_status,
        latency_ms=round(elapsed_ms, 2),
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": config.model.model_name,
        "batching_enabled": config.batching.enabled,
        "caching_enabled": config.caching.enabled,
    }


@app.get("/metrics")
async def metrics():
    """Return current server metrics (cache + batching + resource stats)."""
    result = {}
    if cache:
        result["cache"] = cache.get_stats()
    if batcher:
        result["batching"] = batcher.get_stats()

    # ── Resource / memory utilization ──────────────────────────────────
    import resource
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    proc_mem: dict = {
        "rss_mb": round(rusage.ru_maxrss / (1024 * 1024), 2)   # macOS gives bytes
            if "darwin" in __import__("sys").platform
            else round(rusage.ru_maxrss / 1024, 2),             # Linux gives KB
        "pid": os.getpid(),
    }
    # PyTorch CUDA memory (only if GPU is in use)
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        proc_mem["gpu_memory_allocated_mb"] = round(
            torch.cuda.memory_allocated() / (1024 ** 2), 2
        )
        proc_mem["gpu_memory_reserved_mb"] = round(
            torch.cuda.memory_reserved() / (1024 ** 2), 2
        )
        proc_mem["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        proc_mem["gpu"] = "not available (cpu mode)"

    result["resource_utilization"] = proc_mem
    return result


@app.post("/cache/clear")
async def clear_cache():
    """Clear the inference cache."""
    if cache:
        await cache.clear()
        cache.stats.reset()
        return {"status": "cache cleared"}
    return {"status": "cache not enabled"}


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset all metrics counters."""
    if cache:
        cache.stats.reset()
    if batcher:
        batcher.stats.reset()
    return {"status": "metrics reset"}
