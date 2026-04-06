"""
Synthetic Load Generator

Generates configurable concurrent request loads to benchmark
the inference server. Supports:
- Adjustable concurrency levels
- Configurable mix of unique and repeated prompts (for cache testing)
- Output of per-request timing data for analysis
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp


# ---------------------------------------------------------------------------
# Prompt Pool
# ---------------------------------------------------------------------------
UNIQUE_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "What are the main differences between Python and JavaScript?",
    "Describe how a neural network learns from data.",
    "What is the capital of France and its historical significance?",
    "How does photosynthesis work in plants?",
    "Explain the concept of supply and demand in economics.",
    "What are the key principles of object-oriented programming?",
    "Describe the process of DNA replication.",
    "How does encryption protect data on the internet?",
    "What is the significance of the Turing test?",
    "Explain how a compiler differs from an interpreter.",
    "What are the main causes of climate change?",
    "Describe the architecture of a transformer model.",
    "How do recommendation systems work?",
    "What is reinforcement learning and where is it used?",
    "Explain the concept of containerization in software.",
    "What are the benefits of microservices architecture?",
    "How does a database index improve query performance?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the concept of gradient descent in machine learning.",
    "What are the challenges of deploying models at scale?",
    "Describe how caching improves system performance.",
    "What is the difference between REST and GraphQL APIs?",
    "How does load balancing work in distributed systems?",
    "Explain the CAP theorem in distributed computing.",
    "What are the ethical concerns around AI in healthcare?",
    "Describe the MapReduce programming model.",
    "How does version control help in software development?",
    "What is continuous integration and continuous deployment?",
    "Explain the concept of transfer learning in deep learning.",
]

# Prompts that will be repeated to test cache effectiveness
REPEATED_PROMPTS = [
    "What is machine learning?",
    "Explain neural networks.",
    "What is deep learning?",
    "How does gradient descent work?",
    "What is a transformer model?",
]


@dataclass
class RequestResult:
    """Result from a single benchmark request."""

    prompt: str
    response_length: int
    latency_ms: float
    cached: bool
    cache_status: str
    status_code: int
    error: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class LoadTestResult:
    """Aggregated results from a load test run."""

    name: str
    concurrency: int
    total_requests: int
    successful: int = 0
    failed: int = 0
    total_time_sec: float = 0.0
    results: List[RequestResult] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        """Requests per second."""
        return self.successful / self.total_time_sec if self.total_time_sec > 0 else 0

    @property
    def avg_latency_ms(self) -> float:
        latencies = [r.latency_ms for r in self.results if r.error is None]
        return sum(latencies) / len(latencies) if latencies else 0

    @property
    def p50_latency_ms(self) -> float:
        return self._percentile(50)

    @property
    def p95_latency_ms(self) -> float:
        return self._percentile(95)

    @property
    def p99_latency_ms(self) -> float:
        return self._percentile(99)

    @property
    def cache_hit_rate(self) -> float:
        hits = sum(1 for r in self.results if r.cached)
        total = len([r for r in self.results if r.cache_status != "BYPASS"])
        return hits / total if total > 0 else 0

    def _percentile(self, pct: int) -> float:
        latencies = sorted([r.latency_ms for r in self.results if r.error is None])
        if not latencies:
            return 0
        idx = int(len(latencies) * pct / 100)
        idx = min(idx, len(latencies) - 1)
        return latencies[idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "concurrency": self.concurrency,
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "total_time_sec": round(self.total_time_sec, 3),
            "throughput_rps": round(self.throughput, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
        }


# ---------------------------------------------------------------------------
# Load generator
# ---------------------------------------------------------------------------
async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    use_cache: bool = True,
    use_batching: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 50,
) -> RequestResult:
    """Send a single inference request and record the result."""
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "use_cache": use_cache,
        "use_batching": use_batching,
    }

    start_time = time.time()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            data = await resp.json()
            elapsed_ms = (time.time() - start_time) * 1000
            return RequestResult(
                prompt=prompt,
                response_length=len(data.get("response", "")),
                latency_ms=elapsed_ms,
                cached=data.get("cached", False),
                cache_status=data.get("cache_status", ""),
                status_code=resp.status,
                timestamp=start_time,
            )
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return RequestResult(
            prompt=prompt,
            response_length=0,
            latency_ms=elapsed_ms,
            cached=False,
            cache_status="ERROR",
            status_code=0,
            error=str(e),
            timestamp=start_time,
        )


def generate_prompts(
    num_requests: int,
    repeat_ratio: float = 0.3,
) -> List[str]:
    """
    Generate a list of prompts with a configurable ratio of repeated prompts.

    Args:
        num_requests: Total number of prompts to generate.
        repeat_ratio: Fraction of requests that use repeated prompts (for cache hits).
    """
    prompts = []
    for _ in range(num_requests):
        if random.random() < repeat_ratio:
            prompts.append(random.choice(REPEATED_PROMPTS))
        else:
            prompts.append(random.choice(UNIQUE_PROMPTS))
    return prompts


async def run_load_test(
    name: str,
    base_url: str,
    num_requests: int,
    concurrency: int,
    repeat_ratio: float = 0.3,
    use_cache: bool = True,
    use_batching: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 50,
) -> LoadTestResult:
    """
    Run a load test with the given parameters.

    Sends num_requests with up to `concurrency` in-flight at once.
    """
    url = f"{base_url}/generate"
    prompts = generate_prompts(num_requests, repeat_ratio)
    result = LoadTestResult(
        name=name,
        concurrency=concurrency,
        total_requests=num_requests,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(session, prompt):
        async with semaphore:
            return await send_request(
                session, url, prompt,
                use_cache=use_cache,
                use_batching=use_batching,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [bounded_request(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)

    result.total_time_sec = time.time() - start_time
    result.results = list(results)
    result.successful = sum(1 for r in results if r.error is None)
    result.failed = sum(1 for r in results if r.error is not None)

    return result


def save_results(result: LoadTestResult, output_dir: str = "benchmarks/results"):
    """Save load test results to a JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{output_dir}/{result.name.replace(' ', '_').lower()}.json"
    data = result.to_dict()
    data["per_request"] = [
        {
            "prompt": r.prompt[:50],
            "latency_ms": round(r.latency_ms, 2),
            "cached": r.cached,
            "cache_status": r.cache_status,
            "status_code": r.status_code,
            "error": r.error,
        }
        for r in result.results
    ]
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to: {filename}")


def print_results(result: LoadTestResult):
    """Pretty-print load test results."""
    d = result.to_dict()
    print(f"\n{'=' * 60}")
    print(f"  {d['name']}")
    print(f"{'=' * 60}")
    print(f"  Requests:      {d['total_requests']} ({d['successful']} ok, {d['failed']} failed)")
    print(f"  Concurrency:   {d['concurrency']}")
    print(f"  Total Time:    {d['total_time_sec']:.2f}s")
    print(f"  Throughput:    {d['throughput_rps']:.2f} req/s")
    print(f"  Avg Latency:   {d['avg_latency_ms']:.1f}ms")
    print(f"  P50 Latency:   {d['p50_latency_ms']:.1f}ms")
    print(f"  P95 Latency:   {d['p95_latency_ms']:.1f}ms")
    print(f"  P99 Latency:   {d['p99_latency_ms']:.1f}ms")
    print(f"  Cache Hit Rate: {d['cache_hit_rate']:.1%}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Synthetic load generator for the LLM inference server."
    )
    parser.add_argument(
        "--url", type=str, default="http://localhost:8000",
        help="Base URL of the inference server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--requests", type=int, default=20,
        help="Total number of requests to send (default: 20)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Maximum concurrent requests (default: 5)",
    )
    parser.add_argument(
        "--repeat-ratio", type=float, default=0.3,
        help="Fraction of repeated prompts for cache testing (default: 0.3)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable cache usage for requests",
    )
    parser.add_argument(
        "--no-batching", action="store_true",
        help="Disable batching for requests",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50,
        help="Max tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/results",
        help="Directory to save results (default: benchmarks/results)",
    )
    parser.add_argument(
        "--name", type=str, default="load_test",
        help="Name for this test run (default: load_test)",
    )
    args = parser.parse_args()

    print(f"Running load test: {args.name}")
    print(f"  URL:         {args.url}")
    print(f"  Requests:    {args.requests}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Repeat ratio: {args.repeat_ratio}")
    print(f"  Cache: {'disabled' if args.no_cache else 'enabled'}")
    print(f"  Batching: {'disabled' if args.no_batching else 'enabled'}")

    result = asyncio.run(
        run_load_test(
            name=args.name,
            base_url=args.url,
            num_requests=args.requests,
            concurrency=args.concurrency,
            repeat_ratio=args.repeat_ratio,
            use_cache=not args.no_cache,
            use_batching=not args.no_batching,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )

    print_results(result)
    save_results(result, args.output_dir)


if __name__ == "__main__":
    main()
