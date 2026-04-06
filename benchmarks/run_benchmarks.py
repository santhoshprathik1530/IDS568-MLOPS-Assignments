"""
Benchmark Orchestration Script

Runs a comprehensive suite of benchmarks to measure the impact of
batching and caching optimizations. Generates results and visualizations.

Benchmarks:
1. Baseline (no batching, no caching)
2. Batching only
3. Caching only (cold then warm)
4. Batching + Caching
5. Throughput scaling at multiple load levels
6. Cache hit-rate over time

Usage:
    python benchmarks/run_benchmarks.py --help
    python benchmarks/run_benchmarks.py --url http://localhost:8000
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.load_generator import (
    LoadTestResult,
    print_results,
    run_load_test,
    save_results,
)


# ---------------------------------------------------------------------------
# Helper: reset server metrics and clear cache
# ---------------------------------------------------------------------------
async def reset_server(base_url: str):
    """Reset server metrics and clear cache between benchmark runs."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{base_url}/cache/clear") as resp:
                await resp.json()
            async with session.post(f"{base_url}/metrics/reset") as resp:
                await resp.json()
        except Exception as e:
            print(f"  Warning: Could not reset server: {e}")


async def get_server_metrics(base_url: str) -> dict:
    """Fetch current server metrics."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/metrics") as resp:
                return await resp.json()
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Benchmark 1: Baseline vs Batching vs Caching
# ---------------------------------------------------------------------------
async def benchmark_configurations(
    base_url: str,
    num_requests: int,
    concurrency: int,
    output_dir: str,
) -> dict:
    """Compare different optimization configurations."""
    configs = [
        {
            "name": "baseline_no_optimization",
            "use_cache": False,
            "use_batching": False,
            "desc": "Baseline (no batching, no caching)",
        },
        {
            "name": "batching_only",
            "use_cache": False,
            "use_batching": True,
            "desc": "Batching only",
        },
        {
            "name": "caching_only_cold",
            "use_cache": True,
            "use_batching": False,
            "desc": "Caching only (cold cache)",
        },
        {
            "name": "caching_only_warm",
            "use_cache": True,
            "use_batching": False,
            "desc": "Caching only (warm cache — re-run)",
        },
        {
            "name": "batching_and_caching",
            "use_cache": True,
            "use_batching": True,
            "desc": "Batching + Caching",
        },
    ]

    results = {}
    for cfg in configs:
        print(f"\n>>> Running: {cfg['desc']}")
        # Reset between runs, but NOT before the warm-cache test
        if cfg["name"] != "caching_only_warm":
            await reset_server(base_url)
            await asyncio.sleep(1)

        result = await run_load_test(
            name=cfg["name"],
            base_url=base_url,
            num_requests=num_requests,
            concurrency=concurrency,
            repeat_ratio=0.4,  # 40% repeated prompts
            use_cache=cfg["use_cache"],
            use_batching=cfg["use_batching"],
        )
        print_results(result)
        save_results(result, output_dir)
        results[cfg["name"]] = result

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Throughput scaling at multiple load levels
# ---------------------------------------------------------------------------
async def benchmark_throughput_scaling(
    base_url: str,
    output_dir: str,
) -> list:
    """Test throughput at increasing concurrency levels."""
    levels = [
        {"concurrency": 1, "requests": 10},
        {"concurrency": 5, "requests": 25},
        {"concurrency": 10, "requests": 40},
        {"concurrency": 20, "requests": 60},
        {"concurrency": 30, "requests": 60},
    ]

    results = []
    for level in levels:
        name = f"throughput_c{level['concurrency']}"
        print(f"\n>>> Throughput test: concurrency={level['concurrency']}")
        await reset_server(base_url)
        await asyncio.sleep(1)

        result = await run_load_test(
            name=name,
            base_url=base_url,
            num_requests=level["requests"],
            concurrency=level["concurrency"],
            repeat_ratio=0.3,
            use_cache=True,
            use_batching=True,
        )
        print_results(result)
        save_results(result, output_dir)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Benchmark 3: Cache hit-rate over time
# ---------------------------------------------------------------------------
async def benchmark_cache_hitrate(
    base_url: str,
    output_dir: str,
) -> list:
    """Track cache hit-rate as requests accumulate."""
    await reset_server(base_url)
    await asyncio.sleep(1)

    url = f"{base_url}/generate"
    # Send requests in waves: increasing repeated ratio
    waves = []
    cumulative_hits = 0
    cumulative_total = 0

    for wave_idx in range(6):
        repeat_ratio = 0.1 + wave_idx * 0.15  # 10%, 25%, 40%, 55%, 70%, 85%
        num_requests = 15

        result = await run_load_test(
            name=f"cache_wave_{wave_idx}",
            base_url=base_url,
            num_requests=num_requests,
            concurrency=5,
            repeat_ratio=min(repeat_ratio, 0.9),
            use_cache=True,
            use_batching=True,
        )

        wave_hits = sum(1 for r in result.results if r.cached)
        wave_total = len([r for r in result.results if r.cache_status != "BYPASS"])
        cumulative_hits += wave_hits
        cumulative_total += wave_total

        wave_data = {
            "wave": wave_idx + 1,
            "repeat_ratio": round(repeat_ratio, 2),
            "requests": num_requests,
            "wave_hit_rate": round(wave_hits / wave_total, 4) if wave_total > 0 else 0,
            "cumulative_hit_rate": round(cumulative_hits / cumulative_total, 4) if cumulative_total > 0 else 0,
            "avg_latency_ms": round(result.avg_latency_ms, 2),
        }
        waves.append(wave_data)
        print(f"  Wave {wave_idx + 1}: repeat_ratio={repeat_ratio:.0%}, "
              f"hit_rate={wave_data['wave_hit_rate']:.1%}, "
              f"cumulative={wave_data['cumulative_hit_rate']:.1%}")

    # Save wave data
    wave_file = f"{output_dir}/cache_hitrate_over_time.json"
    with open(wave_file, "w") as f:
        json.dump(waves, f, indent=2)
    print(f"  Cache hit-rate results saved to: {wave_file}")

    return waves


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def generate_visualizations(
    config_results: dict,
    throughput_results: list,
    cache_waves: list,
    viz_dir: str,
):
    """Generate benchmark visualization charts."""
    Path(viz_dir).mkdir(parents=True, exist_ok=True)

    # --- Chart 1: Latency Comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    names = []
    avg_latencies = []
    p95_latencies = []
    for name, result in config_results.items():
        label = name.replace("_", " ").title()
        names.append(label)
        avg_latencies.append(result.avg_latency_ms)
        p95_latencies.append(result.p95_latency_ms)

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width / 2, avg_latencies, width, label="Avg Latency", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, p95_latencies, width, label="P95 Latency", color="#DD8452")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Comparison: Batching & Caching Configurations")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.bar_label(bars1, fmt="%.0f", fontsize=7)
    ax.bar_label(bars2, fmt="%.0f", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/latency_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {viz_dir}/latency_comparison.png")

    # --- Chart 2: Throughput Comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    names = []
    throughputs = []
    for name, result in config_results.items():
        names.append(name.replace("_", " ").title())
        throughputs.append(result.throughput)

    colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2", "#CCB974"]
    bars = ax.bar(names, throughputs, color=colors[:len(names)])
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput Comparison: Batching & Caching Configurations")
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.bar_label(bars, fmt="%.1f", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/throughput_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {viz_dir}/throughput_comparison.png")

    # --- Chart 3: Throughput Scaling ---
    if throughput_results:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        concurrencies = [r.concurrency for r in throughput_results]
        throughputs = [r.throughput for r in throughput_results]
        avg_latencies = [r.avg_latency_ms for r in throughput_results]

        color1 = "#4C72B0"
        ax1.set_xlabel("Concurrency Level")
        ax1.set_ylabel("Throughput (req/s)", color=color1)
        ax1.plot(concurrencies, throughputs, "o-", color=color1, linewidth=2, label="Throughput")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "#DD8452"
        ax2.set_ylabel("Avg Latency (ms)", color=color2)
        ax2.plot(concurrencies, avg_latencies, "s--", color=color2, linewidth=2, label="Avg Latency")
        ax2.tick_params(axis="y", labelcolor=color2)

        fig.suptitle("Throughput & Latency vs. Concurrency Level")
        fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/throughput_scaling.png", dpi=150)
        plt.close()
        print(f"  Saved: {viz_dir}/throughput_scaling.png")

    # --- Chart 4: Cache Hit-Rate Over Time ---
    if cache_waves:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        waves = [w["wave"] for w in cache_waves]
        wave_hr = [w["wave_hit_rate"] * 100 for w in cache_waves]
        cum_hr = [w["cumulative_hit_rate"] * 100 for w in cache_waves]
        latencies = [w["avg_latency_ms"] for w in cache_waves]

        ax1.set_xlabel("Wave Number (increasing repeat ratio)")
        ax1.set_ylabel("Hit Rate (%)")
        ax1.plot(waves, wave_hr, "o-", color="#55A868", linewidth=2, label="Wave Hit Rate")
        ax1.plot(waves, cum_hr, "s-", color="#4C72B0", linewidth=2, label="Cumulative Hit Rate")
        ax1.set_ylim(0, 100)
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Avg Latency (ms)", color="#DD8452")
        ax2.plot(waves, latencies, "^--", color="#DD8452", linewidth=2, label="Avg Latency")
        ax2.tick_params(axis="y", labelcolor="#DD8452")

        plt.title("Cache Hit-Rate and Latency Over Time")
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/cache_hitrate_over_time.png", dpi=150)
        plt.close()
        print(f"  Saved: {viz_dir}/cache_hitrate_over_time.png")

    # --- Chart 5: Cache Hit Rate per Config ---
    fig, ax = plt.subplots(figsize=(8, 5))
    cache_configs = {
        k: v for k, v in config_results.items()
        if "caching" in k or "batching_and" in k
    }
    if cache_configs:
        names = [k.replace("_", " ").title() for k in cache_configs]
        hit_rates = [v.cache_hit_rate * 100 for v in cache_configs.values()]
        bars = ax.bar(names, hit_rates, color=["#55A868", "#8172B2", "#CCB974"][:len(names)])
        ax.set_ylabel("Cache Hit Rate (%)")
        ax.set_title("Cache Hit Rate by Configuration")
        ax.set_ylim(0, 100)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/cache_hitrate_by_config.png", dpi=150)
        plt.close()
        print(f"  Saved: {viz_dir}/cache_hitrate_by_config.png")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
async def run_all_benchmarks(args):
    """Run the complete benchmark suite."""
    base_url = args.url
    output_dir = args.output_dir
    viz_dir = args.viz_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(viz_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LLM INFERENCE BENCHMARK SUITE")
    print("=" * 60)
    print(f"  Server:     {base_url}")
    print(f"  Output:     {output_dir}")
    print(f"  Viz:        {viz_dir}")
    print("=" * 60)

    # Check server health
    print("\n>>> Checking server health...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as resp:
                health = await resp.json()
                print(f"  Server status: {health.get('status')}")
                print(f"  Model: {health.get('model')}")
    except Exception as e:
        print(f"  ERROR: Cannot reach server at {base_url}: {e}")
        print("  Please start the server first: uvicorn src.server:app")
        return

    # Benchmark 1: Configuration comparison
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Configuration Comparison")
    print("=" * 60)
    config_results = await benchmark_configurations(
        base_url, num_requests=args.requests, concurrency=args.concurrency, output_dir=output_dir
    )

    # Benchmark 2: Throughput scaling
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Throughput Scaling")
    print("=" * 60)
    throughput_results = await benchmark_throughput_scaling(base_url, output_dir)

    # Benchmark 3: Cache hit-rate over time
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Cache Hit-Rate Over Time")
    print("=" * 60)
    cache_waves = await benchmark_cache_hitrate(base_url, output_dir)

    # Fetch final server metrics
    print("\n>>> Fetching final server metrics...")
    metrics = await get_server_metrics(base_url)
    metrics_file = f"{output_dir}/final_server_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_file}")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    generate_visualizations(config_results, throughput_results, cache_waves, viz_dir)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 60)
    print(f"  Results:        {output_dir}/")
    print(f"  Visualizations: {viz_dir}/")
    print(f"  Server metrics: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmarks for the LLM inference server."
    )
    parser.add_argument(
        "--url", type=str, default="http://localhost:8000",
        help="Base URL of the inference server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--requests", type=int, default=20,
        help="Number of requests per configuration test (default: 20)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Default concurrency for configuration tests (default: 5)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/results",
        help="Directory for raw benchmark data (default: benchmarks/results)",
    )
    parser.add_argument(
        "--viz-dir", type=str, default="analysis/visualizations",
        help="Directory for visualization charts (default: analysis/visualizations)",
    )
    args = parser.parse_args()

    asyncio.run(run_all_benchmarks(args))


if __name__ == "__main__":
    main()
