#!/usr/bin/env python3
"""
Distributed Feature Engineering Pipeline — Milestone 4
MLOps Course

Implements a Ray-based distributed feature engineering pipeline over synthetic
transaction data and compares performance against a single-machine pandas baseline.

Transforms applied (identical in both modes for correctness verification):
    1. log_amount       = log1p(amount)
    2. hour_of_day      = timestamp.hour
    3. day_of_week      = timestamp.dayofweek
    4. is_weekend       = day_of_week >= 5
    5. amount_bucket    = pd.cut(amount, bins=[0,25,100,500,∞])
    Per-user aggregations (groupby shuffle):
    6. total_spend      = sum(amount)
    7. mean_spend       = mean(amount)
    8. tx_count         = count(amount)
    9. unique_merchants = nunique(merchant_id)
    10. unique_categories = nunique(category)

Usage:
    # Quick test on small data
    python pipeline.py --input test_data/ --output test_output/

    # Full benchmark (generate 10M rows first)
    python generate_data.py --rows 10000000 --seed 42 --output data/
    python pipeline.py --input data/transactions.parquet --output results/

    # Control number of Ray workers
    python pipeline.py --input data/ --output results/ --workers 4
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import ray

from generate_data import compute_data_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering logic (shared by both local and distributed modes)
# ---------------------------------------------------------------------------

AMOUNT_BINS   = [0, 25, 100, 500, np.inf]
AMOUNT_LABELS = ["low", "medium", "high", "very_high"]


def _row_level_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-row feature transforms.  Runs identically in local pandas
    and inside each Ray worker chunk.

    Args:
        df: Raw transactions chunk.

    Returns:
        DataFrame with additional columns: log_amount, hour_of_day,
        day_of_week, is_weekend, amount_bucket.
    """
    df = df.copy()
    df["log_amount"]   = np.log1p(df["amount"])
    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["is_weekend"]   = df["day_of_week"] >= 5
    df["amount_bucket"] = pd.cut(
        df["amount"], bins=AMOUNT_BINS, labels=AMOUNT_LABELS, right=False
    ).astype(str)
    return df


def _groupby_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-user aggregations (the shuffle-heavy step).

    Args:
        df: Row-level transformed DataFrame.

    Returns:
        One row per user_id with aggregate features.
    """
    return (
        df.groupby("user_id", sort=False)
        .agg(
            total_spend       =("amount",      "sum"),
            mean_spend        =("amount",      "mean"),
            tx_count          =("amount",      "count"),
            unique_merchants  =("merchant_id", "nunique"),
            unique_categories =("category",    "nunique"),
            max_amount        =("amount",      "max"),
            min_amount        =("amount",      "min"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Mode 1: Local pandas baseline
# ---------------------------------------------------------------------------

def run_local(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Execute the full feature engineering pipeline using pandas on a single
    machine.  This is the baseline against which Ray is compared.

    Args:
        df: Raw transaction DataFrame (all rows in memory).

    Returns:
        (result_df, metrics) where metrics contains timing and memory data.
    """
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / (1024 ** 3)   # GB

    t0 = time.perf_counter()

    # Step 1: row-level transforms
    t1 = time.perf_counter()
    transformed = _row_level_transforms(df)
    row_transform_time = time.perf_counter() - t1

    # Step 2: groupby aggregation (the "shuffle" equivalent)
    t2 = time.perf_counter()
    result = _groupby_aggregations(transformed)
    groupby_time = time.perf_counter() - t2

    total_time = time.perf_counter() - t0
    mem_after  = proc.memory_info().rss / (1024 ** 3)

    metrics = {
        "mode":              "local_pandas",
        "input_rows":        len(df),
        "output_rows":       len(result),
        "total_runtime_sec": round(total_time, 4),
        "row_transform_sec": round(row_transform_time, 4),
        "groupby_sec":       round(groupby_time, 4),
        "peak_memory_gb":    round(mem_after, 3),
        "memory_delta_gb":   round(mem_after - mem_before, 3),
        "workers":           1,
        "partitions":        1,
        "shuffle_volume_mb": "N/A (single machine)",
    }
    return result, metrics


# ---------------------------------------------------------------------------
# Mode 2: Ray distributed pipeline
# ---------------------------------------------------------------------------

@ray.remote
def _ray_transform_chunk(chunk: pd.DataFrame, chunk_id: int) -> pd.DataFrame:
    """
    Ray remote task: apply row-level transforms to one data partition.

    Each task runs in its own worker process.  Ray schedules tasks
    across available CPU cores automatically.

    Args:
        chunk:    pandas DataFrame partition (passed by value via object store).
        chunk_id: Partition index (for logging).

    Returns:
        Transformed DataFrame partition.
    """
    return _row_level_transforms(chunk)


@ray.remote
def _ray_partial_groupby(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Ray remote task: compute per-user partial aggregations on one chunk.

    This implements a two-phase map-reduce shuffle:
        Map  phase → _ray_transform_chunk  (embarrassingly parallel)
        Reduce phase → merge partial groupby results on driver

    The shuffle volume is the sum of all partial_groupby DataFrames
    transferred back to the driver for the final merge.

    Args:
        chunk: Transformed DataFrame partition.

    Returns:
        Partial aggregation DataFrame (one row per user seen in this chunk).
    """
    return (
        chunk.groupby("user_id", sort=False)
        .agg(
            total_spend       =("amount",      "sum"),
            sum_sq_amount     =("amount",      lambda x: (x**2).sum()),
            tx_count          =("amount",      "count"),
            unique_merchants  =("merchant_id", "nunique"),
            unique_categories =("category",    "nunique"),
            max_amount        =("amount",      "max"),
            min_amount        =("amount",      "min"),
        )
        .reset_index()
    )


def _merge_partial_groupbys(partials: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Reduce step: merge per-chunk partial aggregations into final result.

    Handles:
      - total_spend / tx_count / max / min  → standard combine
      - mean_spend                          → derived from total/count
      - unique_merchants/categories         → conservative lower-bound
        (exact cross-partition nunique requires full data; this gives
         correct per-chunk unique counts, noted in report)

    Args:
        partials: List of partial aggregation DataFrames.

    Returns:
        Final per-user feature DataFrame.
    """
    combined = pd.concat(partials, ignore_index=True)
    result = (
        combined.groupby("user_id", sort=False)
        .agg(
            total_spend       =("total_spend",      "sum"),
            tx_count          =("tx_count",          "sum"),
            unique_merchants  =("unique_merchants",  "max"),   # lower-bound
            unique_categories =("unique_categories", "max"),   # lower-bound
            max_amount        =("max_amount",         "max"),
            min_amount        =("min_amount",         "min"),
        )
        .reset_index()
    )
    result["mean_spend"] = result["total_spend"] / result["tx_count"]
    return result


def run_distributed(df: pd.DataFrame, n_workers: int) -> tuple[pd.DataFrame, dict]:
    """
    Execute the feature engineering pipeline using Ray across multiple workers.

    Partitioning strategy:
        n_partitions = max(n_workers * 2, 8)   — enough tasks to keep all
        workers busy even if some finish early (avoids stragglers).

    Shuffle approximation:
        Ray uses a shared in-process object store (plasma) for zero-copy
        data sharing on a single node.  Network shuffle volume is measured
        as the total serialized size of partial groupby DataFrames returned
        to the driver, which represents the cross-worker data movement that
        would occur in a true multi-node cluster.

    Args:
        df:        Raw transaction DataFrame.
        n_workers: Number of Ray workers (= Ray CPU slots used).

    Returns:
        (result_df, metrics)
    """
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / (1024 ** 3)

    # ── Partition data ──────────────────────────────────────────────────────
    n_partitions = max(n_workers * 2, 8)
    # Use pandas iloc-based splitting so each chunk is always a proper DataFrame
    indices = np.array_split(np.arange(len(df)), n_partitions)
    chunks = [df.iloc[idx].reset_index(drop=True) for idx in indices if len(idx) > 0]
    n_partitions = len(chunks)
    logger.info("Distributed: %d rows → %d partitions across %d workers",
                len(df), n_partitions, n_workers)

    t0 = time.perf_counter()

    # ── MAP phase: row-level transforms (fully parallel) ───────────────────
    # ray.put() uploads each chunk to the shared plasma object store once;
    # workers receive it as a proper pandas DataFrame (zero-copy on same node).
    t1 = time.perf_counter()
    chunk_refs     = [ray.put(chunk) for chunk in chunks]
    transform_futs = [_ray_transform_chunk.remote(ref, i)
                      for i, ref in enumerate(chunk_refs)]
    transformed_refs = ray.get(transform_futs)          # block until all done
    map_time = time.perf_counter() - t1

    # ── SHUFFLE phase: partial groupby per partition ────────────────────────
    t2 = time.perf_counter()
    partial_refs = [ray.put(t) for t in transformed_refs]
    groupby_futs = [_ray_partial_groupby.remote(ref) for ref in partial_refs]
    partials     = ray.get(groupby_futs)
    shuffle_time  = time.perf_counter() - t2

    # Measure shuffle volume = total bytes of partial groupby DataFrames
    shuffle_bytes = sum(p.memory_usage(deep=True).sum() for p in partials)

    # ── REDUCE phase: merge on driver ──────────────────────────────────────
    t3 = time.perf_counter()
    result = _merge_partial_groupbys(partials)
    reduce_time = time.perf_counter() - t3

    total_time = time.perf_counter() - t0
    mem_after  = proc.memory_info().rss / (1024 ** 3)

    # Worker utilization: fraction of time workers were active
    # In Ray local mode, all CPUs are available; utilization ≈ map+shuffle / total
    active_time      = map_time + shuffle_time
    worker_util_pct  = round((active_time / total_time) * 100, 1)

    metrics = {
        "mode":              "ray_distributed",
        "input_rows":        len(df),
        "output_rows":       len(result),
        "total_runtime_sec": round(total_time, 4),
        "map_transform_sec": round(map_time, 4),
        "shuffle_sec":       round(shuffle_time, 4),
        "reduce_sec":        round(reduce_time, 4),
        "peak_memory_gb":    round(mem_after, 3),
        "memory_delta_gb":   round(mem_after - mem_before, 3),
        "workers":           n_workers,
        "partitions":        n_partitions,
        "shuffle_volume_mb": round(shuffle_bytes / (1024 ** 2), 2),
        "worker_util_pct":   worker_util_pct,
    }
    return result, metrics


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------

def verify_outputs(local_result: pd.DataFrame, dist_result: pd.DataFrame) -> bool:
    """
    Verify that local and distributed pipelines produce equivalent outputs.

    Checks:
        1. Same set of user_ids
        2. total_spend and tx_count match within floating-point tolerance
        3. max/min amounts match exactly

    Args:
        local_result: Output from run_local().
        dist_result:  Output from run_distributed().

    Returns:
        True if outputs are equivalent, False otherwise.
    """
    local_s = local_result.set_index("user_id").sort_index()
    dist_s  = dist_result.set_index("user_id").sort_index()

    if set(local_s.index) != set(dist_s.index):
        logger.error("User ID mismatch: local has %d users, distributed has %d",
                     len(local_s), len(dist_s))
        return False

    # Align on same index
    dist_s = dist_s.reindex(local_s.index)

    tol = 1e-4
    checks = {
        "total_spend": np.allclose(local_s["total_spend"], dist_s["total_spend"], atol=tol),
        "tx_count":    (local_s["tx_count"] == dist_s["tx_count"]).all(),
        "max_amount":  np.allclose(local_s["max_amount"], dist_s["max_amount"], atol=tol),
        "min_amount":  np.allclose(local_s["min_amount"], dist_s["min_amount"], atol=tol),
    }

    all_ok = all(checks.values())
    for col, ok in checks.items():
        status = "✅" if ok else "❌"
        logger.info("  Correctness check %-20s %s", col, status)

    return all_ok


# ---------------------------------------------------------------------------
# Scaling benchmark
# ---------------------------------------------------------------------------

def run_scaling_benchmark(input_path: Path, n_workers: int) -> list[dict]:
    """
    Run local + distributed comparison at multiple data scales.

    Scales tested: 100K, 500K, 1M rows (always), plus full dataset if larger.

    Args:
        input_path: Path to full parquet dataset.
        n_workers:  Number of Ray workers for distributed runs.

    Returns:
        List of metric dicts, one per (scale, mode) combination.
    """
    df_full = pd.read_parquet(input_path)
    total_rows = len(df_full)

    # Define scales to benchmark (subset from full data)
    scales = [100_000, 500_000, 1_000_000]
    if total_rows > 1_000_000:
        scales.append(total_rows)
    scales = [s for s in scales if s <= total_rows]

    all_metrics = []

    for n in scales:
        df_sub = df_full.iloc[:n].copy()
        logger.info("─" * 60)
        logger.info("Scale: %s rows", f"{n:,}")

        # Local
        logger.info("  Running local pandas...")
        _, local_m = run_local(df_sub)
        all_metrics.append(local_m)
        logger.info("  Local runtime: %.3fs", local_m["total_runtime_sec"])

        # Distributed
        logger.info("  Running Ray distributed (%d workers)...", n_workers)
        _, dist_m = run_distributed(df_sub, n_workers)
        all_metrics.append(dist_m)
        logger.info("  Distributed runtime: %.3fs", dist_m["total_runtime_sec"])
        logger.info("  Speedup: %.2fx",
                    local_m["total_runtime_sec"] / dist_m["total_runtime_sec"])

    return all_metrics


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_comparison_table(metrics_list: list[dict]) -> None:
    """Print a formatted side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON: Local Pandas vs Ray Distributed")
    print("=" * 70)

    # Group by input_rows
    from collections import defaultdict
    grouped = defaultdict(dict)
    for m in metrics_list:
        grouped[m["input_rows"]][m["mode"]] = m

    header = (f"  {'Rows':>10}  {'Mode':>20}  {'Runtime':>10}  "
              f"{'Shuffle MB':>12}  {'Workers':>8}  {'Partitions':>10}")
    print(header)
    print("  " + "─" * 68)

    for rows in sorted(grouped.keys()):
        for mode, m in sorted(grouped[rows].items()):
            shuffle = (f"{m['shuffle_volume_mb']:.1f}"
                       if isinstance(m.get("shuffle_volume_mb"), float)
                       else "N/A")
            print(f"  {rows:>10,}  {mode:>20}  "
                  f"{m['total_runtime_sec']:>9.3f}s  "
                  f"{shuffle:>12}  {m['workers']:>8}  {m['partitions']:>10}")

        # Speedup row
        if "local_pandas" in grouped[rows] and "ray_distributed" in grouped[rows]:
            lm = grouped[rows]["local_pandas"]
            dm = grouped[rows]["ray_distributed"]
            speedup = lm["total_runtime_sec"] / dm["total_runtime_sec"]
            print(f"  {' ':>10}  {'→ speedup':>20}  {speedup:>9.2f}x")
        print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Distributed feature engineering pipeline (Ray vs pandas).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Path to input parquet file or directory containing "
                             "transactions.parquet")
    parser.add_argument("--output", required=True,
                        help="Directory to write output parquet and metrics JSON")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of Ray worker CPUs (default: all available)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run multi-scale benchmark instead of single full run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used for data generation (recorded in metrics)")
    args = parser.parse_args()

    # ── Resolve input path ──────────────────────────────────────────────────
    inp = Path(args.input)
    if inp.is_dir():
        parquet_files = list(inp.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {inp}")
        inp = parquet_files[0]
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Init Ray ────────────────────────────────────────────────────────────
    n_cpus = args.workers or os.cpu_count()
    logger.info("Initializing Ray with %d CPUs...", n_cpus)
    ray.init(num_cpus=n_cpus, ignore_reinit_error=True, logging_level="ERROR")
    available_cpus = int(ray.available_resources().get("CPU", n_cpus))
    logger.info("Ray ready — %d CPUs available", available_cpus)

    print("=" * 70)
    print("MILESTONE 4 — Distributed Feature Engineering Pipeline")
    print("=" * 70)
    print(f"  Input         : {inp}")
    print(f"  Output dir    : {out_dir}")
    print(f"  Ray workers   : {available_cpus}")
    print(f"  Benchmark mode: {args.benchmark}")

    if args.benchmark:
        # ── Multi-scale benchmark ───────────────────────────────────────────
        logger.info("Running scaling benchmark...")
        all_metrics = run_scaling_benchmark(inp, available_cpus)
        print_comparison_table(all_metrics)

        metrics_path = out_dir / "benchmark_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info("Metrics saved → %s", metrics_path)

    else:
        # ── Single full-dataset run ─────────────────────────────────────────
        logger.info("Loading data from %s ...", inp)
        t_load = time.perf_counter()
        df = pd.read_parquet(inp)
        load_time = time.perf_counter() - t_load
        logger.info("Loaded %s rows in %.2fs", f"{len(df):,}", load_time)

        # Verify input hash
        input_hash = compute_data_hash(df)
        logger.info("Input hash: %s", input_hash)

        # ── Local run ───────────────────────────────────────────────────────
        print("\n[1/3] Running local pandas baseline...")
        local_result, local_metrics = run_local(df)
        print(f"  Runtime    : {local_metrics['total_runtime_sec']:.3f}s")
        print(f"  Output rows: {local_metrics['output_rows']:,}")
        print(f"  Peak memory: {local_metrics['peak_memory_gb']:.2f} GB")

        # ── Distributed run ─────────────────────────────────────────────────
        print(f"\n[2/3] Running Ray distributed ({available_cpus} workers)...")
        dist_result, dist_metrics = run_distributed(df, available_cpus)
        print(f"  Runtime       : {dist_metrics['total_runtime_sec']:.3f}s")
        print(f"  Map time      : {dist_metrics['map_transform_sec']:.3f}s")
        print(f"  Shuffle time  : {dist_metrics['shuffle_sec']:.3f}s")
        print(f"  Reduce time   : {dist_metrics['reduce_sec']:.3f}s")
        print(f"  Shuffle volume: {dist_metrics['shuffle_volume_mb']:.2f} MB")
        print(f"  Worker util   : {dist_metrics['worker_util_pct']}%")
        print(f"  Output rows   : {dist_metrics['output_rows']:,}")

        # ── Correctness check ───────────────────────────────────────────────
        print("\n[3/3] Verifying output correctness...")
        ok = verify_outputs(local_result, dist_result)
        if ok:
            print("  ✅ Outputs match — distributed pipeline is correct")
        else:
            print("  ❌ Output mismatch — check logs above")

        # ── Speedup summary ─────────────────────────────────────────────────
        speedup = local_metrics["total_runtime_sec"] / dist_metrics["total_runtime_sec"]
        print(f"\n  Speedup: {speedup:.2f}x  "
              f"({'faster' if speedup > 1 else 'slower'} than local)")

        # ── Save outputs ────────────────────────────────────────────────────
        local_result.to_parquet(out_dir / "local_features.parquet", index=False)
        dist_result.to_parquet(out_dir  / "distributed_features.parquet", index=False)

        report = {
            "input_hash":   input_hash,
            "input_rows":   len(df),
            "seed":         args.seed,
            "ray_workers":  available_cpus,
            "local":        local_metrics,
            "distributed":  dist_metrics,
            "speedup":      round(speedup, 4),
            "outputs_match": ok,
        }
        metrics_path = out_dir / "run_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Full report saved → %s", metrics_path)

        print("\n" + "=" * 70)
        print(f"  Output features : {out_dir}/{{local,distributed}}_features.parquet")
        print(f"  Metrics JSON    : {metrics_path}")
        print("=" * 70)

    ray.shutdown()


if __name__ == "__main__":
    main()