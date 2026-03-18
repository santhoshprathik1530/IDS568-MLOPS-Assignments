#!/usr/bin/env python3
"""
Synthetic Data Generator for Distributed Computing Benchmarks.
Milestone 4 - MLOps Course

Generates reproducible synthetic transaction datasets with configurable size,
optional data skew, and Parquet/CSV output.

Usage (module):
    from generate_data import generate_transactions, compute_data_hash
    df = generate_transactions(n_rows=10_000_000, seed=42)

Usage (CLI):
    python generate_data.py --rows 10000000 --seed 42 --output data/transactions.parquet
    python generate_data.py --rows 10000000 --seed 42 --output data/transactions.parquet --skew
    python generate_data.py --rows 1000 --output test_data/  # checklist sanity check format
"""

import argparse
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
CATEGORIES = ["food", "electronics", "clothing", "travel",
               "entertainment", "utilities", "healthcare", "education"]
CATEGORY_WEIGHTS = [0.25, 0.15, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07]
BASE_TIMESTAMP = pd.Timestamp("2025-01-01")
WINDOW_DAYS = 90


def generate_transactions(
    n_rows: int,
    seed: int = 42,
    skew: bool = False,
) -> pd.DataFrame:
    """
    Generate synthetic transaction data with realistic distributions.

    Args:
        n_rows:  Number of transaction rows to generate.
        seed:    Random seed for full reproducibility.
        skew:    If True, apply Zipf distribution to user_id so a small
                 number of users generate the majority of transactions.
                 This simulates realistic data skew and stresses
                 distributed shuffle operations.

    Returns:
        DataFrame with columns:
            user_id     – string  e.g. "u000042"
            amount      – float   exponential(scale=50), rounded to 2dp
            timestamp   – datetime64  uniform over 90 days from 2025-01-01
            category    – string  weighted across 8 categories
            merchant_id – string  e.g. "m00123"
    """
    rng = np.random.default_rng(seed)   # modern Generator (reproducible)

    # User pool: ~1% of rows, capped at [10, 100_000]
    n_users = max(10, min(100_000, n_rows // 100))
    user_ids = np.array([f"u{i:06d}" for i in range(n_users)])

    # Merchant pool: capped at [5, 10_000]
    n_merchants = max(5, min(10_000, n_rows // 500))
    merchant_ids = np.array([f"m{i:05d}" for i in range(n_merchants)])

    # User selection: uniform or Zipf-skewed
    if skew:
        # Zipf(a=1.5) → heavy tail; a few users dominate
        raw = rng.zipf(a=1.5, size=n_rows)
        user_indices = (raw - 1) % n_users          # clamp into [0, n_users)
        selected_users = user_ids[user_indices]
    else:
        selected_users = rng.choice(user_ids, size=n_rows, replace=True)

    # Timestamps: uniform seconds offset over 90-day window
    ts_offsets = rng.uniform(0, WINDOW_DAYS * 24 * 3600, size=n_rows)
    timestamps = [BASE_TIMESTAMP + pd.Timedelta(seconds=float(s)) for s in ts_offsets]

    data = {
        "user_id":     selected_users,
        "amount":      np.round(rng.exponential(scale=50.0, size=n_rows), 2),
        "timestamp":   timestamps,
        "category":    rng.choice(CATEGORIES, size=n_rows, p=CATEGORY_WEIGHTS),
        "merchant_id": rng.choice(merchant_ids, size=n_rows, replace=True),
    }

    return pd.DataFrame(data)


def compute_data_hash(df: pd.DataFrame) -> str:
    """
    Compute a deterministic SHA-256 fingerprint of a DataFrame.

    Same data + same seed always produces the same hash, enabling
    independent verification of reproducibility.

    Args:
        df: DataFrame to fingerprint.

    Returns:
        First 16 hex characters of the SHA-256 digest.
    """
    content = pd.util.hash_pandas_object(df).values
    return hashlib.sha256(content.tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction data for benchmarking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Support both --rows (checklist) and --n_rows (professor's original)
    parser.add_argument("--rows", "--n_rows", dest="rows", type=int, default=100_000,
                        help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path: file (.parquet/.csv) or directory "
                             "(saves transactions.parquet inside it)")
    parser.add_argument("--skew", action="store_true",
                        help="Apply Zipf skew to user distribution")
    args = parser.parse_args()

    logger.info("Generating %s rows  seed=%d  skew=%s",
                f"{args.rows:,}", args.seed, args.skew)
    start = time.perf_counter()
    df = generate_transactions(n_rows=args.rows, seed=args.seed, skew=args.skew)
    elapsed = time.perf_counter() - start

    data_hash = compute_data_hash(df)

    print(f"  Generated in : {elapsed:.3f}s")
    print(f"  Shape        : {df.shape}")
    print(f"  Hash (sha256): {data_hash}")
    print(f"  Unique users : {df['user_id'].nunique():,}")
    print(f"  Amount range : ${df['amount'].min():.2f} – ${df['amount'].max():.2f}")
    print(f"  Skew enabled : {args.skew}")

    if args.output:
        out = Path(args.output)
        # If output looks like a directory (no suffix or ends with /), save inside it
        if out.suffix == "" or str(args.output).endswith("/"):
            out.mkdir(parents=True, exist_ok=True)
            out = out / "transactions.parquet"
        else:
            out.parent.mkdir(parents=True, exist_ok=True)

        if out.suffix == ".csv":
            df.to_csv(out, index=False)
        else:
            df.to_parquet(out, index=False)

        size_mb = out.stat().st_size / (1024 * 1024)
        logger.info("Saved → %s  (%.1f MB)", out, size_mb)
        print(f"  Saved to     : {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()