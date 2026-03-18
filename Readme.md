# Milestone 4 — Distributed & Streaming Pipeline

**MLOps Course — Module 5**  
Framework: Ray | Language: Python 3.9+

---

## Repository Structure

```
ids568-milestone4-[netid]/
├── pipeline.py          # Distributed feature engineering (Ray vs pandas)
├── generate_data.py     # Synthetic data generator (10M+ rows)
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── REPORT.md            # Performance analysis and architecture evaluation
```

---

## Prerequisites

- Python 3.9, 3.10, or 3.11
- ~3 GB free RAM (for 10M row benchmark)
- ~500 MB disk space

No Docker required. Everything runs locally.

---

## Setup (5 Steps)

### 1. Clone the repository

```bash
git clone https://github.com/[your-github]/ids568-milestone4-[netid].git
cd ids568-milestone4-[netid]
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
# Quick test (1,000 rows — runs in seconds)
python generate_data.py --rows 1000 --seed 42 --output test_data/

# Full benchmark dataset (10M rows — takes ~30-60s)
python generate_data.py --rows 10000000 --seed 42 --output data/
```

Expected output:
```
  Generated in : 38.4s
  Shape        : (10000000, 5)
  Hash (sha256): ecd55642f6493866
  Unique users : 100,000
  Amount range : $0.01 – $847.23
```

**Reproducibility:** The hash `ecd55642f6493866` is deterministic. Running the
same command again on any machine with the same numpy/pandas versions will
produce the identical hash.

### 5. Run the pipeline

```bash
# Quick correctness test on 1,000 rows
python pipeline.py --input test_data/ --output test_output/

# Full run (local vs distributed, single pass)
python pipeline.py --input data/ --output results/

# Multi-scale benchmark (100K, 500K, 1M, 10M rows)
python pipeline.py --input data/ --output results/ --benchmark

# Control the number of Ray workers (default: all CPU cores)
python pipeline.py --input data/ --output results/ --workers 4
```

---

## Expected Output

### `pipeline.py` (single full run)

```
======================================================================
MILESTONE 4 — Distributed Feature Engineering Pipeline
======================================================================
  Input         : data/transactions.parquet
  Output dir    : results/
  Ray workers   : 8
  Benchmark mode: False

[1/3] Running local pandas baseline...
  Runtime    : 18.432s
  Output rows: 100,000
  Peak memory: 2.41 GB

[2/3] Running Ray distributed (8 workers)...
  Runtime       : 6.817s
  Map time      : 3.204s
  Shuffle time  : 2.891s
  Reduce time   : 0.722s
  Shuffle volume: 12.34 MB
  Worker util   : 89.2%
  Output rows   : 100,000

[3/3] Verifying output correctness...
  ✅ Outputs match — distributed pipeline is correct

  Speedup: 2.70x  (faster than local)
```

*Note: exact numbers vary by machine. Hash values are deterministic.*

---

## Outputs

After running the pipeline, `results/` contains:

| File | Description |
|------|-------------|
| `local_features.parquet` | Per-user features from pandas baseline |
| `distributed_features.parquet` | Per-user features from Ray pipeline |
| `run_metrics.json` | Timing, memory, shuffle, and speedup metrics |
| `benchmark_metrics.json` | Multi-scale metrics (if `--benchmark` used) |

---

## Reproducibility Verification

Run these two commands and confirm the hashes match:

```bash
python generate_data.py --rows 10000 --seed 42 --output /tmp/run1/
python generate_data.py --rows 10000 --seed 42 --output /tmp/run2/

# Should print the same hash both times
```

To verify pipeline determinism:

```bash
python pipeline.py --input /tmp/run1/ --output /tmp/out1/
python pipeline.py --input /tmp/run1/ --output /tmp/out2/

# Compare output files — should be identical
python -c "
import pandas as pd
a = pd.read_parquet('/tmp/out1/local_features.parquet').sort_values('user_id')
b = pd.read_parquet('/tmp/out2/local_features.parquet').sort_values('user_id')
print('Match:', a.equals(b))
"
```

---

## Feature Engineering Logic

Both local (pandas) and distributed (Ray) modes apply the identical transforms:

| Feature | Formula |
|---------|---------|
| `log_amount` | `log1p(amount)` |
| `hour_of_day` | `timestamp.hour` |
| `day_of_week` | `timestamp.dayofweek` |
| `is_weekend` | `day_of_week >= 5` |
| `amount_bucket` | `cut(amount, [0,25,100,500,∞])` |
| `total_spend` | `sum(amount)` per user |
| `mean_spend` | `mean(amount)` per user |
| `tx_count` | `count(amount)` per user |
| `unique_merchants` | `nunique(merchant_id)` per user |
| `unique_categories` | `nunique(category)` per user |

---

## Ray Architecture

```
Driver (pipeline.py)
    │
    ├── ray.put(chunk_0) ──► Object Store (shared memory)
    ├── ray.put(chunk_1)
    │   ...
    │
    ├── _ray_transform_chunk.remote(chunk_0) ──► Worker 0
    ├── _ray_transform_chunk.remote(chunk_1) ──► Worker 1   ← MAP phase
    │   ...                                          (parallel)
    │
    ├── _ray_partial_groupby.remote(transformed_0) ──► Worker 0
    ├── _ray_partial_groupby.remote(transformed_1) ──► Worker 1  ← SHUFFLE phase
    │   ...
    │
    └── _merge_partial_groupbys(partials) ──► Driver   ← REDUCE phase
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: ray` | Activate your venv: `source venv/bin/activate` |
| `ray.init()` hangs | Kill stale Ray: `ray stop && ray start --head` |
| Out of memory at 10M rows | Reduce workers: `--workers 2` or use `--benchmark` |
| Parquet write fails | Ensure pyarrow installed: `pip install pyarrow` |
| Hash mismatch on same seed | Check numpy version matches (`numpy>=1.24.0`) |
| Low speedup | Normal on machines with <4 cores; overhead > benefit at small scale |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ray` | ≥ 2.9.3 | Distributed task execution |
| `pandas` | ≥ 2.0.0 | Data manipulation |
| `numpy` | ≥ 1.24.0 | Numerical operations |
| `pyarrow` | ≥ 14.0.0 | Parquet I/O |
| `psutil` | ≥ 5.9.0 | Memory / CPU metrics |