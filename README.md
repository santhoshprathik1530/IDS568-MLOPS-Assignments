# LLM Inference Optimization with Batching and Caching

**IDS 568 MLOps – Milestone 5**

A production-ready LLM inference API implementing **dynamic request batching** and **intelligent caching** to optimize throughput and latency, with comprehensive benchmarking and governance analysis.

---

## Architecture

```
┌─────────────┐
│   Client     │
└──────┬──────┘
       │  POST /generate
       ▼
┌──────────────────────────────────────────┐
│            FastAPI Server                 │
│                                          │
│  ┌─────────┐   ┌──────────────────────┐  │
│  │  Cache   │──▶│  Cache HIT → return  │  │
│  │  Lookup  │   └──────────────────────┘  │
│  └────┬─────┘                             │
│       │ MISS                              │
│       ▼                                   │
│  ┌──────────────┐   ┌─────────────────┐  │
│  │   Dynamic     │──▶│  Batch Inference │  │
│  │   Batcher     │   │  (HuggingFace)  │  │
│  └──────────────┘   └─────────────────┘  │
│       │                                   │
│       ▼                                   │
│  ┌──────────┐                             │
│  │  Cache    │                            │
│  │  Store    │                            │
│  └──────────┘                             │
└──────────────────────────────────────────┘
```

## Features

- **Dynamic Request Batching**: Hybrid strategy (batch-size + timeout) to group concurrent requests, amortizing GPU compute
- **Intelligent Caching**: In-memory LRU cache with configurable TTL and max-entry limits
- **Privacy-Preserving**: SHA-256 hashed cache keys — no plaintext user identifiers stored
- **Async-First**: Built on `asyncio` with proper locking (`asyncio.Lock`, `asyncio.Semaphore`)
- **Comprehensive Benchmarks**: Automated suite comparing baseline, batching, caching, and combined optimizations
- **Visualization**: Auto-generated charts for latency, throughput, cache hit-rate, and scaling behavior

---

## Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- ~2 GB disk for the `gpt2` model (downloaded automatically on first run)
- (Optional) CUDA-capable GPU for faster inference

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ids568-milestone5-<your_netid>.git
cd ids568-milestone5-<your_netid>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Server

```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

The server will download the `gpt2` model on first launch (~500 MB). You should see:

```
INFO: Model loaded successfully.
INFO: Batcher initialized (batch=8, wait=100ms)
INFO: Cache initialized (TTL=300s, max=1000 entries)
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 5. Test a Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "temperature": 0.0, "max_tokens": 50}'
```

### 6. Run Benchmarks

In a **separate terminal** (while the server is running):

```bash
python benchmarks/run_benchmarks.py --url http://localhost:8000
```

This will:
- Run all benchmark configurations (baseline, batching, caching, combined)
- Test throughput at multiple concurrency levels
- Track cache hit-rate patterns
- Generate visualizations in `analysis/visualizations/`
- Save raw data in `benchmarks/results/`

---

## Configuration

All parameters are configurable via **environment variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `gpt2` | HuggingFace model identifier |
| `MAX_NEW_TOKENS` | `50` | Max tokens per generation |
| `DEVICE` | `cpu` | Inference device (`cpu`, `cuda`, `mps`) |
| `MAX_BATCH_SIZE` | `8` | Max requests per batch |
| `MAX_WAIT_MS` | `100` | Batch timeout (ms) |
| `BATCHING_ENABLED` | `true` | Enable/disable batching |
| `CACHE_TTL_SECONDS` | `300` | Cache entry TTL (seconds) |
| `CACHE_MAX_ENTRIES` | `1000` | Max cache entries |
| `CACHING_ENABLED` | `true` | Enable/disable caching |

Example with custom settings:

```bash
MAX_BATCH_SIZE=4 MAX_WAIT_MS=200 CACHE_TTL_SECONDS=600 \
  uvicorn src.server:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate` | Generate text from a prompt |
| `GET` | `/health` | Server health check |
| `GET` | `/metrics` | Cache & batching statistics |
| `POST` | `/cache/clear` | Clear all cached entries |
| `POST` | `/metrics/reset` | Reset metrics counters |

### POST /generate

**Request Body:**
```json
{
    "prompt": "What is machine learning?",
    "temperature": 0.0,
    "max_tokens": 50,
    "use_cache": true,
    "use_batching": true
}
```

**Response:**
```json
{
    "prompt": "What is machine learning?",
    "response": "Machine learning is a subset of artificial intelligence...",
    "cached": false,
    "cache_status": "MISS",
    "latency_ms": 245.3,
    "batch_size": 1
}
```

---

## Benchmark Suite

### Running Benchmarks

```bash
# Full benchmark suite
python benchmarks/run_benchmarks.py

# Custom options
python benchmarks/run_benchmarks.py \
    --url http://localhost:8000 \
    --requests 30 \
    --concurrency 10

# Standalone load test
python benchmarks/load_generator.py \
    --url http://localhost:8000 \
    --requests 50 \
    --concurrency 10 \
    --repeat-ratio 0.4 \
    --name "custom_test"
```

### Benchmark Scenarios

1. **Baseline** — No batching, no caching
2. **Batching Only** — Dynamic batching enabled
3. **Caching Only (Cold)** — Cache enabled, empty cache
4. **Caching Only (Warm)** — Cache enabled, pre-populated
5. **Batching + Caching** — Both optimizations active
6. **Throughput Scaling** — Multiple concurrency levels (1, 5, 10, 20, 30)
7. **Cache Hit-Rate Over Time** — Waves of increasing prompt repetition

---

## Project Structure

```
ids568-milestone5/
├── src/
│   ├── __init__.py          # Package marker
│   ├── server.py            # FastAPI inference server
│   ├── batching.py          # Dynamic request batching logic
│   ├── caching.py           # Cache with TTL, LRU, hashed keys
│   └── config.py            # Configuration management
├── benchmarks/
│   ├── run_benchmarks.py    # Benchmark orchestration
│   ├── load_generator.py    # Synthetic load generator
│   └── results/             # Raw benchmark data (JSON)
├── analysis/
│   ├── performance_report.pdf   # Performance analysis (3-4 pages)
│   ├── governance_memo.pdf      # Governance considerations (1 page)
│   └── visualizations/         # Auto-generated charts
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Key Design Decisions

### Batching Strategy
- **Hybrid approach**: Processes a batch when either the max batch size is reached OR the max wait timeout expires — whichever comes first
- Prevents individual requests from waiting indefinitely
- `asyncio.Lock` protects the shared pending-request queue

### Cache Design
- **Keys**: SHA-256 hash of `(prompt + model + temperature + max_tokens)` — no PII stored
- **Eviction**: Dual strategy — TTL-based expiration + LRU when at capacity
- **Bypass**: Non-deterministic requests (`temperature > 0`) skip cache entirely
- **OrderedDict** for O(1) LRU tracking

### Concurrency
- FastAPI's async endpoints + `asyncio.Lock` / `asyncio.Semaphore` for thread safety
- Model inference runs in a thread pool via `run_in_executor` to avoid blocking the event loop

---

## Governance Notes

See `analysis/governance_memo.pdf` for the full governance memo. Key points:

- **Privacy**: Cache keys are SHA-256 hashed; no user identifiers or raw prompts stored
- **Retention**: TTL ensures entries expire automatically; max-entry limit bounds memory
- **Compliance**: Supports GDPR "right to erasure" via `/cache/clear` endpoint
- **Misuse**: Rate limiting and cache bypass for sensitive prompts recommended

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root: `cd ids568-milestone5` |
| Server slow on first request | Model warmup is normal (~10-30s). Subsequent requests are faster. |
| `CUDA out of memory` | Reduce `MAX_BATCH_SIZE` or use `DEVICE=cpu` |
| Benchmark connection errors | Ensure the server is running before starting benchmarks |
| Port already in use | Change port: `uvicorn src.server:app --port 8001` |

---

## License

This project is submitted as part of IDS 568 (MLOps) at UIC. For academic use only.
