#!/usr/bin/env python3
"""Run a local monitoring demo and export dashboard evidence.

This script exercises the instrumented service with simulated traffic, saves a
Prometheus metrics snapshot, and renders a dashboard-style PNG from the
observed request logs. It is meant to produce the evidence required for
Component 1 without depending on a live Grafana instance during grading.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.monitoring.service import (
    AgentRequest,
    AnswerRequest,
    app,
    answer,
    generate_latest,
    metrics,
    run_agent,
)


ANSWER_QUERIES = [
    "What is retrieval-augmented generation and why does it reduce hallucination?",
    "How should latency be measured in a RAG pipeline?",
    "Why does chunk overlap matter in document retrieval?",
    "What failure modes should be documented during RAG evaluation?",
    "How do embedding models help semantic retrieval?",
]

AGENT_TASKS = [
    "Explain how RAG reduces hallucination, then summarize the answer in two sentences.",
    "Collect the important points about embedding model tradeoffs.",
    "Find the latency guidance and summarize what should be reported separately.",
    "Explain when an agent should retrieve before summarizing.",
]

ANOMALOUS_INPUTS = [
    "???",
    "!!!!!!!!!!!!!!!!",
    "help help help help help help help help",
    "what? why? how? where?",
]


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def simulate_request() -> dict[str, Any]:
    """Route one synthetic request through the instrumented endpoints."""

    mode = random.random()
    if mode < 0.6:
        question = random.choice(ANSWER_QUERIES)
        payload = AnswerRequest(question=question, top_k=3)
        result = answer(payload)
        return {
            "endpoint": "answer",
            "status": "ok",
            "text": question,
            **result,
        }

    if mode < 0.85:
        task = random.choice(AGENT_TASKS)
        payload = AgentRequest(task=task)
        result = run_agent(payload)
        return {
            "endpoint": "agent",
            "status": "ok",
            "text": task,
            "retrieval_latency_ms": None,
            "generation_latency_ms": None,
            "end_to_end_latency_ms": None,
            **result,
        }

    question = random.choice(ANOMALOUS_INPUTS)
    payload = AnswerRequest(question=question, top_k=3)
    result = answer(payload)
    return {
        "endpoint": "answer",
        "status": "ok",
        "text": question,
        **result,
    }


def write_dashboard_png(records: list[dict[str, Any]], output_path: Path) -> None:
    """Render a dashboard-style PNG from simulated traffic records."""

    answer_records = [record for record in records if record["endpoint"] == "answer"]
    agent_records = [record for record in records if record["endpoint"] == "agent"]
    endpoint_counts = Counter(f"{record['endpoint']}:{record['status']}" for record in records)
    integrity_counts = Counter(
        flag for record in records for flag in record.get("integrity_flags", [])
    )
    drift_values = [record.get("drift_score", 0.0) for record in records]
    retrieval_latencies = [
        record["retrieval_latency_ms"]
        for record in answer_records
        if record.get("retrieval_latency_ms") is not None
    ]
    generation_latencies = [
        record["generation_latency_ms"]
        for record in answer_records
        if record.get("generation_latency_ms") is not None
    ]
    step_counts = [record.get("step_count", 0) for record in agent_records]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("IDS 568 Final Project Monitoring Snapshot", fontsize=16)

    # Request volume
    ax = axes[0, 0]
    labels = list(endpoint_counts.keys())
    values = [endpoint_counts[label] for label in labels]
    ax.bar(labels, values, color=["#4C78A8", "#72B7B2", "#F58518"][: len(labels)])
    ax.set_title("Request Volume by Endpoint")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=20)

    # Latency comparison
    ax = axes[0, 1]
    ax.plot(retrieval_latencies, label="retrieval latency", marker="o", linewidth=1.5)
    ax.plot(generation_latencies, label="generation latency", marker="o", linewidth=1.5)
    ax.set_title("Answer Endpoint Latency")
    ax.set_ylabel("milliseconds")
    ax.set_xlabel("request index")
    ax.legend()

    # Drift score over traffic
    ax = axes[1, 0]
    ax.plot(drift_values, color="#E45756", marker="o", linewidth=1.5)
    ax.axhline(1.0, linestyle="--", color="gray", label="watch threshold")
    ax.set_title("Rolling Query Drift Score")
    ax.set_ylabel("z-score distance")
    ax.set_xlabel("request index")
    ax.legend()

    # Integrity and agent behavior
    ax = axes[1, 1]
    integrity_labels = list(integrity_counts.keys()) or ["none"]
    integrity_values = [integrity_counts[label] for label in integrity_labels] if integrity_counts else [0]
    ax.bar(integrity_labels, integrity_values, color="#54A24B", alpha=0.7, label="integrity anomalies")
    if step_counts:
        ax.plot(range(len(step_counts)), step_counts, color="#B279A2", marker="o", label="agent step count")
    ax.set_title("Integrity Signals and Agent Steps")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(records: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    """Save a JSON summary that can be cited in the interpretation document."""

    answer_records = [record for record in records if record["endpoint"] == "answer"]
    agent_records = [record for record in records if record["endpoint"] == "agent"]
    generation_latencies = [
        record["generation_latency_ms"]
        for record in answer_records
        if record.get("generation_latency_ms") is not None
    ]
    retrieval_latencies = [
        record["retrieval_latency_ms"]
        for record in answer_records
        if record.get("retrieval_latency_ms") is not None
    ]
    drift_values = [record.get("drift_score", 0.0) for record in records]
    integrity_counts = Counter(
        flag for record in records for flag in record.get("integrity_flags", [])
    )
    low_support = sum(1 for record in answer_records if "[source:" not in record.get("answer", ""))

    summary = {
        "total_requests": len(records),
        "answer_requests": len(answer_records),
        "agent_requests": len(agent_records),
        "avg_retrieval_latency_ms": round(statistics.mean(retrieval_latencies), 2) if retrieval_latencies else None,
        "avg_generation_latency_ms": round(statistics.mean(generation_latencies), 2) if generation_latencies else None,
        "max_generation_latency_ms": round(max(generation_latencies), 2) if generation_latencies else None,
        "avg_agent_steps": round(statistics.mean(record.get("step_count", 0) for record in agent_records), 2)
        if agent_records
        else None,
        "max_drift_score": round(max(drift_values), 3) if drift_values else None,
        "integrity_counts": dict(integrity_counts),
        "requests_without_source_citations": low_support,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local monitoring demo.")
    parser.add_argument("--requests", type=int, default=36, help="Number of synthetic requests to execute.")
    parser.add_argument("--seed", type=int, default=568, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)
    screenshots_dir = ensure_dir("screenshots")
    logs_dir = ensure_dir("logs")

    records = [simulate_request() for _ in range(args.requests)]
    metrics_snapshot = generate_latest()
    (logs_dir / "monitoring_metrics.prom").write_bytes(metrics_snapshot)
    summary = write_summary(records, logs_dir / "monitoring_summary.json")
    write_dashboard_png(records, screenshots_dir / "dashboard-monitoring.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
