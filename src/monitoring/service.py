#!/usr/bin/env python3
"""Instrumented API for the final project.

This service wraps the Milestone 6 RAG pipeline and agent so they can be observed
like a small production system. The focus is not scale; it is visibility.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.common.agent_controller import EvaluationTask, MultiToolAgent
from src.common.rag_pipeline import RAGPipeline, build_evaluation_queries

app = FastAPI(title="IDS568 Final Project Service", version="0.1.0")


REQUEST_COUNT = Counter(
    "ids568_requests_total",
    "Total number of requests served by endpoint and status.",
    labelnames=("endpoint", "status"),
)
REQUEST_LATENCY = Histogram(
    "ids568_request_latency_ms",
    "End-to-end request latency in milliseconds.",
    labelnames=("endpoint",),
    buckets=(25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 90000),
)
RETRIEVAL_LATENCY = Histogram(
    "ids568_retrieval_latency_ms",
    "Latency of retrieval operations in milliseconds.",
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000),
)
GENERATION_LATENCY = Histogram(
    "ids568_generation_latency_ms",
    "Latency of generation operations in milliseconds.",
    buckets=(100, 250, 500, 1000, 2500, 5000, 10000, 30000, 90000),
)
AGENT_STEP_COUNT = Histogram(
    "ids568_agent_step_count",
    "Number of tool-selection steps taken by the agent.",
    buckets=(1, 2, 3, 4, 5, 6),
)
ERROR_COUNT = Counter(
    "ids568_errors_total",
    "Total number of failed requests by endpoint.",
    labelnames=("endpoint", "reason"),
)
INPUT_ANOMALY_COUNT = Counter(
    "ids568_input_integrity_anomalies_total",
    "Count of input integrity anomalies.",
    labelnames=("kind",),
)
DRIFT_SCORE = Gauge(
    "ids568_query_drift_score",
    "Simple rolling drift score based on query-length deviation from baseline.",
)
RETRIEVAL_SCORE = Gauge(
    "ids568_retrieval_confidence_score",
    "Average retrieval similarity score for the latest request.",
)
KNOWLEDGE_GAP_COUNT = Counter(
    "ids568_low_support_requests_total",
    "Requests where retrieval support looked weak.",
    labelnames=("endpoint",),
)


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=3, ge=1, le=5)


class AgentRequest(BaseModel):
    task: str = Field(..., min_length=5)


pipeline = RAGPipeline()
pipeline.ingest()
agent = MultiToolAgent(pipeline)

baseline_query_lengths = [len(case.question) for case in build_evaluation_queries()]
baseline_mean = statistics.mean(baseline_query_lengths)
baseline_stdev = statistics.pstdev(baseline_query_lengths) or 1.0
recent_query_lengths: deque[int] = deque(maxlen=200)


def classify_integrity(question: str) -> list[str]:
    """Return lightweight integrity flags for a request."""

    flags: list[str] = []
    stripped = question.strip()
    if len(stripped) < 8:
        flags.append("too_short")
    if len(stripped) > 500:
        flags.append("too_long")
    if stripped.count("?") > 3:
        flags.append("multi_question")
    if any(token * 4 in stripped.lower() for token in ["!", "?", "."]):
        flags.append("repeated_punctuation")
    if len(set(stripped.lower())) < max(5, len(stripped) // 8):
        flags.append("low_character_diversity")
    return flags


def update_drift_metrics(question: str) -> float:
    """Update the rolling drift gauge and return the current score."""

    recent_query_lengths.append(len(question))
    current_mean = statistics.mean(recent_query_lengths)
    drift_score = abs(current_mean - baseline_mean) / baseline_stdev
    DRIFT_SCORE.set(drift_score)
    return drift_score


def record_integrity_flags(question: str) -> list[str]:
    flags = classify_integrity(question)
    for flag in flags:
        INPUT_ANOMALY_COUNT.labels(kind=flag).inc()
    return flags


def weak_support(similarity_scores: list[float]) -> bool:
    if not similarity_scores:
        return True
    return statistics.mean(similarity_scores) < 0.2


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "ids568-final-project"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/answer")
def answer(payload: AnswerRequest) -> dict[str, Any]:
    start = time.perf_counter()
    flags = record_integrity_flags(payload.question)
    drift_score = update_drift_metrics(payload.question)
    try:
        result = pipeline.answer(payload.question, top_k=payload.top_k)
    except Exception as exc:
        ERROR_COUNT.labels(endpoint="answer", reason="pipeline_error").inc()
        REQUEST_COUNT.labels(endpoint="answer", status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    retrieval_scores = [item.similarity_score for item in result.retrieved_chunks]
    RETRIEVAL_SCORE.set(statistics.mean(retrieval_scores))
    RETRIEVAL_LATENCY.observe(result.retrieval_latency_ms)
    GENERATION_LATENCY.observe(result.generation_latency_ms)
    if weak_support(retrieval_scores):
        KNOWLEDGE_GAP_COUNT.labels(endpoint="answer").inc()

    elapsed_ms = (time.perf_counter() - start) * 1000
    REQUEST_COUNT.labels(endpoint="answer", status="ok").inc()
    REQUEST_LATENCY.labels(endpoint="answer").observe(elapsed_ms)
    return {
        "question": payload.question,
        "answer": result.answer,
        "sources": [item.chunk.source for item in result.retrieved_chunks],
        "retrieval_latency_ms": result.retrieval_latency_ms,
        "generation_latency_ms": result.generation_latency_ms,
        "end_to_end_latency_ms": result.end_to_end_latency_ms,
        "integrity_flags": flags,
        "drift_score": drift_score,
    }


@app.post("/agent")
def run_agent(payload: AgentRequest) -> dict[str, Any]:
    start = time.perf_counter()
    flags = record_integrity_flags(payload.task)
    drift_score = update_drift_metrics(payload.task)
    try:
        trace = agent.run_task(EvaluationTask("adhoc", payload.task, []))
    except Exception as exc:
        ERROR_COUNT.labels(endpoint="agent", reason="agent_error").inc()
        REQUEST_COUNT.labels(endpoint="agent", status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    REQUEST_COUNT.labels(endpoint="agent", status="ok").inc()
    REQUEST_LATENCY.labels(endpoint="agent").observe((time.perf_counter() - start) * 1000)
    AGENT_STEP_COUNT.observe(len(trace.steps))
    return {
        "task": payload.task,
        "status": trace.status,
        "step_count": len(trace.steps),
        "final_answer": trace.final_answer,
        "integrity_flags": flags,
        "drift_score": drift_score,
        "steps": trace.to_dict()["steps"],
    }
