#!/usr/bin/env python3
"""Generate local traffic for the monitoring dashboard."""

from __future__ import annotations

import argparse
import random
import time

import requests


ANSWER_QUERIES = [
    "What is retrieval-augmented generation?",
    "Why does chunk overlap matter?",
    "How should latency be measured in a RAG pipeline?",
    "What failure modes should be documented during evaluation?",
]

AGENT_TASKS = [
    "Explain how RAG reduces hallucination and summarize the answer in two sentences.",
    "Collect the important points about embedding model tradeoffs.",
    "Find the latency guidance and summarize what should be reported separately.",
]

ANOMALOUS_INPUTS = [
    "???",
    "!!!!!!!!!!!!!!!!",
    "help help help help help help help help",
    "what? why? how? where?",
]


def call_endpoint(base_url: str, endpoint: str, payload: dict) -> None:
    response = requests.post(f"{base_url}{endpoint}", json=payload, timeout=120)
    response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate traffic for the monitoring service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Service base URL.")
    parser.add_argument("--requests", type=int, default=40, help="Number of synthetic requests to send.")
    parser.add_argument("--sleep-ms", type=int, default=250, help="Delay between requests.")
    args = parser.parse_args()

    for _ in range(args.requests):
        mode = random.random()
        try:
            if mode < 0.55:
                question = random.choice(ANSWER_QUERIES)
                call_endpoint(args.base_url, "/answer", {"question": question, "top_k": 3})
            elif mode < 0.9:
                task = random.choice(AGENT_TASKS)
                call_endpoint(args.base_url, "/agent", {"task": task})
            else:
                bad = random.choice(ANOMALOUS_INPUTS)
                call_endpoint(args.base_url, "/answer", {"question": bad, "top_k": 3})
        except Exception:
            pass
        time.sleep(args.sleep_ms / 1000)


if __name__ == "__main__":
    main()
