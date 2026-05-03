#!/usr/bin/env python3
"""Drift and integrity analysis for the final project.

The project system is a RAG + agent workflow, so the most meaningful "features"
to monitor are request characteristics and response-side support signals rather
than classic training columns alone.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REFERENCE_QUERIES = [
    "What is retrieval-augmented generation?",
    "Why does chunk overlap matter in retrieval?",
    "How should latency be measured in a RAG pipeline?",
    "What failure modes should be documented during evaluation?",
    "How do embedding models support semantic retrieval?",
    "When should an agent retrieve instead of summarize?",
    "What makes an answer grounded instead of hallucinatory?",
    "Why is FAISS IndexFlatL2 acceptable for a small corpus?",
]

PRODUCTION_QUERIES = [
    "what is rag and can you explain how it really reduces hallucination in long answers with examples?",
    "why does chunk overlap matter and what happens if I set it too high or too low?",
    "Summarize the most important failure modes in this pipeline and list them as bullets.",
    "Explain latency metrics for retrieval generation and end-to-end response time in one answer please",
    "what are the biggest risks if the retrieved context is stale or partially wrong?",
    "Can the agent extract evidence and summarize it in one shot?",
    "what happens if I ask several questions at once? how does the system handle that? why?",
    "How would drift in user questions affect retrieval quality over time?",
    "What if the prompt contains unusual punctuation???? why would that matter????",
    "Give a checklist for groundedness, latency, citations, and retrieval confidence.",
    "How do I know when to retrain or re-index the system if user behavior changes?",
    "Explain all the important details about governance, monitoring, compliance, and the risk matrix in a single response.",
]


@dataclass
class DriftMetric:
    feature: str
    reference_mean: float
    production_mean: float
    drift_score: float
    p_value: float


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def query_features(queries: list[str], window_label: str) -> pd.DataFrame:
    rows = []
    wh_words = {"what", "why", "how", "when", "where", "who", "which"}
    for text in queries:
        tokens = tokenize(text)
        token_count = len(tokens)
        char_count = len(text)
        punctuation_count = sum(1 for ch in text if ch in "?!.,:")
        multi_question = int(text.count("?") >= 2)
        wh_count = sum(1 for token in tokens if token in wh_words)
        avg_token_length = float(np.mean([len(token) for token in tokens])) if tokens else 0.0
        lexical_diversity = len(set(tokens)) / token_count if token_count else 0.0
        rows.append(
            {
                "window": window_label,
                "query": text,
                "token_count": token_count,
                "char_count": char_count,
                "punctuation_count": punctuation_count,
                "multi_question": multi_question,
                "wh_count": wh_count,
                "avg_token_length": avg_token_length,
                "lexical_diversity": lexical_diversity,
            }
        )
    return pd.DataFrame(rows)


def population_stability_index(reference: pd.Series, production: pd.Series, bins: int = 5) -> float:
    ref = reference.to_numpy()
    prod = production.to_numpy()
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=quantiles)
    prod_hist, _ = np.histogram(prod, bins=quantiles)
    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, None)
    prod_pct = np.clip(prod_hist / max(prod_hist.sum(), 1), 1e-6, None)
    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))


def evaluate_drift(reference_df: pd.DataFrame, production_df: pd.DataFrame) -> list[DriftMetric]:
    metrics: list[DriftMetric] = []
    features = [
        "token_count",
        "char_count",
        "punctuation_count",
        "multi_question",
        "wh_count",
        "avg_token_length",
        "lexical_diversity",
    ]

    for feature in features:
        ref = reference_df[feature]
        prod = production_df[feature]
        psi = population_stability_index(ref, prod)
        _, p_value = stats.ks_2samp(ref, prod)
        metrics.append(
            DriftMetric(
                feature=feature,
                reference_mean=float(ref.mean()),
                production_mean=float(prod.mean()),
                drift_score=psi,
                p_value=float(p_value),
            )
        )
    return metrics


def anomaly_summary(production_df: pd.DataFrame) -> dict:
    anomalies = {
        "multi_question_requests": int(production_df["multi_question"].sum()),
        "high_punctuation_requests": int((production_df["punctuation_count"] >= 4).sum()),
        "very_long_queries": int((production_df["char_count"] >= 100).sum()),
        "low_diversity_queries": int((production_df["lexical_diversity"] <= 0.55).sum()),
    }
    return anomalies


def render_drift_visualizations(reference_df: pd.DataFrame, production_df: pd.DataFrame, metrics: list[DriftMetric], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Drift and Integrity Monitoring", fontsize=16)

    axes[0, 0].hist(reference_df["token_count"], alpha=0.65, label="reference")
    axes[0, 0].hist(production_df["token_count"], alpha=0.65, label="production")
    axes[0, 0].set_title("Token Count Distribution")
    axes[0, 0].set_xlabel("tokens")
    axes[0, 0].legend()

    axes[0, 1].hist(reference_df["punctuation_count"], alpha=0.65, label="reference")
    axes[0, 1].hist(production_df["punctuation_count"], alpha=0.65, label="production")
    axes[0, 1].set_title("Punctuation Distribution")
    axes[0, 1].set_xlabel("punctuation marks")
    axes[0, 1].legend()

    axes[1, 0].plot(reference_df["char_count"].to_list(), marker="o", label="reference")
    axes[1, 0].plot(production_df["char_count"].to_list(), marker="o", label="production")
    axes[1, 0].set_title("Character Count by Time Window")
    axes[1, 0].set_xlabel("request index")
    axes[1, 0].set_ylabel("characters")
    axes[1, 0].legend()

    top_features = sorted(metrics, key=lambda item: item.drift_score, reverse=True)[:5]
    axes[1, 1].bar([item.feature for item in top_features], [item.drift_score for item in top_features], color="#E45756")
    axes[1, 1].set_title("Top PSI Drift Scores")
    axes[1, 1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_dir / "drift_overview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run drift and integrity analysis.")
    parser.add_argument("--output-json", default="logs/drift_analysis.json")
    parser.add_argument("--visualization-dir", default="visualizations")
    args = parser.parse_args()

    reference_df = query_features(REFERENCE_QUERIES, "reference")
    production_df = query_features(PRODUCTION_QUERIES, "production")
    metrics = evaluate_drift(reference_df, production_df)
    anomalies = anomaly_summary(production_df)

    render_drift_visualizations(reference_df, production_df, metrics, Path(args.visualization_dir))

    payload = {
        "reference_size": len(reference_df),
        "production_size": len(production_df),
        "metrics": [asdict(metric) for metric in metrics],
        "anomalies": anomalies,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
