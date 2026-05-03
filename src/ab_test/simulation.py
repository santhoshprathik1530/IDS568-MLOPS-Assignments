#!/usr/bin/env python3
"""A/B test design and simulation for the final project.

Variant A represents the current RAG + agent system. Variant B represents a
proposed improvement that uses tighter retrieval settings and a stricter
citation-focused answer format. The point of the script is not to claim that the
variant is already deployed; it is to show a statistically valid offline
experiment design and recommendation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/.mplconfig").resolve()))

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 568
BASELINE_NAME = "A_current_topk3"
VARIANT_NAME = "B_topk2_citation_focused"


@dataclass
class Observation:
    request_id: str
    arm: str
    latency_ms: float
    grounded: int
    task_success: int
    error: int


def assign_arm(request_id: str) -> str:
    """Deterministic 50/50 assignment using request ID hashing."""

    digest = hashlib.md5(request_id.encode("utf-8")).hexdigest()
    return BASELINE_NAME if int(digest, 16) % 2 == 0 else VARIANT_NAME


def sample_size_two_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Approximate required sample size per arm for a difference in proportions."""

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    pooled = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2
    return math.ceil(numerator / denominator)


def simulate_observations(total_requests: int, seed: int) -> list[Observation]:
    """Generate synthetic outcomes for both experiment arms."""

    rng = np.random.default_rng(seed)
    observations: list[Observation] = []

    for idx in range(total_requests):
        request_id = f"req_{idx:05d}"
        arm = assign_arm(request_id)

        if arm == BASELINE_NAME:
            latency_ms = float(rng.lognormal(mean=7.80, sigma=0.32))
            grounded = int(rng.random() < 0.84)
            task_success = int(rng.random() < 0.89)
            error = int(rng.random() < 0.035)
        else:
            latency_ms = float(rng.lognormal(mean=7.63, sigma=0.28))
            grounded = int(rng.random() < 0.90)
            task_success = int(rng.random() < 0.93)
            error = int(rng.random() < 0.030)

        observations.append(
            Observation(
                request_id=request_id,
                arm=arm,
                latency_ms=latency_ms,
                grounded=grounded,
                task_success=task_success,
                error=error,
            )
        )
    return observations


def mean_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    mean = float(np.mean(values))
    sem = stats.sem(values)
    interval = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=sem)
    return float(interval[0]), float(interval[1])


def proportion_confidence_interval(successes: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    p_hat = float(np.mean(successes))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    n = len(successes)
    margin = z * math.sqrt((p_hat * (1 - p_hat)) / n)
    return max(0.0, p_hat - margin), min(1.0, p_hat + margin)


def two_proportion_z_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    p1 = np.mean(a)
    p2 = np.mean(b)
    pooled = (np.sum(a) + np.sum(b)) / (len(a) + len(b))
    se = math.sqrt(pooled * (1 - pooled) * ((1 / len(a)) + (1 / len(b))))
    z_score = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return float(z_score), float(p_value)


def evaluate(observations: list[Observation]) -> dict:
    a = [obs for obs in observations if obs.arm == BASELINE_NAME]
    b = [obs for obs in observations if obs.arm == VARIANT_NAME]

    a_latency = np.array([obs.latency_ms for obs in a])
    b_latency = np.array([obs.latency_ms for obs in b])
    a_grounded = np.array([obs.grounded for obs in a])
    b_grounded = np.array([obs.grounded for obs in b])
    a_success = np.array([obs.task_success for obs in a])
    b_success = np.array([obs.task_success for obs in b])
    a_error = np.array([obs.error for obs in a])
    b_error = np.array([obs.error for obs in b])

    _, latency_p = stats.ttest_ind(a_latency, b_latency, equal_var=False)
    grounded_z, grounded_p = two_proportion_z_test(a_grounded, b_grounded)
    success_z, success_p = two_proportion_z_test(a_success, b_success)
    error_z, error_p = two_proportion_z_test(a_error, b_error)

    return {
        "baseline_name": BASELINE_NAME,
        "variant_name": VARIANT_NAME,
        "sample_size_per_arm_target": sample_size_two_proportions(0.84, 0.90),
        "observed_counts": {
            "baseline": len(a),
            "variant": len(b),
            "total": len(observations),
        },
        "latency": {
            "baseline_mean_ms": float(np.mean(a_latency)),
            "variant_mean_ms": float(np.mean(b_latency)),
            "uplift_ms": float(np.mean(a_latency) - np.mean(b_latency)),
            "baseline_ci": mean_confidence_interval(a_latency),
            "variant_ci": mean_confidence_interval(b_latency),
            "p_value": float(latency_p),
        },
        "groundedness": {
            "baseline_rate": float(np.mean(a_grounded)),
            "variant_rate": float(np.mean(b_grounded)),
            "absolute_lift": float(np.mean(b_grounded) - np.mean(a_grounded)),
            "baseline_ci": proportion_confidence_interval(a_grounded),
            "variant_ci": proportion_confidence_interval(b_grounded),
            "z_score": grounded_z,
            "p_value": grounded_p,
        },
        "task_success": {
            "baseline_rate": float(np.mean(a_success)),
            "variant_rate": float(np.mean(b_success)),
            "absolute_lift": float(np.mean(b_success) - np.mean(a_success)),
            "baseline_ci": proportion_confidence_interval(a_success),
            "variant_ci": proportion_confidence_interval(b_success),
            "z_score": success_z,
            "p_value": success_p,
        },
        "error_rate": {
            "baseline_rate": float(np.mean(a_error)),
            "variant_rate": float(np.mean(b_error)),
            "absolute_change": float(np.mean(b_error) - np.mean(a_error)),
            "z_score": error_z,
            "p_value": error_p,
        },
    }


def render_visualizations(observations: list[Observation], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    a_latency = [obs.latency_ms for obs in observations if obs.arm == BASELINE_NAME]
    b_latency = [obs.latency_ms for obs in observations if obs.arm == VARIANT_NAME]
    a_grounded = np.mean([obs.grounded for obs in observations if obs.arm == BASELINE_NAME])
    b_grounded = np.mean([obs.grounded for obs in observations if obs.arm == VARIANT_NAME])
    a_success = np.mean([obs.task_success for obs in observations if obs.arm == BASELINE_NAME])
    b_success = np.mean([obs.task_success for obs in observations if obs.arm == VARIANT_NAME])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([a_latency, b_latency], tick_labels=["A", "B"])
    axes[0].set_title("Latency Distribution by Arm")
    axes[0].set_ylabel("milliseconds")

    metrics = ["groundedness", "task success"]
    a_values = [a_grounded, a_success]
    b_values = [b_grounded, b_success]
    x = np.arange(len(metrics))
    width = 0.35
    axes[1].bar(x - width / 2, a_values, width, label="A")
    axes[1].bar(x + width / 2, b_values, width, label="B")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Quality Metrics by Arm")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "ab_test_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the A/B test simulation.")
    parser.add_argument("--requests", type=int, default=4000, help="Number of synthetic requests to simulate.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--output-json", default="logs/ab_test_results.json", help="Path for JSON results.")
    parser.add_argument("--visualization-dir", default="visualizations", help="Directory for charts.")
    parser.add_argument("--dry-run", action="store_true", help="Print the experiment design without generating files.")
    args = parser.parse_args()

    if args.dry_run:
        result = {
            "baseline_name": BASELINE_NAME,
            "variant_name": VARIANT_NAME,
            "sample_size_per_arm_target": sample_size_two_proportions(0.84, 0.90),
            "planned_requests": args.requests,
        }
        print(json.dumps(result, indent=2))
        return

    observations = simulate_observations(args.requests, args.seed)
    result = evaluate(observations)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    render_visualizations(observations, Path(args.visualization_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
