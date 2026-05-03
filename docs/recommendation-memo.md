# Recommendation Memo

## Recommendation

Based on the simulated A/B test, my recommendation is to ship **Variant B**.

## Rationale

Variant B performed better on the two metrics that mattered most in this experiment: latency and groundedness.

Average latency improved from `2561.90 ms` in Variant A to `2138.49 ms` in Variant B. That is a reduction of roughly `423 ms`, and the associated p-value (`1.13e-71`) indicates that the result is not marginal.

Groundedness improved from `0.8319` to `0.8927`, which is an absolute lift of about `6.1` percentage points. This result was also statistically significant (`p = 2.35e-08`).

Task success moved in the same direction, increasing from `0.8842` to `0.9311` (`p = 2.95e-07`).

The error-rate guardrail did not show a meaningful regression. Variant B’s error rate was slightly higher (`0.0379` versus `0.0361`), but the difference was small and not statistically significant (`p = 0.7562`).

## Decision

Taken together, the results support a move to Variant B. It is faster, more grounded, and more successful on tasks, while the guardrail metric remains effectively unchanged.

## Qualification

This recommendation is based on offline simulation rather than real user traffic. If I were rolling the change out in a live environment, I would still want to watch:

- p95 latency
- citation quality
- error rate
- changes in query complexity that might affect a smaller retrieval budget

Within the scope of this assignment, however, the experiment gives a clear answer: Variant B is the better choice.
