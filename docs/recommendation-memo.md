# Recommendation Memo

## Recommendation

Based on the simulated A/B results, I would recommend **shipping Variant B**.

## Reasoning

Variant B outperformed the current system on the two metrics that matter most for this experiment: latency and groundedness.

The simulated mean latency improved from `2561.90 ms` in Variant A to `2138.49 ms` in Variant B, a reduction of about `423 ms`. The p-value for that difference was effectively zero (`1.13e-71`), so the latency result is not borderline.

Groundedness also improved in a meaningful way. Variant A produced a groundedness rate of `0.8319`, while Variant B reached `0.8927`, for an absolute lift of about `6.1 percentage points`. That difference was also statistically significant (`p = 2.35e-08`).

Task success followed the same pattern, improving from `0.8842` to `0.9311` (`p = 2.95e-07`).

The error-rate guardrail did not show a meaningful regression. The simulated error rate was slightly higher in Variant B (`0.0379` vs `0.0361`), but the difference was small and not statistically significant (`p = 0.7562`).

## Decision

Because Variant B is:

- faster
- more grounded
- more successful on tasks
- not meaningfully worse on the guardrail metric

I would move forward with Variant B.

## Caveat

This recommendation is based on an offline simulation rather than live production traffic. Before a real rollout, I would still watch:

- p95 latency
- citation or grounding quality
- error rate
- any drift in query mix that could change the behavior of the smaller retrieval budget

Even so, the experiment as designed gives a clear answer: **ship B rather than keep A unchanged**.
