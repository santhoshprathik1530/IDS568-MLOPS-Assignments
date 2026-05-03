# Experiment Specification

## Objective

For the A/B test component, I framed the experiment around a practical decision for this system: whether to keep the current RAG workflow unchanged or move to a slightly tighter retrieval configuration with more explicit citation-oriented output formatting.

The two arms are:

- **Variant A**
  - current configuration
  - top-k retrieval of 3
  - current answer format
- **Variant B**
  - proposed configuration
  - top-k retrieval of 2
  - stronger citation-oriented answer formatting

The motivation for Variant B is straightforward. A smaller retrieval set may reduce latency, and the stricter answer format may improve groundedness by pushing the model to stay closer to the retrieved evidence.

## Hypothesis

My primary hypothesis is that Variant B will reduce response latency while maintaining or improving groundedness.

My secondary hypothesis is that Variant B will improve overall task success without creating a meaningful increase in the error rate.

## Success Metrics

The primary metrics are:

- average end-to-end latency
- groundedness rate

The secondary metrics are:

- task success rate
- error rate

I treated error rate as a guardrail metric. A faster model behavior would not be enough to justify deployment if it came with a noticeable increase in failures.

## Randomization Method

Requests are assigned to Variant A or Variant B with a deterministic 50/50 split based on a hash of the request ID. I used deterministic assignment because it is easy to reproduce, easy to audit, and sufficient for an offline simulation setting.

## Sample Size and Duration

I based the sample-size calculation on groundedness. Using a baseline groundedness of `0.84`, a target improvement to `0.90`, a significance level of `0.05`, and power of `0.80`, the script estimates:

- required sample size: `492` requests per arm

For the actual simulation, I used `4000` total synthetic requests:

- Variant A: `1969`
- Variant B: `2031`

This comfortably exceeds the minimum and gives stable rate estimates. If I were running the same design with live traffic at roughly 300 eligible requests per hour, the minimum total sample would be reachable in a little over three hours, although in practice I would run longer to capture different traffic periods and monitor the guardrail metric.

## Statistical Evaluation

For latency, I used Welch’s two-sample t-test and confidence intervals on the mean.

For groundedness, task success, and error rate, I used two-proportion z-tests together with confidence intervals for the observed rates.

## Observed Simulation Results

### Latency

- Variant A mean latency: `2561.90 ms`
- Variant B mean latency: `2138.49 ms`
- improvement: `423.41 ms`
- p-value: `1.13e-71`

### Groundedness

- Variant A groundedness: `0.8319`
- Variant B groundedness: `0.8927`
- absolute lift: `0.0608`
- p-value: `2.35e-08`

### Task Success

- Variant A task success: `0.8842`
- Variant B task success: `0.9311`
- absolute lift: `0.0469`
- p-value: `2.95e-07`

### Error Rate

- Variant A error rate: `0.0361`
- Variant B error rate: `0.0379`
- absolute change: `+0.0019`
- p-value: `0.7562`

The error-rate difference is not statistically significant, so the gains in latency and quality are not offset by a meaningful reliability regression in this simulation.

## Decision Rule

I would recommend shipping Variant B if all of the following are true:

1. latency improves in a statistically meaningful way
2. groundedness improves, or at least does not decline
3. the guardrail metric does not show a meaningful regression

Based on the simulated results, Variant B satisfies those conditions.
