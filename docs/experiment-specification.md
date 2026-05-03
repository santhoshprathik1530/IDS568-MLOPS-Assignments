# Experiment Specification

## Objective

For this component, I designed an A/B test around a realistic product question for the final-project system: should I keep the current RAG + agent workflow as it is, or move to a slightly tighter and more citation-focused variant?

The two arms are:

- **Variant A:** current system
  - top-k retrieval of 3
  - current answer format
- **Variant B:** proposed variant
  - top-k retrieval of 2
  - more citation-focused answer formatting

The basic idea behind Variant B is that a slightly smaller retrieval context may reduce latency, and a more explicit citation style may improve groundedness.

## Hypothesis

Primary hypothesis:

> Variant B will reduce average response latency while maintaining or improving answer groundedness.

Secondary hypothesis:

> Variant B will improve overall task success without materially increasing the error rate.

## Success Metrics

Primary metrics:

- average end-user latency
- groundedness rate

Secondary metrics:

- task success rate
- error rate

The guardrail metric is error rate. Even if Variant B is faster, I would not recommend it if failures increase noticeably.

## Randomization Method

Requests are assigned to A or B with a deterministic 50/50 split based on a hash of the request ID. I used deterministic assignment because it is easy to reproduce and keeps the simulation logic simple and auditable.

## Sample Size and Duration

I based the sample-size calculation on groundedness, using a baseline rate of `0.84` and a target improvement to `0.90`, with:

- significance level: `0.05`
- power: `0.80`

The script calculates an approximate required sample size of:

- **492 requests per arm**

For the actual simulation run, I used a much larger total of `4000` requests:

- baseline: `1969`
- variant: `2031`

That gives more than enough synthetic data to test the decision logic and confidence intervals.

If this were converted to a live rollout and the service received about 300 eligible requests per hour, a 984-request total minimum would correspond to roughly 3.3 hours of traffic. In practice, I would run longer to cover day-part effects and watch the guardrail metrics.

## Statistical Evaluation

For latency, I used Welch’s two-sample t-test and confidence intervals around the mean.

For groundedness, task success, and error rate, I used two-proportion z-tests and confidence intervals for the observed rates.

## Observed Simulation Results

### Latency

- A mean latency: `2561.90 ms`
- B mean latency: `2138.49 ms`
- improvement: `423.41 ms`
- p-value: `1.13e-71`

### Groundedness

- A groundedness: `0.8319`
- B groundedness: `0.8927`
- absolute lift: `0.0608`
- p-value: `2.35e-08`

### Task Success

- A task success: `0.8842`
- B task success: `0.9311`
- absolute lift: `0.0469`
- p-value: `2.95e-07`

### Error Rate

- A error rate: `0.0361`
- B error rate: `0.0379`
- absolute change: `+0.0019`
- p-value: `0.7562`

The simulated error-rate difference is not statistically significant, so the speed and quality gains are not offset by a meaningful reliability penalty in this run.

## Decision Rule

I would recommend shipping Variant B if:

1. latency improves significantly
2. groundedness improves or at least does not regress
3. error rate does not show a meaningful increase

Based on the observed simulation output, Variant B meets those conditions.
