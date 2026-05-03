# Dashboard Interpretation

## Monitoring Setup

For Component 1, I instrumented the final-project service with Prometheus metrics and created a starter Grafana dashboard configuration. The instrumented service is implemented in `src/monitoring/service.py`, the Prometheus scrape configuration is stored in `dashboards/prometheus.yml`, and the dashboard definition is stored in `dashboards/grafana-dashboard.json`.

To generate live evidence for this component, I ran a local monitoring demo that sent simulated traffic through the instrumented RAG and agent endpoints. That run produced:

- `logs/monitoring_metrics.prom`
- `logs/monitoring_summary.json`
- `screenshots/dashboard-monitoring.png`

The screenshot file is a dashboard-style export built from the observed traffic in the demo run.

## What the Dashboard Shows

The dashboard focuses on four operational questions:

1. Is the service receiving traffic normally?
2. Where is latency coming from?
3. Are there integrity anomalies in the inputs?
4. Are there early signs that the workload is drifting away from the baseline query mix?

The four panels reflect those questions:

- request volume by endpoint
- answer-endpoint latency, split into retrieval and generation
- rolling query drift score
- integrity anomalies and agent step counts

## Observed Results from the Demo Run

From the simulated run:

- total requests: `24`
- answer requests: `19`
- agent requests: `5`
- average retrieval latency: `56.02 ms`
- average generation latency: `2420.03 ms`
- maximum generation latency: `4970.4 ms`
- average agent step count: `1.8`
- maximum drift score: `1.981`
- integrity anomalies observed: one `multi_question` case
- requests without source citations: `3`

## What This Reveals About System Health

The system looks healthy in the sense that it handled both answer and agent traffic without crashing and produced measurable telemetry across the full request path. The main operational pattern is that retrieval is relatively cheap while generation dominates latency. That is consistent with the Milestone 6 evaluation results and is exactly the kind of bottleneck I would expect in a small RAG service.

The average retrieval latency of roughly `56 ms` is low enough that it would not be the first place I would optimize. The generation step, on the other hand, averaged about `2.4 s` during the monitoring run and peaked at just under `5 s`. That means the user-facing latency budget is driven mainly by the model call, not by vector search.

The agent also behaved in a stable way. The average step count was `1.8`, which means most tasks were resolved either with direct retrieval plus answer generation or with retrieval followed by one synthesis step. That is a good sign because excessive tool chaining would make both latency and debugging harder.

## Bottlenecks and Risks

The clearest bottleneck is generation latency. Even in this short synthetic run, model time dominated everything else. If this service were exposed to more sustained traffic, generation would be the first resource pressure point and the first place where queuing would likely appear.

The second risk is weak grounding. Three answer requests did not contain source citations in the final answer. That does not automatically mean the answer was wrong, but it is a useful operational proxy for low-support or poorly grounded responses. In a production setting, I would treat that as an early warning signal rather than a purely cosmetic issue.

The third risk is workload drift. The maximum drift score in the run was `1.981`, which is not catastrophic, but it is high enough to justify watching. In this project, the drift score is based on rolling query-length deviation from the evaluation baseline. It is a simple proxy rather than a full drift framework, but it is enough to show how unusual traffic could be detected early.

## Suggested Alert Thresholds

If I were turning this into a production alert policy, I would start with:

- p95 request latency above `4000 ms` for the answer endpoint
- average generation latency above `3000 ms` over a recent time window
- drift score above `2.0`
- repeated integrity anomalies in a short window
- an increase in requests without source citations

These thresholds are intentionally conservative. They are not meant to be final SLOs, but they are reasonable first-pass alerts for this type of service because they point to user-visible degradation, unusual traffic, or reduced grounding quality.

## Design Justification

I chose Prometheus-style instrumentation because it keeps the metrics explicit and easy to inspect in Python. The dashboard configuration is lightweight, open source, and matches the course requirement to avoid proprietary monitoring services.

I also included both conventional service metrics and AI-specific signals. Request counts and latency are necessary but not sufficient for an LLM system. Integrity anomalies, weak-citation signals, and drift indicators are more useful for diagnosing AI-specific failure modes such as prompt quality changes, unusual user behavior, or degraded grounding.

## Takeaway

The dashboard suggests that the system is functional and observable, but the main performance risk is still model latency. The monitoring design is useful because it does more than report raw speed: it also surfaces integrity and grounding indicators that would matter if this RAG + agent workflow were used in a real deployment.
