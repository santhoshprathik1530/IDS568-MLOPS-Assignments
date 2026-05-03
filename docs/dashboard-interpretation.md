# Dashboard Interpretation

## Monitoring Setup

For this component, I instrumented the final-project service with Prometheus metrics and created a Grafana dashboard definition to visualize the behavior of the RAG and agent endpoints. The service code is in `src/monitoring/service.py`, the collector configuration is in `dashboards/prometheus.yml`, and the dashboard export is in `dashboards/grafana-dashboard.json`.

To populate the dashboard with activity, I ran a local monitoring demo that generated simulated traffic against both endpoints. That run produced the raw metrics snapshot in `logs/monitoring_metrics.prom`, the summary file in `logs/monitoring_summary.json`, and the dashboard image in `screenshots/dashboard-monitoring.png`.

## What the Dashboard Shows

I designed the dashboard around four practical questions that I would want answered if this system were running in production:

1. Is the service handling traffic normally?
2. Where is the request latency coming from?
3. Are there unusual or low-quality inputs entering the system?
4. Is the query mix drifting away from the baseline used during evaluation?

The panels therefore focus on request volume, latency by stage, drift score, integrity anomalies, and agent step counts.

## Observed Results

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

## System Health

The system appears operationally healthy in the narrow sense that it handled the test traffic without failures and emitted telemetry across the full request path. What stands out immediately is that retrieval is relatively inexpensive while generation is responsible for most of the end-to-end latency. That pattern is consistent with the architecture of a small RAG service and also matches what I observed earlier in the milestone evaluations.

The retrieval latency of about `56 ms` is not a concern by itself. The generation latency is the more important signal. An average of about `2.4 s`, with a peak near `5 s`, means that the user experience is dominated by model inference rather than by vector search. If I were optimizing the system, I would start there.

The agent behavior was also stable during the demo. The average of `1.8` steps suggests that most tasks were completed with either a direct retrieval-answer path or retrieval plus one synthesis step. That is desirable for this project because an unconstrained agent loop would make both latency and failure analysis much harder.

## Bottlenecks and Risks

The clearest bottleneck is generation time. Even with a small synthetic workload, model inference dominated all other latency sources. If this service received steadier traffic, generation would be the first place where queueing or throughput issues would appear.

The second issue is grounding quality. Three answers were returned without source citations. I do not treat that as proof that the answers were incorrect, but it is still a useful operational warning sign. In a RAG system, missing citations often correlate with weak retrieval support or overly confident summarization.

The third issue is workload drift. The maximum drift score reached `1.981`, which is close enough to a threshold value that I would watch it carefully. In this implementation, the drift score is only a simple proxy based on query-length deviation, but it still shows how the system can detect that users may be shifting toward a different prompt style.

## Alert Conditions

If I were converting this into an initial production alert policy, I would start with the following conditions:

- p95 answer latency above `4000 ms`
- average generation latency above `3000 ms` over a recent window
- drift score above `2.0`
- repeated integrity anomalies in a short interval
- an increase in responses without source citations

These are not final service-level targets. They are first-pass alert triggers that correspond to the most visible failure modes in this architecture: slow responses, unusual inputs, and weaker grounding.

## Design Justification

I chose a Prometheus and Grafana style setup because it is lightweight, open source, and easy to explain in a course project. More importantly, it let me track both standard service metrics and AI-specific signals in the same place. Request count and latency are necessary, but they are not enough for a retrieval-based system. Integrity anomalies, drift signals, and citation gaps are more helpful when the goal is to diagnose behavior rather than just measure traffic.

## Conclusion

The dashboard suggests that the system is observable and basically stable under the simulated workload, but it also makes the main operational risk obvious: generation latency. Beyond that, the most valuable part of the monitoring design is that it surfaces signals related to grounding quality and workload change, which are exactly the issues that become important in a RAG plus agent workflow.
