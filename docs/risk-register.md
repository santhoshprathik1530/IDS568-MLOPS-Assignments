# Risk Register

| Risk ID | Category | Description | Likelihood | Impact | Mitigation |
|---|---|---|---|---|---|
| R1 | Bias | The small synthetic corpus may overrepresent the assumptions built into the course materials and underrepresent alternative perspectives. | Medium | Medium | Keep the use case narrow, document corpus scope clearly, and avoid presenting outputs as domain-complete truth. |
| R2 | Robustness | Retrieval can return only partially relevant context, leading to incomplete answers that still look plausible. | High | Medium | Monitor retrieval confidence, track low-support responses, and require citations in answer formatting. |
| R3 | Robustness | LLM latency spikes can degrade user experience and make monitoring noisy. | Medium | Medium | Track generation latency separately from retrieval latency and alert on sustained p95 degradation. |
| R4 | Privacy | If users submit sensitive or personal information in prompts, the service could transmit that content to the external inference provider. | Medium | High | Limit use to non-sensitive data, document the external API path clearly, and add prompt hygiene guidance. |
| R5 | Compliance | The project uses API-based inference with instructor approval, but that path may not meet stricter data residency or regulated deployment rules. | Low | High | Record the approved exception in the README and governance docs; do not treat this setup as a general production pattern. |
| R6 | Governance | Missing citations in final answers can reduce transparency and make review harder. | Medium | Medium | Track citation failures in monitoring and use stronger citation-oriented prompts. |
| R7 | Tool Misuse | The agent may select the wrong synthesis tool if tasks become more open-ended than the current policy expects. | Medium | Medium | Keep the tool set narrow, constrain allowed actions by stage, and log every step for auditability. |
| R8 | Knowledge Freshness | The local corpus can become stale because it is manually maintained and not refreshed automatically. | High | Medium | Add corpus refresh checks and document a re-indexing workflow in the audit trail. |
| R9 | Security | Prompt-like or malformed inputs could stress the retrieval/generation workflow or bypass intended behavior. | Medium | Medium | Track integrity anomalies, reject clearly malformed inputs, and monitor abnormal traffic patterns. |
| R10 | Operational | Drift in query style or complexity could cause the monitored metrics to degrade over time even if the code has not changed. | Medium | Medium | Track rolling drift signals and tie drift alerts to review or retraining/reindexing actions. |
